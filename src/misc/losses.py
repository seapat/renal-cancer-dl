import torch
from torch import Tensor
import numpy as np
from torchmetrics import Metric
from torch import nn

def kl_weight_scheduler(epoch, num_epochs, initial_weight=0.0, final_weight=1.0):
    return min((epoch + 1) / num_epochs, 1.0) * final_weight + initial_weight

'''
combined_loss_fn = CombinedLoss(alpha=0.8, beta=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    kl_weight = kl_weight_scheduler(epoch, num_epochs, initial_weight=0.0, final_weight=1.0)
    for i, (image, seg_mask) in enumerate(dataloader):
        optimizer.zero_grad()
        recon_image, mu, logvar = model(image)
        loss = combined_loss_fn(image, recon_image, mu, logvar, kl_weight=kl_weight)
        loss.backward()
        optimizer.step()
'''

class CombinedVAELoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedVAELoss, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x, x_recon, mu, logvar, kl_weight=1.0):
        bce_loss = nn.BCELoss(reduction='mean')(x, x_recon)
        mse_loss = nn.MSELoss(reduction='mean')(x, x_recon)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        combined_loss = self.alpha * bce_loss + self.beta * mse_loss + kl_weight * kl_div
        return combined_loss

def coxnnetLoss(hazard_pred, survtime, censor, device) -> Tensor:
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat: torch.Tensor | np.ndarray = np.zeros(
        [current_batch_len, current_batch_len], dtype=int
    )
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.as_tensor(R_mat)
    theta = hazard_pred.reshape(-1)
    exp_theta: Tensor = torch.exp(theta)
    loss_cox: Tensor = -torch.mean(
        (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor,
    )
    return loss_cox

class CoxNNetLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, hazard_pred, durations:Tensor, events:Tensor) -> Tensor:
        # durations: Tensor
        # events: Tensor
        # # durations, events = y.T
        # durations, events = y

        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(durations)
        R_mat: torch.Tensor | np.ndarray = np.zeros(
        [current_batch_len, current_batch_len], dtype=int
        )
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = durations[j] >= durations[i]

        R_mat = torch.as_tensor(R_mat)#, device=hazard_pred.device)
        theta = hazard_pred.reshape(-1)
        exp_theta: Tensor = torch.exp(theta)
        loss_cox: Tensor = -torch.mean(
        (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * events,
        )
        return loss_cox #.to( device=hazard_pred.device)

class FastCPH(torch.nn.Module):
    """Loss for CoxPH model. 
    
    Implementation from Lassonet/fastCPH with Efron Tie handling thrown out since we do not expect many ties in our data.
    Thus Breslow is assumed to be sufficient.
    https://github.com/lasso-net/lassonet/blob/e12ca78347a49af00b33aab4dba098d11634b742/lassonet/cox.py#L13"""

    def __init__(self):
        super().__init__()

    def forward(self, log_h:Tensor, y:tuple[Tensor, Tensor], eps=1e-10) -> Tensor:
        log_h = log_h.flatten() #+ eps

        durations, events = y #y.T

        # if events.sum() == 0:
        #     # reason see below
        #     return torch.tensor(0.0, device=log_h.device, requires_grad=True)

        # sort input
        durations, idx = durations.sort(descending=True)
        log_h = log_h[idx]
        events = events[idx]

        event_ind = events.nonzero().flatten()

        # numerator
        log_num = log_h[event_ind].nanmean()

        # logcumsumexp of events
        event_lcse = torch.logcumsumexp(log_h, dim=0)[event_ind]

        # number of events for each unique risk set
        _, tie_inverses, tie_count = torch.unique_consecutive(
            durations[event_ind], return_counts=True, return_inverse=True
        )

        # position of last event (lowest duration) of each unique risk set
        tie_pos = tie_count.cumsum(axis=0) - 1

        # logcumsumexp by tie for each event
        event_tie_lcse = event_lcse[tie_pos][tie_inverses]

        # breslow tie handling
        log_den = event_tie_lcse.nanmean()

        # neg_log_likelihood = log_den - log_num

        # loss is negative log likelihood
        #  Why nan_to_num?
        #   When all samples are censored, there is no information about the hazard rate for the samples that did not experience an event. 
        #   Therefore, the likelihood of observing the censored data is equal to the probability of not observing the event, 
        #   which is equal to one. Taking the logarithm of this likelihood function, the result is zero. 
        #   Thus, the loss function should return a value of zero when all samples are censored.
        return torch.nan_to_num(log_den - log_num) # log_den - log_num # neg_log_likelihood if not neg_log_likelihood.isnan().any() else torch.tensor(0.0, requires_grad=True)

class CoxNNetLossMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("risk", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=[], dist_reduce_fx="cat")
        self.add_state("event", default=[], dist_reduce_fx="cat")
        self.risks: list
        self.times: list
        self.events: list


    def update(self, risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor):
        assert (
        len(risk) == len(time) == len(event)
        ), f"inputs have unequal lengths, risk: {len(risk)}, time: {len(time)}, event: {len(event)}"

        self.risks.extend(risk)
        self.times.extend(time)
        self.events.extend(event)

    def compute(self):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(self.times)
        R_mat: torch.Tensor | np.ndarray = np.zeros(
        [current_batch_len, current_batch_len], dtype=int
        )
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = self.times[j] >= self.times[i]

        R_mat = torch.as_tensor(R_mat)
        theta = self.risks.reshape(-1)
        exp_theta: Tensor = torch.exp(theta)
        loss_cox: Tensor = -torch.mean(
        (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * self.events,
        )
        return loss_cox
    
# CoxPH loss as a torchmetric
class CoxPHMetric(Metric):
    def __init__(self):
        super().__init__()        
        self.times: Tensor
        self.risks: Tensor
        self.events: Tensor
        self.add_state("risks", default=torch.tensor([], requires_grad=True), dist_reduce_fx=None)
        self.add_state("times", default=torch.tensor([], requires_grad=True), dist_reduce_fx=None)
        self.add_state("events", default=torch.tensor([], requires_grad=True), dist_reduce_fx=None)

    def update(self, risks: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> None:
        assert (
        len(risks) == len(risks) == len(risks)
        ), f"inputs have unequal lengths, risk: {len(risks)}, time: {len(times)}, event: {len(events)}"

        # print(f"risks: {risks.shape}, times: {times.shape}, events: {events.shape}")
        # print(f"risks: {risks}, times: {times}, events: {events}")

        # print(f"self.risks: {self.risks.shape}, self.times: {self.times.shape}, self.events: {self.events.shape}")
        # print(f"self.risks: {self.risks}, self.times: {self.times}, self.events: {self.events}")

        self.risks = torch.cat((self.risks, risks)).flatten()
        self.times = torch.cat((self.times, times),).flatten()
        self.events = torch.cat((self.events, events),).flatten()

    def compute(self)-> Tensor: 
        idx: torch.LongTensor

        durations, idx = self.times.sort(descending=True)
        log_h = self.risks[idx]
        events = self.events[idx]

        event_ind = events.nonzero().flatten()

        # Hack if events only contains censored (=0) events
        # if len(event_ind) == 0:
        #     return torch.tensor(torch.nan, device=log_h.device)

        # numerator
        log_num = log_h[event_ind].nanmean()

        # logcumsumexp of events
        event_lcse: Tensor = torch.logcumsumexp(log_h, dim=0)[event_ind]

        # number of events for each unique risk set
        _, tie_inverses, tie_count = torch.unique_consecutive(
            durations[event_ind], return_counts=True, return_inverse=True
        )

        # position of last event (lowest duration) of each unique risk set
        tie_pos = tie_count.cumsum(axis=0) - 1

        # logcumsumexp by tie for each event
        event_tie_lcse: Tensor = event_lcse[tie_pos][tie_inverses]

        # breslow tie handling:
        log_den: Tensor = event_tie_lcse.nanmean()

        return log_den - log_num
