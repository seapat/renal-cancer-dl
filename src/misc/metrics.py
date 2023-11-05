import torch
from torch import Tensor
import numpy as np
from torchmetrics import Metric
from sortedcontainers import SortedList

from torch import nn

def concordance_index(risk, time, event)-> float:
    """
    O(n log n) implementation of https://square.github.io/pysurvival/metrics/c_index.html

    risk: risk score, predicted by the model
    time: time of event or censoring in days (unit should not matter?)
    event: 1 if uncensored, 0 otherwise

    Taken from Lassonet (Lemhadri et al.; 2021)
    https://github.com/lasso-net/lassonet/blob/e12ca78347a49af00b33aab4dba098d11634b742/lassonet/cox.py#LL79C8-L79C8
    """

    assert (
        len(risk) == len(time) == len(event)
    ), f"inputs have unequal lengths, risk: {len(risk)},    time: {len(time)}, event: {len(event)}"
    
    n = len(risk)
    order = sorted(range(n), key=time.__getitem__)
    past = SortedList()
    num = 0
    den = 0
    for i in order:
        num += len(past) - past.bisect_right(risk[i])
        den += len(past)
        if event[i]:
            past.add(risk[i])
    return num / den #if den != 0 else torch.nan

def brier_score(y_true: torch.Tensor, y_prob: torch.Tensor):
    '''Implementation of the Brier Score
    y_true: ground truth labels (0 or 1)
    y_prob: predicted probabilities between 0 and 1
    '''
    assert y_true.ndim == 1 and y_prob.ndim == 1
    assert len(y_true) == len(y_prob)
    assert ((y_true == 0.0) | (y_true== 1.0)).all(), f"The truth tensor can only contain 0.0 or 1.0, y_true: {y_true}"

    # Calculate squared differences between y_true and y_prob
    squared_diffs = torch.square(y_prob - y_true)

    # Calculate mean squared difference
    mean_squared_diff = torch.nanmean(squared_diffs)

    return mean_squared_diff

class ConcordanceIndex(nn.Module):
    def __init__(self):
        super().__init__()
        # self._flush_state()

    def _flush_state(self, device):
        self.risks = torch.tensor([], device=device)
        self.times = torch.tensor([], device=device)
        self.events = torch.tensor([], device=device)

    def update(self, risks, times, events):
        assert risks.ndim == 1 and  times.ndim == 1 and events.ndim == 1, f"inputs have unequal shapes, risk: {risks.shape}, time: {times.shape}, event: {events.shape}"
        assert risks.shape == times.shape == events.shape, f"inputs have unequal shapes, risk: {risks.shape}, time: {times.shape}, event: {events.shape}"
        assert risks.device == times.device == events.device, f"inputs have unequal devices, risk: {risks.device}, time: {times.device}, event: {events.device}"
        self._flush_state(risks.device)

        print(f"risks: {risks} - {risks.shape}")
        print(f"self.risks: {self.risks} - {self.risks.shape}")
        print(f"times: {times} - {times.shape}")
        print(f"self.times: {self.times} - {self.times.shape}")
        print(f"events: {events} - {events.shape}")
        print(f"self.events: {self.events} - {self.events.shape}")
        
        self.risks = torch.cat((self.risks, risks)).flatten()
        self.times = torch.cat((self.times, times)).flatten()
        self.events = torch.cat((self.events, events)).flatten()

    def compute(self):

        

        c_idx = self(self.risks, self.times, self.events)
        self._flush_state(self.risks.device)

        return c_idx

    def forward(self, risk, time, event) -> Tensor:
        assert  ( len(risk) == len(time) == len(event)
        ), f"inputs have unequal lengths, risk: {len(risk)}, time: {len(time)}, event: {len(event)}"

        n = len(risk)
        order: list[int] = sorted(range(n), key=time.__getitem__)
        past: SortedList = SortedList()
        num: int = 0
        den: int = 0
        for i in order:
            num += len(past) - past.bisect_right(risk[i])
            den += len(past)
            if event[i]:
                past.add(risk[i])
        return torch.tensor(num / den) if den != 0 else torch.tensor(torch.nan)

class ConcordanceMetric(Metric):
    def __init__(self):
        super().__init__()
        # self.add_state("risk", default=[], dist_reduce_fx="cat")

        self.add_state("risk", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("time", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("event", default=torch.tensor([]), dist_reduce_fx="cat")
        self.time: Tensor
        self.risk: Tensor
        self.event: Tensor

    def update(self, risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor):
        assert (
        len(risk) == len(time) == len(event)
        ), f"inputs have unequal lengths, risk: {len(risk)}, time: {len(time)}, event: {len(event)}"

        self.risk = torch.cat((self.risk, risk.flatten())).flatten()
        self.time = torch.cat((self.time, time.flatten())).flatten()
        self.event = torch.cat((self.event, event.flatten())).flatten()

    def compute(self):

        risk = self.risk.view(-1)
        time = self.time.view(-1)
        event = self.event.view(-1)

        n: int = len(risk)
        
        order = sorted(range(n), key=time.__getitem__) # type: ignore
        past = SortedList()
        num = 0
        den = 0
        for i in order:
            num += len(past) - past.bisect_right(risk[i])
            den += len(past)

            if event[i]:
                past.add(risk[i])

        return torch.tensor(num / den) #if den != 0 else torch.tensor(torch.nan) 


'''TODO
As a performance measure for assessment of the model, the c-statistic was used. The c-statistic indicates the discriminative power of a regression model. The approach of Harrell et al. was used to calculate the c-statistic [32], which is the preferred approach for studies focusing on long term risk prediction and in which not all individuals experience the event of interest [33]. As an additional experiment, we analysed the results of AMC on the total cohort of 597 TNBC tumours, applying the procedure described above.

32: F.E. Harrell, K.L. Lee, D.B. Mark, Tutorial in biostatistics: Multivariable prognostic models: Issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. Stat. Med. 15, 361–387 (1996).
33: M.J. Pencina, R.B. D’Agostino, L. Song, Quantifying discrimination of Framingham risk functions with different survival C statistics. Stat. Med. 31, 1543–1553 (2012)
'''
