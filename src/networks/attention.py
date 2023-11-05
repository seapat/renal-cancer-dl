"""Attention-based multimodal fusion.
https://github.com/luisvalesilva/multisurv/blob/deb35e12502f61a59726bfb543210735406d5959/src/attention.py#L1
"""

import torch


class Attention(torch.nn.Module):
    """Attention mechanism for multimodal representation fusion."""
    def __init__(self, size):
        """
        Parameters
        ----------
        size: int
            Attention vector size, corresponding to the feature representation
            vector size.
        """
        super(Attention, self).__init__()
        self.fc = torch.nn.Linear(size, size, bias=False)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(0)  # Across feature vector stack

    def _scale_for_missing_modalities(self, x, out):
        """Scale fused feature vector up according to missing data.

        If there were all-zero data modalities (missing/dropped data for
        patient), scale feature vector values up accordingly.
        """
        batch_dim = x.shape[1]
        for i in range(batch_dim):
            patient = x[:, i, :]
            zero_dims = 0
            for modality in patient:
                if modality.sum().data == 0:
                    zero_dims += 1

            if zero_dims > 0:
                scaler = zero_dims + 1
                out[i, :] = scaler * out[i, :]

        return out

    def forward(self, x):
        scores = self.tanh(self.fc(x))
        weights = self.softmax(scores)
        out = torch.sum(x * weights, dim=0)

        out = self._scale_for_missing_modalities(x, out)

        return out
