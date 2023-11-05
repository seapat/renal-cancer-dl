# 
# MIT License
# 
# Copyright (c) 2020 Luís
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


'''SOURCE
@article{CHOI2019259,
title = {EmbraceNet: A robust deep learning architecture for multimodal classification},
journal = {Information Fusion},
volume = {51},
pages = {259-270},
year = {2019},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2019.02.010},
url = {https://www.sciencedirect.com/science/article/pii/S1566253517308242},
author = {Jun-Ho Choi and Jong-Seok Lee},
keywords = {Multimodal data fusion, deep learning, classification, data loss},
abstract = {Classification using multimodal data arises in many machine learning applications. It is crucial not only to model cross-modal relationship effectively but also to ensure robustness against loss of part of data or modalities. In this paper, we propose a novel deep learning-based multimodal fusion architecture for classification tasks, which guarantees compatibility with any kind of learning models, deals with cross-modal information carefully, and prevents performance degradation due to partial absence of data. We employ two datasets for multimodal classification tasks, build models based on our architecture and other state-of-the-art models, and analyze their performance on various situations. The results show that our architecture outperforms the other multimodal fusion architectures when some parts of data are not available.}
}
'''
import torch


class EmbraceNet(torch.nn.Module):
    """Embracement modality feature aggregation layer."""
    def __init__(self, device='cuda:0'):
        """Embracement modality feature aggregation layer.

        Note: EmbraceNet needs to deal with mini batch elements differently
        (check missing data and adjust sampling probailities accordingly). This
        way, we take the unusual measure of considering the batch dimension in
        every operation.

        Parameters
        ----------
        device: "torch.device" object
            Device to which input data is allocated (sampling index tensor is
            allocated to the same device).
        """
        super(EmbraceNet, self).__init__()
        self.device = device

    def _get_selection_probabilities(self, d, batch_size:int):
        p = torch.ones(d.size(0), batch_size)  # Size modalities x batch

        # Handle missing data
        for i, modality in enumerate(d):
            for j, batch_element in enumerate(modality):
                if len(torch.nonzero(batch_element)) < 1:
                    p[i, j] = 0

        # Equal chances to all available modalities in each mini batch element
        m_vector = torch.sum(p, dim=0)
        p /= m_vector

        return p

    def _get_sampling_indices(self, p, c, m):
        r = torch.multinomial(
            input=p.transpose(0, 1), num_samples=c, replacement=True)
        r = torch.nn.functional.one_hot(r.long(), num_classes=m)
        r = r.permute(2, 0, 1)

        return r


    def forward(self, x):
        m, b, c = x.size()

        p = self._get_selection_probabilities(x, b)
        r = self._get_sampling_indices(p, c, m).float().to(self.device)

        d_prime = r * x
        e = d_prime.sum(dim=0)

        return e