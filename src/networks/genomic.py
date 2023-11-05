# Fusing Histology and Genomics via Deep Learning - IEEE TMI 
# Copyright (C) 2020  Mahmood Lab 

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses

# https://www.sciencedirect.com/science/article/pii/S089561112200146X
# Generally speaking, genomic data are high-dimensional vectors, usually composed of thousands of values. For learning hundreds to thousands of values with relatively few training samples, feedforward networks are prone to overfitting. We adopt the Self-Normalizing Networks (SNN) (Chen et al., 2020) (Fig. 3) to mitigate overfitting on high-dimensional feature vectors. The genomic data is firstly converted into form through a fully-connected layer with 2560 nodes. Our network architecture consists of one fully-connected layer followed by Exponential Linear Unit (ELU) activation and Alpha Dropout, and the same gated attention mechanism as in the pathology module to aggregate the genomic features.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable
from torch.nn import init, Parameter
from torch.utils.data import DataLoader
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler


def init_max_weights(module):
    # https://github.com/mahmoodlab/PathomicFusion/blob/44b5513cb90c31337276bba92abe515e21c7f6ca/utils.py#L225
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


class SNNet(nn.Module):
    # https://github.com/mahmoodlab/PathomicFusion/blob/44b5513cb90c31337276bba92abe515e21c7f6ca/networks.py#L166
    def __init__(
        self,
        input_dim=80,
        feature_size=32,
        dropout_rate=0.25,
        act=None,
        label_dim=1,
        elu=False,
        init_max=True,
    ):
        super(SNNet, self).__init__()
        hidden = [500, 250, 125, 125]
        self.act = act

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ELU() if elu else nn.SELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),
        )

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU() if elu else nn.SELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),
        )

        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU() if elu else nn.SELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),
        )

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], feature_size),
            nn.ELU() if elu else nn.SELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),
        )

        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Linear(feature_size, label_dim)

        self.encoderX = nn.Sequential(
            nn.Linear(input_dim, feature_size),
            nn.ELU() if elu else nn.SELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),
        )

        if init_max:
            init_max_weights(self)

    def forward(self, x):
        features = self.encoder(x)
        hazard = self.classifier(features)

        return features, hazard


class simpleNet(nn.Module):
    #
    def __init__(
        self,
        input_dim=80,
        feature_size=32,
        dropout_rate=0.25,
        act=None,
        label_dim=1,
        elu=True,
        init_max=True,
    ):
        super(simpleNet, self).__init__()
        hidden = [335, 250, 125, 125]
        self.act = act

        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.SELU(inplace=True) if elu else nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=False),
        )

        # encoder2 = nn.Sequential(
        #     nn.Linear(hidden[0], hidden[1]),
        #     nn.ELU() if elu else nn.SELU(),
        #     nn.AlphaDropout(p=dropout_rate, inplace=False),
        # )

        # encoder3 = nn.Sequential(
        #     nn.Linear(hidden[1], hidden[2]),
        #     nn.ELU() if elu else nn.SELU(),
        #     nn.AlphaDropout(p=dropout_rate, inplace=False),
        # )

        # encoder4 = nn.Sequential(
        #     nn.Linear(hidden[2], feature_size),
        #     nn.ELU() if elu else nn.SELU(),
        #     nn.AlphaDropout(p=dropout_rate, inplace=False),
        # )

        # self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Linear(hidden[0], label_dim)

        if init_max:
            init_max_weights(self)

    def forward(self, x):
        features = self.encoder1(x)
        hazard = self.classifier(features)

        return features, hazard
