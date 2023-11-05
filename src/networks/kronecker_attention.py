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

import torch
import torch.nn as nn
import math


def init_max_weights(module):
    # https://github.com/mahmoodlab/PathomicFusion/blob/44b5513cb90c31337276bba92abe515e21c7f6ca/utils.py#L225
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


class KroneckerFusion(nn.Module):
    # https://github.com/mahmoodlab/PathomicFusion/blob/44b5513cb90c31337276bba92abe515e21c7f6ca/fusion.py#L6
    def __init__(
        self,
        skip=1,
        use_bilinear=1,
        gate1=1,
        gate2=1,
        dim1=32,
        dim2=32,
        scale_dim1=1,
        scale_dim2=1,
        feature_size=64,
        dropout_rate=0.25,
    ):
        super(KroneckerFusion, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = (
            dim1,
            dim2,
            dim1 // scale_dim1,
            dim2 // scale_dim2,
        )
        skip_dim = dim1 + dim2 + 2 if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim2_og, dim1)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
        )
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim1_og, dim2_og, dim2)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
        )
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1), feature_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(feature_size + skip_dim, feature_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        init_max_weights(self)

    def forward(self, x):
        assert (
            len(x) == 2
        ), f"Input must be a tensor of length 2, got len {len(x)} and shape {x.shape}"

        vec1, vec2 = x.unbind(0)

        # asssert that vec1 and vec2 ar 1d vectors
        # assert (
        #     vec1.ndim == 1 and vec2.ndim == 1
        # ), f"input needs to contain 2 1D vectors, got {vec1.shape} and {vec2.shape}"

        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = (
                self.linear_z1(vec1, vec2)
                if self.use_bilinear
                else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            )
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec1, vec2)
                if self.use_bilinear
                else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            )
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        ### Fusion
        # o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        # o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=self.device, dtype=torch.float)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=self.device, dtype=torch.float)), 1)

        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(
            start_dim=1
        )  # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, o1, o2), 1)
        out = self.encoder2(out)
        return out
