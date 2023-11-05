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

"""SOURCE
Vale-Silva, L. A., & Rohr, K. (2021). 
Long-term cancer survival prediction using multimodal deep learning. 
Scientific reports, 11(1), 13505. 
https://doi.org/10.1038/s41598-021-92799-4

"""

import torch
from torch import nn
from src.networks.kronecker_attention import KroneckerFusion
from src.networks.embrace import EmbraceNet
from src.networks.genomic import SNNet
from src.networks.pathology import CoxResNet
from src.networks.attention import Attention
import warnings
from bisect import bisect_left

import torch


class Fusion(nn.Module):
    "Multimodal data aggregator."

    def __init__(self, method, feature_size, device):
        super(Fusion, self).__init__()
        self.method = method
        methods = ["cat", "max", "sum", "prod", "embrace", "attention", "kronecker"]

        if self.method not in methods:
            raise ValueError('"method" must be one of ', methods)

        if self.method == "embrace":
            if device is None:
                raise ValueError('"device" is required if "method" is "embrace"')

            self.embrace = EmbraceNet(device=device)

        if self.method == "kronecker":
            if feature_size is None:
                raise ValueError(
                    '"feature_size" is required if "method" is "kronecker"'
                )
            self.kronecker = KroneckerFusion(
                dim1=feature_size,
                scale_dim1=4,
                dim2=feature_size,
                scale_dim2=4,
                feature_size=feature_size,
            )

        if self.method == "attention":
            if not feature_size:
                raise ValueError(
                    '"feature_size" is required if "method" is "attention"'
                )
            self.attention = Attention(size=feature_size)

    def forward(self, x):
        match self.method:
            case "attention":
                return self.attention(x)
            case "cat":
                return torch.cat([m for m in x], dim=1)
            case "max":
                return x.max(dim=0)[0] 
            case "sum":
                return x.sum(dim=0) # maybe divide by the number of non-zero modalities to apply some value scaling
            case "prod":
                return x.prod(dim=0) # maybe we should set zero vectors to ones first
            case "embrace":
                return self.embrace(x)
            case "kronecker":
                return self.kronecker(x)

        raise ValueError(f"Unknown fusion method, got: {self.method}")


class MultiSurv(torch.nn.Module):
    """Deep Learning model for MULTImodal pan-cancer SURVival prediction."""

    def __init__(
        self, data_modalities: list[str], fusion_method="max", modality_feature_size=1000, device=None
    ):
        super(MultiSurv, self).__init__()
        self.data_modalities = data_modalities
        self.modality_feature_size = modality_feature_size
        valid_mods = ["rcc", "wsi", "mRNA", "miRNA", "DNAm", "CNV"]
        assert all(
            mod in valid_mods for mod in data_modalities
        ), f"Accepted input data modalitites are: {valid_mods}"

        assert len(data_modalities) > 0, "At least one input must be provided."

        if fusion_method == "cat":
            self.num_features = 0
        else:
            self.num_features = self.modality_feature_size

        self.submodels = {}

        # --------------------------------------------------------------------------------
        if "rcc" in self.data_modalities:
            self.nanostring_submodel = SNNet(
                input_dim=750,
                feature_size=self.modality_feature_size,
                label_dim=self.modality_feature_size,
                elu=False, init_max=False, dropout_rate=0.25
            )
            self.submodels["rcc"] = self.nanostring_submodel

            if fusion_method == "cat":
                self.num_features += self.modality_feature_size

        # --------------------------------------------------------------------------------
        if "wsi" in self.data_modalities:
            self.wsi_submodel = CoxResNet(
                input_channels=8, feature_size=self.modality_feature_size
            )
            self.submodels["wsi"] = self.wsi_submodel

            if fusion_method == "cat":
                self.num_features += self.modality_feature_size

        # --------------------------------------------------------------------------------
        # Instantiate multimodal aggregator
        if len(data_modalities) > 1:
            self.aggregator = Fusion(fusion_method, self.modality_feature_size, device)
        else:
            if fusion_method is not None:
                warnings.warn("Input data is unimodal: no fusion procedure.")

        # --------------------------------------------------------------------------------
        # Fully-connected and risk layers
        n_fc_layers = 1
        n_neurons = self.modality_feature_size

        self.fc_block = FC(self.num_features, n_neurons, n_fc_layers)

        self.risk_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_neurons, out_features=1), # torch.nn.Sigmoid()
        )

    def forward(self, x):
        multimodal_features = tuple()

        # Run data through modality sub-models (generate feature vectors) ----#
        for modality in x:
            # get feature representations [0] from subodels
            multimodal_features += (self.submodels[modality](x[modality])[0],)

        # Feature fusion/aggregation -----------------------------------------#
        if len(multimodal_features) > 1:
            x = self.aggregator(torch.stack(multimodal_features))
            feature_repr = {"modalities": multimodal_features, "fused": x}
        else:  # skip if running unimodal data
            x = multimodal_features[0]
            feature_repr = {"modalities": multimodal_features[0]}

        # Outputs ------------------------------------------------------------#
        x = self.fc_block(x)
        risk = self.risk_layer(x)

        # Return non-zero features (not missing input data)
        output_features = tuple()

        for modality in multimodal_features:
            # print(f"modality {modality.shape} \n {modality}")

            modality_features = torch.stack(
                [batch_element for batch_element in modality]
            )  # if batch_element.sum() != 0
            output_features += (modality_features,)

        feature_repr["modalities"] = output_features

        return feature_repr, risk


class FC(nn.Module):
    "Fully-connected model to generate final output."

    def __init__(
        self,
        in_features,
        out_features,
        n_layers,
        dropout=True,
        batchnorm=False,
        scaling_factor=4,
    ):
        super(FC, self).__init__()

        layers = nn.ModuleList()
        if n_layers == 1:
            layers = self._make_layer(in_features, out_features, dropout, batchnorm)
        elif n_layers > 1:
            n_neurons = self._pick_n_neurons(in_features)
            if n_neurons < out_features:
                n_neurons = out_features

            if n_layers == 2:
                layers = self._make_layer(
                    in_features, n_neurons, dropout, batchnorm=True
                )
                layers += self._make_layer(n_neurons, out_features, dropout, batchnorm)
            else:
                for layer in range(n_layers):
                    last_layer_i = range(n_layers)[-1]

                    if layer == 0:
                        n_neurons *= scaling_factor
                        layers = self._make_layer(
                            in_features, n_neurons, dropout, batchnorm=True
                        )
                    elif layer < last_layer_i:
                        n_in = n_neurons
                        n_neurons = self._pick_n_neurons(n_in)
                        if n_neurons < out_features:
                            n_neurons = out_features
                        layers += self._make_layer(
                            n_in, n_neurons, dropout, batchnorm=True
                        )
                    else:
                        layers += self._make_layer(
                            n_neurons, out_features, dropout, batchnorm
                        )
        else:
            raise ValueError('"n_layers" must be positive.')

        self.fc = nn.Sequential(*layers)

    def _make_layer(
        self, in_features, out_features, dropout, batchnorm
    ) -> nn.ModuleList:
        layer = nn.ModuleList()
        if dropout:
            layer.append(nn.Dropout(p=0.25))
        layer.append(nn.Linear(in_features, out_features))
        layer.append(nn.ReLU(inplace=True))
        if batchnorm:
            layer.append(nn.BatchNorm1d(out_features))

        return layer

    def _pick_n_neurons(self, n_features):
        # Pick number of features from list immediately below n input
        n_neurons = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        idx = bisect_left(n_neurons, n_features)

        return n_neurons[0 if idx == 0 else idx - 1]

    def forward(self, x):
        return self.fc(x)
