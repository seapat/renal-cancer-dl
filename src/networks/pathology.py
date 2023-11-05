from pytorch_lightning import LightningModule
import torch
from torch import Tensor, nn
from torchvision import models, ops


class CoxEffNet(nn.Module):
    def __init__(
        self, input_channels: int, feature_size: int, act: nn.Module | None = None, weights: str = "DEFAULT", 
    ):
        super(CoxEffNet, self).__init__()
        self.act = act
        self.effnet = models.efficientnet_v2_s(weights)
        self.effnet.classifier = nn.Identity()  # type: ignore

        # reshape input channels
        self.downsample = ops.Conv2dNormActivation(
            input_channels, 3, kernel_size=1, bias=False
        )

        # reshape output channels
        self.last_conv = ops.Conv2dNormActivation(
            256, feature_size, kernel_size=1, bias=False
        )
        self.effnet.features[-1] = self.last_conv

        # custom classifier, called separately
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(feature_size, 1),  # type: ignore
        )

    def forward(self, x):
        x = self.downsample(x)
        features = self.effnet(x)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)
        return features, hazard

class CoxResNet(nn.Module):
    def __init__(
        self,
        input_channels,
        feature_size: int = 1,
        act=None,
        weights: str = "DEFAULT",
    ):
        super(CoxResNet, self).__init__()
        self.act = act
        self.resnet = models.resnet101(weights=weights)

        self.non_linear_downsample = nn.Sequential(
            nn.Conv2d(input_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(
                3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )

        self.feature_extractor = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.resnet.fc.in_features, feature_size)
        )
        

        self.classifier = nn.Linear(feature_size, 1)
        self.resnet.fc = nn.Identity()  # type: ignore


    def forward(self, x: Tensor):

        x = self.non_linear_downsample(x)
        x = self.resnet(x)
        features = self.feature_extractor(x)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)  # type: ignore
        return features, hazard

class _CoxResNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        act: nn.Module,
        feature_size: int,
        weights: str = "DEFAULT",
    ):
        super(_CoxResNet, self).__init__()
        self.act = act
        self.resnet = models.resnet101(weights)

        # copy of initial block 
        self.non_linear_downsample = nn.Sequential(
            nn.Conv2d(input_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(
                3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        
        # replace last conv block
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.resnet.fc.in_features, feature_size, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(feature_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        # disable fc layer and use our own separately
        self.classifier = nn.Linear(feature_size, 1) # type: ignore
        self.resnet.fc = nn.Identity()  # type: ignore

    def forward(self, x: Tensor):
        x = self.non_linear_downsample(x)
        x = self.resnet(x)
        features = self.feature_extractor(x)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)  # type: ignore
        return features, hazard