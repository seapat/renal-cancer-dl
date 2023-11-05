import logging
import sys

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from datasets.base_dataset import dataset_globber
from networks.autoencoder import ResNet_Decoder, ResNet_Encoder
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelSummary,
    StochasticWeightAveraging,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_finder import LearningRateFinder
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch import nn
from torch.autograd import Variable

from src.datasets.base_dataset import dataset_globber
from src.lightning.data_modules import PatchDataModule

from src.misc.variables import annos_of_interest

pl.seed_everything(1234)

class LitNanostringSNN(pl.LightningModule):
    pass