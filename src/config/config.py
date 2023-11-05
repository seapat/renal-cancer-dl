import sys
from dataclasses import dataclass
from typing import Literal, List, Dict, Union

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelSummary,
    StochasticWeightAveraging,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger, WandbLogger

from src.datasets.base_dataset import dataset_globber

@dataclass
class Config:
    # base
    data_dir: str
    ckpt_dir: str
    log_dir: str
    root_dir: str
    
    # Model
    input_size: int
    num_classes: int
    learning_rate: float
    batch_size: int
    max_epochs: int
    min_epochs: int
    freeze: bool
    annos_of_interest: List[str]
    
    # Dataset
    num_workers: int
    read_level: int
    cache_path: str
    input_dicts: List[Dict[str, Union[str, List[str]]]]
    datamodule_kwargs: Dict[str, Union[int, bool]]
    splits: List[float]
    
    # Trainer
    accelerator: str
    strategy: str
    devices: Union[int, List[int]]
    accumulations: int
    precision: Literal["16-mixed", "32", "bf16-mixed"]
    log_frequency: int
    sync_batch_norm: bool
    enable_progress_bar: bool
    num_nodes: int
    overfit_batches: float
    profiler: str
    benchmark: bool
    
    # Logging
    experiment_name: str
    loggers: List[Union[TensorBoardLogger, MLFlowLogger, CSVLogger, WandbLogger]]
    
    # Callbacks
    callbacks: List[Union[ModelSummary, StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint]]


config = Config(
    # base
    data_dir="/data2/projects/DigiStrudMed_sklein/",
    ckpt_dir="/data2/projects/DigiStrudMed_sklein/checkpoints/",
    log_dir="/data2/projects/DigiStrudMed_sklein/logs/",
    root_dir="/data2/projects/DigiStrudMed_sklein/",
    
    # Model
    input_size=8,
    num_classes=1,
    learning_rate=1e-2,
    batch_size=3,
    max_epochs=25,
    min_epochs=10,
    freeze=True,
    annos_of_interest=[
        "Tissue",
        "Tumor_vital",
        "Angioinvasion",
        "Tumor_necrosis",
        "Tumor_regression",
    ],
    
    # Dataset
    num_workers=9,
    read_level=3,
    cache_path="/data2/projects/DigiStrudMed_sklein/downsampled_datasets/cached_DownsampleDataset_level_3.json",
    input_dicts=dataset_globber("/data2/projects/DigiStrudMed_sklein/DigiStrucMed_Braesen/all_data/", "/data2/projects/DigiStrudMed_sklein/survival_status.csv"),
    datamodule_kwargs={
        "num_workers": 9,
        "prefetch_factor": 1,
        "pin_memory": True,
    },
    splits=[0.8, 0.15, 0.05],
    
    # Trainer
    accelerator="gpu",
    strategy="ddp",
    devices=[2, 3],
    accumulations=40,
    precision="bf16-mixed",
    log_frequency=1,
    sync_batch_norm=True,
    enable_progress_bar
