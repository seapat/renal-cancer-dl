import sys
from typing import Literal

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelSummary,
    StochasticWeightAveraging,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger, WandbLogger

from src.datasets.base_dataset import dataset_globber

# base
DATA_DIR: str = "/data2/projects/DigiStrudMed_sklein/"
CKPT_DIR: str = DATA_DIR + "checkpoints/"
LOG_DIR: str = DATA_DIR + "logs/"
ROOT_DIR: str = DATA_DIR

# Model
INPUT_SIZE: int = 8
NUM_CLASSES: int = 1
LEARNING_RATE: float = 1e-2
BATCH_SIZE: int = 3
MAX_EPOCHS: int = 25
MIN_EPOCHS: int = 10
FREEZE: bool = True
ANNOS_OF_INTEREST: list[str] = [
    "Tissue",
    "Tumor_vital",
    "Angioinvasion",
    "Tumor_necrosis",
    "Tumor_regression",
]

# Dataset
NUM_WORKERS: int = 9  # TODO: check if 9 works better
READ_LEVEL: int = 3
CACHE_PATH: str = (
    DATA_DIR + f"downsampled_datasets/cached_DownsampleDataset_level_{READ_LEVEL}.json"
)
INPUT_DICTS: list = dataset_globber(
    DATA_DIR + "DigiStrucMed_Braesen/all_data/", DATA_DIR + "survival_status.csv"
)
DATAMODULE_KWARGS: dict[str, object] = {
    "num_workers": NUM_WORKERS,
    "prefetch_factor": 1,  # batches per worker,
    "pin_memory": True,
}
SPLITS = [0.8, 0.15, 0.05]


# Trainer
ACCELERATOR: str = "gpu"
STRATEGY: str = "ddp"
DEVICES: list | int = [2, 3]
ACCUMULATIONS: int = 40
PRECISION: Literal["16-mixed", 32, "bf16-mixed"] = "bf16-mixed"
LOG_FREQUENCY: int = 1
SYNC_BATCH_NORM: bool = True
ENABLE_PROGRESS_BAR: bool = False
NUM_NODES: int = 1
OVERFIT_BATCHES: float = 0.0
PROFILER: str = "simple"
BENCHMARK: bool = True

# Logging
try:
    EXPERIMENT_NAME: str = sys.argv[sys.argv.index("--experiment-name") + 1]
except:
    EXPERIMENT_NAME = "MISSING_EXPERIMENT_NAME"
LOGGERS: list = [
    # TensorBoardLogger(
    #     name=EXPERIMENT_NAME,
    #     save_dir=LOG_DIR + "tensorboard",
    #     max_queue=10,  # only keep 10 metrics in memory
    #     flush_secs=120,  # flush every 2 minutes
    #     log_graph=True,
    # ),
    # MLFlowLogger(
    #     experiment_name=EXPERIMENT_NAME,
    #     save_dir=LOG_DIR + "mlruns",
    #     log_model=True,
    # ),
    CSVLogger(
        name=EXPERIMENT_NAME,
        save_dir=LOG_DIR + "csv",
    ),
    WandbLogger(
        dir=LOG_DIR,
        group="Lightning",
        magic=True,
        save_code=True,
        name="test",
        mode="online",
        project="Thesis",
    )
]

# Callbacks
CALLBACKS: list = [
    ModelSummary(max_depth=3),
    # StochasticWeightAveraging(swa_lrs=LEARNING_RATE),
    LearningRateMonitor(log_momentum=True),
    ModelCheckpoint(
        CKPT_DIR,
        filename=EXPERIMENT_NAME,
        save_top_k=1,
        save_last=True,
        monitor="loss/val_loss",
        mode="min",
        auto_insert_metric_name=True,
    ),
]


# https://pytorch-lightning.readthedocs.io/en/stable/common/early_stopping.html
# EarlyStopping(
#     monitor="val_loss_epoch_mean",
#     mode="min",  # stop if the metric does not decrease # FIXME: check if loss func is maximized or minimized
#     patience=10,  # at least 3 Epochs if check_val_every_n_epoch in trainer equals 1 (formula: check_val_every_n_epoch * (patence + 1))
#     strict=True,  # crash is monitor does not exist
#     min_delta=0.00001,
# ),
# LearningRateFinder(
#     early_stop_threshold=4.0,
#     num_training_steps=1000,
# ),  # TODO: https://arxiv.org/abs/1506.01186
