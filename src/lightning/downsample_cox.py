import argparse
import json
import logging
import sys
from importlib import import_module

import pytorch_lightning as pl
import torch
from src.misc.utils import save_network
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, optim

from src.lightning.data_modules import DownsampleDataModule
from src.misc.losses import CoxNNetLoss, FastCPH, CoxNNetLossMetric, CoxPHMetric
from src.misc.metrics import ConcordanceMetric
from src.networks.pathology import CoxResNet
from datetime import datetime

torch.set_float32_matmul_precision("medium")
pl.seed_everything(1234)

logger = logging.getLogger("train")


class LitCoxNet(pl.LightningModule):
    def __init__(
        self,
        input_channels,
        # config,
        num_classes: int = 1,
        learning_rate: float = 1e-2,
        freeze: bool = True,
    ):
        super(LitCoxNet, self).__init__()

        self.learning_rate: float = learning_rate
        # self.loss = CoxNNetLoss()  # CoxPHLoss() #pycox.CoxPHLoss()

        self.network = CoxResNet(
            input_channels=input_channels,
            num_classes=num_classes,
            freeze=freeze,
        )

        self.loss = FastCPH()

        self.val_CI = ConcordanceMetric()
        self.train_CI = ConcordanceMetric()
        self.test_CI = ConcordanceMetric()

        self.save_hyperparameters()

    def forward(self, x: Tensor):
        return self.network(x)

    def on_train_start(self):
        if self.logger is not None:
            [logger.log_hyperparams(
                dict(self.hparams) | 
                {
                    "accumulations": int(self.trainer.accumulate_grad_batches),
                    "devices": str(list(self.trainer.device_ids)),
                    "num_devices": int(self.trainer.num_devices),
                    "initial_lr": int(self.learning_rate),
                }
            ) for logger in self.loggers]

    def training_step(self, batch, batch_idx):
        (inputs, survtime, censor, *other) = list(batch.values())
        prediction: Tensor = self.network(inputs) 
        loss = self.loss(prediction, (survtime, censor))

        self.log(
            "loss/train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.train_CI(prediction, survtime, censor)
        self.log(
            "concordance_index/train_concordance_index",
            self.train_CI,
            # on_step=True,
            # on_epoch=True,
        )
        return loss 

    def validation_step(self, batch, batch_idx):
        (inputs, survtime, censor, *other) = list(batch.values())
        prediction: Tensor = self.network(inputs) 
        loss = self.loss(prediction, (survtime, censor))
        
        self.log(
            "loss/val_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )        
        
        self.val_CI(prediction, survtime, censor)
        self.log(
            "concordance_index/val_concordance_index",
            self.val_CI,
            # on_step=True,
            # on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        (inputs, survtime, censor) = list(batch.values())[:3]
        prediction: Tensor = self.network(inputs)  # (features, hazard)
        loss = self.loss(prediction, (survtime, censor))

        self.log(
            "loss/test_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.test_CI.update(prediction, survtime, censor)
        self.log(
            "concordance_index/test_concordance_index",
            self.test_CI,
        )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.1,
                    patience=10,
                    threshold=1e-4,
                    cooldown=50,
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "loss/val_loss",
                "strict": True,
            },
        }


def train(experiment_name: str):
    today = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    # setup logging for lightning and our own logger
    # We want to save it to file sincce stdout is polluted by TIFF warnings
    log_file_handler = logging.FileHandler(
        filename=f"{experiment_name}.log",
        mode="w",
    )
    pl_light = logging.getLogger("pytorch-lightning")
    pl_core = logging.getLogger("[lightning.pytorch.core]")

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(log_file_handler)
    logger.setLevel(logging.DEBUG)

    pl_core.addHandler(logging.StreamHandler(sys.stdout))
    pl_core.addHandler(log_file_handler)
    pl_core.setLevel(logging.DEBUG)

    pl_light.addHandler(logging.StreamHandler(sys.stdout))
    pl_light.addHandler(log_file_handler)
    pl_light.setLevel(logging.DEBUG)

    logger.info("Starting training script...")

    model: LitCoxNet = LitCoxNet(
        input_channels=config.INPUT_SIZE,
        learning_rate=config.LEARNING_RATE,
        freeze=config.FREEZE,
    )
    model.example_input_array = torch.rand(1, 8, 2923, 2849)

    data = DownsampleDataModule(
        batch_size=config.BATCH_SIZE,
        input_dicts=config.INPUT_DICTS,
        annos_of_interest=config.ANNOS_OF_INTEREST,
        path=config.CACHE_PATH,
        level=config.READ_LEVEL,
        splits=config.SPLITS,
        # kwargs
        # transforms=config.TRANSFORMS,
        **config.DATAMODULE_KWARGS,
    )

    trainer = pl.Trainer(
        fast_dev_run=args.debug,
        default_root_dir=config.ROOT_DIR,
        benchmark=config.BENCHMARK,
        accumulate_grad_batches=config.ACCUMULATIONS,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        log_every_n_steps=config.LOG_FREQUENCY,
        logger=config.LOGGERS,
        max_epochs=config.MAX_EPOCHS,
        num_nodes=config.NUM_NODES,
        sync_batchnorm=config.SYNC_BATCH_NORM,
        overfit_batches=config.OVERFIT_BATCHES,
        profiler=config.PROFILER,
        strategy=config.STRATEGY,
        callbacks=config.CALLBACKS,
        precision=config.PRECISION,
        enable_progress_bar=config.ENABLE_PROGRESS_BAR,
    )

    # save the config to the log dir
    json.dump(
        {
            k: v
            for k, v in vars(config).items()
            if not k.startswith("_") or not k.isupper()
        },
        open(
            f"{config.LOG_DIR+'configs/'}/{today}_{experiment_name}.json",
            "w",
        ),
        default=lambda o: str(o),  # if str(o).isupper(),
        indent=4,
    )

    logger.info(f"Starting Training for '{experiment_name}'")
    trainer.fit(model=model, datamodule=data)
    logger.info(f"Starting single Validation run for '{experiment_name}'")

    trainer.validate(model=model, datamodule=data)
    if not args.debug:  # ckpt_path does not work with debug mode
        logger.info(f"Starting Test for '{experiment_name}'")
        trainer.test(ckpt_path="best", model=model, datamodule=data)

    save_network(model, data)


if __name__ == "__main__":
    # add argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name of the experiment",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Run in debug mode",
    )

    args = parser.parse_args()
    # sys.argv.append(args.experiment_name)
    config = import_module(name=f".{args.config}", package="src.config")

    train(args.experiment_name)

    # # @rank_zero_only
    # def on_train_epoch_end(self):
    # #     epoch_mean = torch.stack(self.training_step_outputs).nanmean()
    # #     logger.info(f"Epoch Mean Training loss: {epoch_mean} \n")
    # #     logger.info(f"Epoch Training CI: {self.train_CI.compute()} \n")
    # #     self.log(
    # #         "loss/train_loss_epoch_mean",
    # #         epoch_mean,
    # #         logger=True,
    # #         sync_dist=True,
    #         # reduce_fx=torch.nanmean,
    # #     )
    #     logger.info(f"Training Losses length {len(self.training_step_outputs)} \n")
    #     self.training_step_outputs.clear()

    # # @rank_zero_only
    # def on_validation_epoch_end(self):
    # #     # We call nanmean, since some batches might be empty (-> all samples are censored)
    # #     epoch_mean = torch.stack(self.validation_step_outputs).nanmean()
    # #     logger.info(f"Epoch Mean Validation loss: {epoch_mean} \n")
    # #     logger.info(f"Epoch Validation CI: {self.val_CI.compute()} \n")

    # #     self.log(
    # #         "loss/val_loss_epoch_mean",
    # #         epoch_mean,
    # #         logger=True,
    # #         sync_dist=True,
    # #     )
    #     logger.info(f"Validation Losses length {len(self.validation_step_outputs)} \n")
    #     self.validation_step_outputs.clear()

    # @rank_zero_only
    # def on_test_epoch_end(self):
    #     epoch_mean = torch.stack(self.testing_step_outputs).nanmean()
    #     self.log(
    #         "loss/test_loss_epoch_mean",
    #         epoch_mean,
    #         logger=True,
    #         sync_dist=True,
    #     )
    #     # free up the memory
    #     self.testing_step_outputs.clear()
