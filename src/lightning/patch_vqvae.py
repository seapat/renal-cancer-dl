# https://github.com/aisinai/vqvae2/blob/7fcf8a278dbcc130c0dadb6e4a2e579be9d5bd1d/networks.py#L38
# https://github.com/aisinai/vqvae2/blob/master/train_vqvae.py
from importlib import import_module
import logging
import sys
from misc.store_data import save_data_to_hdf
from networks.vqvae import VQVAE

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
from torchvision.utils import save_image

config = import_module(name=f".{sys.argv[2]}", package="src.config")
logger = logging.getLogger("train")
pl.seed_everything(1234)

class LitVQVAE(pl.LightningModule):
    def __init__(
            self, 
            first_stride: int =4, 
            second_stride:int =2, 
            embed_dim:int = 64, 
            learning_rate=3e-4, 
            n_embed=512, 
            in_channel=3, 
            channel=128, 
            n_res_block=2, 
            n_res_channel=32,
        ) -> None:
        super().__init__()

        self.vqvae = VQVAE(first_stride=first_stride, second_stride=second_stride, embed_dim=embed_dim)
        self.learning_rate = learning_rate 

        self.criterion = nn.MSELoss()
        self.latent_loss_weight = 0.25

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        img, surv_days, event_ind, case_id, stain_id, location = batch
        out, latent_loss = self.vqvae(img)
        recon_loss = self.criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss

        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=False)
        self.log('train_mse_loss', recon_loss, on_epoch=True, on_step=True, prog_bar=False)
        self.log('train_latent_loss', latent_loss, on_epoch=True, on_step=True, prog_bar=False)

        # if batch_idx % 10 == 0:
        #     recon_image_grid(n_row=5, sample_img)

        if batch_idx % 100 == 0:
            self.logger.experiment.add_image('train_img', img[0], self.current_epoch)
            self.logger.experiment.add_image('train_recon', out[0], self.current_epoch)

    def validation_step(self, batch, batch_idx):
        img, surv_days, event_ind, case_id, stain_id, location = batch
        out, latent_loss = self.vqvae(img)
        recon_loss = self.criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss

        self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=False)
        self.log('val_mse_loss', recon_loss, on_epoch=True, on_step=True, prog_bar=False)
        self.log('val_latent_loss', latent_loss, on_epoch=True, on_step=True, prog_bar=False)

        # if batch_idx % 10 == 0:
        #     recon_image_grid(n_row=5, sample_img)

        if batch_idx % 100 == 0:
            self.logger.experiment.add_image('val_img', img[0], self.current_epoch)
            self.logger.experiment.add_image('val_recon', out[0], self.current_epoch)


    def test_step(self, batch, batch_idx):
        img, surv_days, event_ind, case_id, stain_id, location = batch
        out, latent_loss = self.vqvae(img)
        recon_loss = self.criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss

        self.log('test_loss', loss, on_epoch=True, on_step=True, prog_bar=False)
        self.log('test_mse_loss', recon_loss, on_epoch=True, on_step=True, prog_bar=False)
        self.log('test_latent_loss', latent_loss, on_epoch=True, on_step=True, prog_bar=False)

    def predict_step(self, batch, batch_idx):
        img, surv_days, event_ind, case_id, stain_id, location = batch

        decoded_img, = self.vqvae(img)
        quant_t, quant_b, diff_t + diff_b, id_t, id_b = self.encode(img)
        upsample_t = vqvae_pretrain.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)

        outputs = {
            "reconstructed": decoded_img,
            "latent_code": quant, 
            "metadata": {
                "survival_days": surv_days,
                "uncensored": event_ind,
                "case_id": case_id,
                "stain_id": stain_id,
                "location": location,
            } 
            }

        save_data_to_hdf(**outputs, method="case", self.save_dir)
        save_data_to_hdf(**outputs, method="stain", self.save_dir)

        def on_training_end(self):
            # TODO

        return outputs

    def recon_image_grid(self, n_row, original_img, save_path):
        """Saves a grid of decoded / reconstructed digits."""
        self.vqvae.eval()
        original_img = original_img[0:n_row**2, :]
        with torch.no_grad():
            out, _ = self.vqvae(original_img)

        typeTensor = Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # remove normalization
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).type(typeTensor)
        std = torch.FloatTensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).type(typeTensor)
        original_img = torch.cat(original_img[0:3] * std + mean, original_img[3:])
        out = out * std + mean

        save_image(original_img.data, f'{save_path}/sample/original.png',
                nrow=n_row, normalize=True, range=(0, 1))
        save_image(out.data, f'{save_path}/sample/{str(self.current_epoch + 1).zfill(4)}.png',
                nrow=n_row, normalize=True, range=(0, 1))
        save_image(torch.cat([original_img, out], 0).data, f'{save_path}/sample/flat_{str(self.current_epoch + 1).zfill(4)}.png',
                nrow=n_row**2, normalize=True, range=(0, 1))
        self.vqvae.train()

def train():
      today = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    # setup logging for lightning and our own logger
    # We want to save it to file sincce stdout is polluted by TIFF warnings
    log_file_handler = logging.FileHandler(
        filename=f"{experiment_name}.log",
        mode="w",
    )

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(log_file_handler)
    logger.setLevel(logging.DEBUG)

    logger.info("Starting training script...")

    model: LitVQVAE = LitVQVAE(
        input_channels=config.INPUT_SIZE,
        learning_rate=config.LEARNING_RATE,
        in_channel = config.IN_CHANNEL
        # freeze=config.FREEZE,
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
        train_transforms=config.TRAINING_TRANSFORMS,
        val_transforms=config.VALIDATION_TRANSFORMS,
        test_transforms=config.TEST_TRANSFORMS,
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
        {k:v for k,v in vars(config).items() if not k.startswith("_") or not k.isupper()},
        open(
            f"{config.LOG_DIR+'configs/'}/{today}_{experiment_name}.json",
            "w",
        ),
        default = lambda o: str(o), # if str(o).isupper(),
        indent=4,
    )


    logger.info(f"Starting Training for '{experiment_name}'")
    trainer.fit(model=model, datamodule=data)
    logger.info(f"Starting single Validation run for '{experiment_name}'")
    
    trainer.validate(model=model, datamodule=data)
    if not args.debug: # ckpt_path does not work with debug mode
        logger.info(f"Starting Test for '{experiment_name}'")
        trainer.test(ckpt_path="best", model=model, datamodule=data)

    save_network(model, data)


if __name__ ="__main__":
    train()