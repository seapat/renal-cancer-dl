# https://www.youtube.com/watch?v=mV7bhf6b2Hs
# https://www.youtube.com/watch?v=CTsSrOKSPNo
# https://arxiv.org/abs/1906.02691

from importlib import import_module
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

config = import_module(name=f".{sys.argv[2]}", package="src.config")
logger = logging.getLogger("train")
pl.seed_everything(1234)

class LitVAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_channels=3):
        super().__init__()

        self.save_hyperparameters()

        self.encoder = ResNet_Encoder(input_channels)
        self.decoder = ResNet_Decoder(input_channels)

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z), per-sample likelihood
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # def reparameterize(self, mu, logvar):
        #     if self.training:
        #         std = logvar.mul(0.5).exp_()
        #         eps = Variable(std.data.new(std.size()).normal_())
        #         return eps.mul(std).add_(mu)
        #     else:
        #         return mu

        # def forward(self, x):
        #     mu, logvar = self.encode(x)
        #     z = self.reparameterize(mu, logvar)
        #     x_reconst = self.decode(z)

        #     return x_reconst, z, mu, logvar

        # decoded
        x_recon = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_recon, self.log_scale, x)

        # Kullback–Leibler divergence
        kl_div = self.kl_divergence(z, mu, std)

        # Evidence lower bound == loss
        elbo = (kl_div - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl_div.mean(),
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
        })

        return elbo

    def on_train_end(self) -> None:
        # TODO
        self.log("MSE_weight", )
        self.log("BCE_weight", )
        return super().on_train_end()


def train(experiment_name: str):
    # setup logging for lightning and our own logger
    # We want to save it to file sincce stdout is polluted by TIFF warnings
    log_file_handler = logging.FileHandler(
        filename=f"{experiment_name}.log",
        mode="w",
    )
    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

    logger.info("Starting training script...")

    model: LitVAE = LitVAE(
        input_channels=config.INPUT_SIZE,
        learning_rate=config.LEARNING_RATE,
        # config=vars(config),
    )
    model.example_input_array = torch.rand(1, 8, 2923, 2849)

    data = PatchDataModule(
        batch_size=config.BATCH_SIZE,
        input_dicts=config.INPUT_DICTS,
        annos_of_interest=config.ANNOS_OF_INTEREST,
        path=config.CACHE_PATH,
        level=config.READ_LEVEL,
        # kwargs
        train_transforms=config.TRAINING_TRANSFORMS,
        val_transforms=config.VALIDATION_TRANSFORMS,
        test_transforms=config.TEST_TRANSFORMS,
        **config.DATAMODULE_KWARGS,
    )

    trainer = pl.Trainer(
        fast_dev_run=config.DEBUG,
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
        vars(config),
        open(
            f"{config.LOG_DIR}/{experiment_name}.json",
            "w",
        ),
        default = lambda o: str(o),
    )


    logger.info(f"Starting Training for '{sys.argv[1]}'")
    trainer.fit(model=model, datamodule=data)
    logger.info(f"Starting single Validation run for '{sys.argv[1]}'")
    trainer.validate(model=model, datamodule=data)
    logger.info(f"Starting Test for '{sys.argv[1]}'")
    trainer.test(ckpt_path="best", model=model, datamodule=data)

    save_network(model, data)

if __name__ == '__main__':
    train(sys.argv[1])