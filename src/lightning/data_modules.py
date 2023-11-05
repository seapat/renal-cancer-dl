import pytorch_lightning as pl
import torch
from torch.utils.data import (
    DataLoader,
    random_split,
    SubsetRandomSampler,
    Subset,
    Dataset,
)
import numpy as np

from src.datasets.downsample_dataset import DownsampleMaskDataset
from src.datasets.patch_dataset import PatchMaskDataset, BaseWSIDataset
from torchvision import transforms


class DownsampleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        input_dicts,
        annos_of_interest,
        path,
        level,
        splits: list,
        # transforms: tuple[Compose | None, Compose | None],
        **kwargs
    ) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.dataset = DownsampleMaskDataset(
            input_dicts,
            foreground_key="Tissue",
            image_key="image",
            label_key="surv_days",
            keys=annos_of_interest,
            censor_key="uncensored",
            json_key="geojson",
            cache=True,
            cache_file=path,
            target_level=level,
            transform1=transforms.Compose(
                [
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    # transforms.ColorJitter(
                    #     brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01
                    # ),
                ]
            ),
            transform2=None,
            # transforms.Compose(
                # [
                    # transforms.RandomHorizontalFlip(0.5),
                    # transforms.RandomVerticalFlip(0.5),
                # ]
            # ),
        )
        self.data_split = splits

    def train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            **self.kwargs
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            **self.kwargs
        )
        return val_dataloader

    def test_dataloader(self) -> DataLoader:
        test_dataloader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            **self.kwargs
        )
        return test_dataloader

    def setup(self, stage) -> None:
        # runs on every single cpu, we can use self here
        # setup copies model to each gpu
        self.train_ds: Dataset
        self.val_ds: Dataset
        self.test_ds: Dataset

        self.train_ds, self.val_ds, self.test_ds = random_split(
            self.dataset, self.data_split, generator=torch.Generator().manual_seed(42)
        )


class PatchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        input_dicts,
        annos_of_interest,
        path,
        level,
        splits,
        transforms: tuple[transforms.Compose | None, transforms.Compose | None] = (
            None,
            None,
        ),
        **kwargs
    ) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.dataset = PatchMaskDataset(
            input_dicts,
            foreground_key="Tissue",
            image_key="image",
            label_key="surv_days",
            keys=annos_of_interest,
            censor_key="uncensored",
            json_key="geojson",
            transform1=transforms[0],
            transform2=transforms[1],
        )
        self.data_split = splits

        self.hparams["dataset_size"] = len(self.dataset)

    def train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, **self.kwargs
        )  # True

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, **self.kwargs
        )
        return val_dataloader

    def test_dataloader(self) -> DataLoader:
        test_dataloader = DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False, **self.kwargs
        )
        return test_dataloader

    def setup(self, stage) -> None:
        # runs on every single cpu, we can use self here
        # setup copies model to each gpu
        # this will create them reduantly but since we only hold a few file paths that should not be a problem
        self.train_ds: Dataset
        self.val_ds: Dataset
        self.test_ds: Dataset

        self.train_ds, self.val_ds, self.test_ds = random_split(
            self.dataset, self.data_split, generator=torch.Generator().manual_seed(42)
        )
