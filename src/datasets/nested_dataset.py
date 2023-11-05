import itertools
import json
import logging
import os
import re
import sys
import time
from glob import glob
from typing import Any, Callable, Sequence, Union
import pandas as pd
import torch
from src.misc.utils import get_case_id


import torchvision
from torch.utils.data import DataLoader, Dataset

from src.datasets.downsample_dataset import DownsampleMaskDataset
from src.datasets.rcc_dataset import RCCDataset, rcc_to_csv


class MultimodalDataset(Dataset):
    def __init__(
        self,
        wsi_ds: DownsampleMaskDataset,
        # genomics_ds: RCCDataset,
        rcc_path: str, 
        surv_path: str,
    ) -> None:
        super().__init__()
        self.wsi_ds: DownsampleMaskDataset = wsi_ds
        # self.genomics_ds: pd.DataFrame = genomics_ds.data

        self.tabular: pd.DataFrame = pd.read_csv(surv_path)
        # self.tabular.index = self.tabular["case"]  # type: ignore

        self.normalized_df: pd.DataFrame = rcc_to_csv(rcc_path)
        # self.normalized_df = self.normalized_df.reindex(self.tabular.index, fill_value=0)

        self.data = pd.merge(self.normalized_df.reset_index(), self.tabular, how="outer", left_on='SampleID', right_on="case")
        self.data.index = self.data['case'] # type: ignore
        self.data = self.data.fillna(0)

        assert len(self.data) == len(self.tabular), f"{len(self.data)} != {len(self.tabular)}"
        assert sorted(list(self.data.index)) == sorted(list(self.tabular.loc[:, 'case'])), f"{sorted(list(self.data.index))} \n != \n {sorted(list(self.tabular.loc[:, 'case']))}"
        assert self.data.index.dtype == self.tabular.index.dtype, f"{self.data.index.dtype} != {self.tabular.index.dtype}"

    def __len__(self) -> int:
        return max(len(self.wsi_ds), len(self.normalized_df), len(self.tabular))

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor], int, int, list[str]]:
        inputs_wsi, times_wsi, events_wsi, sample_id_wsi, stain_id, _ = list(
            self.wsi_ds[idx].values()
        )
        # inputs_rcc = self.normalized_df.loc[sample_id_wsi]
        sample = torch.Tensor(self.data.loc[sample_id_wsi])
        inputs_rcc = sample[1:-4] # first value is sample id from normalized_df
        sample_id_rcc, times_rcc, death, events_rcc = sample[-4:] # values from tabular

        genes = list(self.normalized_df.columns)

        # sample_id_rcc, times_rcc, death, events_rcc = self.tabular[sample_id_wsi]

        # missing rcc files are treated as 0s
        assert (
            times_rcc == 0 or times_rcc == times_wsi
        ), f"Times are not equal, {times_rcc} != {times_wsi}"
        assert (
            events_rcc == 0 or events_rcc == events_wsi
        ), f"Events are not equal, {events_rcc} != {events_wsi}"
        assert (
            sample_id_rcc == 0 or sample_id_rcc == sample_id_wsi
        ), f"Sample ids are not equal, {sample_id_rcc} != {sample_id_wsi}"

        sample = {
            "wsi": inputs_wsi,
            "rcc": inputs_rcc,
        }

        return sample, (times_wsi, events_wsi), sample_id_wsi, stain_id, genes
