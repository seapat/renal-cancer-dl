import logging
import os
import random
import sys
from typing import Any

import cucim
import numpy as np
import pyvips
import tifffile
import torch
import torchvision
import zarr
from cucim import CuImage
import shapely
from src.datasets.base_dataset import BaseWSIDataset
from src.misc.geojson_processing import (
    geojson_key_bounds,
    geojson_to_shapely,
    get_grid_locations,
)
from src.misc.utils import get_case_id
from tifffile import imread
from torch.utils.data import DataLoader, Dataset
import time
from src.datasets.patch_dataset import PatchMaskDataset

logging.getLogger("pyvips").setLevel(logging.CRITICAL)
logging.getLogger("cucim").setLevel(logging.CRITICAL)

timer: logging.Logger = logging.getLogger("Timer")
timer.setLevel(logging.DEBUG)

debug_logger: logging.Logger = logging.getLogger("Debugger")
debug_logger.setLevel(logging.INFO)

# create console handler and set level
stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_logger = logging.getLogger("StdOut")
stdout_logger.addHandler(stdout_handler)

class GroupedPatchMaskDataset(PatchMaskDataset):

    def __init__(
        self,
        patch_num: int,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            data: list of dicts containing paths to WSI and mask images that belong together
            keys: list of keys to annotations in data
            foreground_key: key in keys to be used to extract foreground of image
            image_key: key in data of image path
            label_key: key in data that holds the label (i.e. survival time)
            censor_key: the key for accessing censoring status of the patient
            json_key: key in data that holds the path to the geojson file.
                This should contain the ploygons of the foreground area.
            patch_size: side-length of the quadratic patches to extract
            root_dir (string): optional directory with all the images. if not supplied data has to contain full filepaths.
            transform (callable, optional): Optional transform to be applied on a sample.
            overlap: overlap between patches in pixels
            normalization_factors: tuple of 2 tuples containing means and stds for each channel respectively.
                Defaults to the recommneded values for Resnet and VGG in pytorch
        """

        self.patch_num = patch_num

        super().__init__(
            *args,
            **kwargs,
        )

    def _build_location_dictlist(
        self, sample: dict[str, Any], patch_locations: list[tuple[int, int]]
    ) -> list[dict[str, int | tuple[int, int] | list[int] | str | list[tuple[int, int]]]]:
        return [
            {
                **sample,
                "locations": patch_locations,
            }
        ]

    # TODO: implement __getitem__ for grouped dataset

    def __getitem__(self, idx) -> dict[str, Any]:
        curr_sample: dict  = self.data[idx]
        shuffled_data: list[tuple[int, int]] = self.data[idx]['locations']
        random.shuffle(shuffled_data)

        subset: list[tuple[int, int]] = shuffled_data[:self.patch_num]

        samples = []
        for idx, location in enumerate(subset):
            # FIXME: this does not make sense
            samples.append(super().__getitem__(idx))

        return {
            "data": torch.cat(samples),
            self.label_key: curr_sample[self.label_key],
            self.censor_key: curr_sample[self.censor_key],
            "case_id": curr_sample["case_id"],
            "stain_id": curr_sample["stain_id"],
            "location": curr_sample["location"],
        }