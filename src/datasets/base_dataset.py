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
from src.misc.utils import get_case_id


import torchvision
from torch.utils.data import DataLoader, Dataset


def dataset_globber(
    input_dir,
    tab_dir,
    censor_key="uncensored",
    json_key="geojson",
    image_key="image",
    label_key="surv_days",
) -> list:
    # step 0: define filter for sorting AND grouping
    # \d{3}.\d{3} case.stain, then either `~` and 1 letter or arbitrary letters
    base_case: Callable = lambda x: re.search(r"RCC-TA-\d{3}.\d{3}(?:~\w|\w*)", x).group() or ""  # type: ignore

    # Step 1 glob files starting with RCC (.json, .svs, .tif)
    files: list[str] = glob(os.path.join(input_dir, "RCC-TA-*.*"))
    files = sorted(files, key=base_case)

    # step 2: group files where the starting basename is the same
    groups: list[tuple[str, list[str]]] = [
        (k, list(g)) for k, g in itertools.groupby(files, key=base_case)
    ]

    # load data and set index to case of easier access
    tabular: pd.DataFrame = pd.read_csv(tab_dir)
    tabular.index = tabular["case"]  # type: ignore

    # create dict of group-keys, where each path has its own key
    input_dict = {}
    for key, group in groups:
        # key, group = dict.items()
        case: int = get_case_id(key)
        if case in tabular.index:
            image = {image_key: path for path in group if path.endswith(".svs")}
            json = {json_key: path for path in group if path.endswith(".json")}
            masks = {
                path.split("-")[-1].removesuffix(".tif"): path
                for path in group
                if path.endswith(".tif")
            }

            if (len(image) == 1) and (len(json) == 1) and (len(masks) > 0):
                paths = (
                    image | json | masks
                )
                # add tabular data to key-separated paths
                input_dict[key] = paths | {
                    censor_key: tabular.loc[case, censor_key],
                    label_key: tabular.loc[case, label_key],
                }


    # convert to list of dicts and take only those with image and json
    input_dicts = list(input_dict.values())
    input_dicts = list(
        filter(lambda x: image_key in x.keys() and json_key in x.keys(), input_dicts)
    )

    return input_dicts

class BaseWSIDataset(Dataset):
    def __init__(
        self,
        data: list[dict],
        keys: list[str],
        foreground_key: str,
        image_key: str,
        label_key: str,
        censor_key: str,
        json_key: str,
        root_dir: str | None,
        # normalization_factors: tuple[
        #     tuple[float, float, float], tuple[float, float, float]
        # ],
        transform1: torchvision.transforms.Compose | None = None,
        transform2: torchvision.transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.keys: list[str] = keys
        self.foreground_key: str = foreground_key
        self.image_key: str = image_key
        self.label_key: str = label_key
        self.root_dir: str | None = root_dir
        self.censor_key: str = censor_key
        self.json_key: str = json_key

        # self.mean_std: tuple[
        #     tuple[float, float, float], tuple[float, float, float]
        # ] = normalization_factors
        # self.mean: tuple[float, float, float] = self.mean_std[0]
        # self.std: tuple[float, float, float] = self.mean_std[1]

        self.transform1 = transform1
        self.transform2 = transform2

    def _get_ids(self, x: str) -> list[str]:
        # return re.findall(r"(?:\d{3})(?!/RCC-TA-)", x)
        filename = re.findall(r"([^/]*$)", x)
        if filename is not None:
            return re.findall(r"(\d{3})(?:\.|\~|.)", filename[0])
        else:
            raise ValueError("Could not determine ids from filename")

    # _get_ids: Callable = lambda self, x: re.findall(r"(?:\d{3})(?!RCC-TA-)", x)

    def _get_case_id(self, sample: dict[str, Any]) -> int:
        # import pdb; pdb.set_trace()
        return int(self._get_ids(sample[self.image_key]) [0])

    def _get_stain_id(self, sample: dict[str, Any]) -> int:
        basename_parts = self._get_ids(sample[self.image_key])
        if len(basename_parts) == 3:
            return int(basename_parts[2])  # Haemalaun stain
        elif len(basename_parts) == 2:
            return int(basename_parts[1])  # all other stains
        else:
            raise ValueError(f"Could not determine stain id from filename {sample[self.image_key]}")

    def to_json(self, path: os.PathLike | str) -> None:
        """Converts the dataset to JSON format."""

        path = os.path.abspath(path)

        with open(path, "w") as file:
            file.write(json.dumps(self.data))

        # return json.dump(self.data, open(path, 'w'), indent=4, sort_keys=True)

    def from_json(self, path):
        """Loads the dataset from JSON format."""
        with open(path, "r") as f:
            # self.data = json.load(f)

            return json.load(f)

    def get_sample(self, index):
        return self.data[index]

    def set_transforms(self, first, second):
        self.transform1 = first
        self.transform2 = second