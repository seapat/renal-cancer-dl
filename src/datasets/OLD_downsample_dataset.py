import logging
import os
import sys

import cucim
from tifffile import imread
import numpy as np
import pyvips
import torch
import torchvision
from cucim import CuImage

from src.misc.geojson_processing import (
    geojson_key_bounds,
)
from typing import Any
from src.datasets.base_dataset import BaseWSIDataset
# make pyvips shut up
# logging.basicConfig(level=logging.DEBUG)
# logging.disable(level=logging.CRITICAL)

logging.getLogger("pyvips").setLevel(logging.CRITICAL)
logging.getLogger("cucim").setLevel(logging.CRITICAL)

# create logger
timer: logging.Logger = logging.getLogger("Timer")
timer.setLevel(logging.DEBUG)

debug_logger: logging.Logger = logging.getLogger("Debugger")
debug_logger.setLevel(logging.INFO)

# create console handler and set level
stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_logger = logging.getLogger("StdOut")
stdout_logger.addHandler(stdout_handler)

class DownsampleMaskDataset(BaseWSIDataset):
    def __init__(
        self,
        data: list[dict],
        keys: list[str],
        foreground_key: str,
        image_key: str,
        target_level: int,
        censor_key: str,
        label_key: str,
        json_key: str,
        root_dir: str | None = "",
        transform: bool | torchvision.transforms.Compose = False,
        cache: bool = False,
        cache_file: str = "",
        normalization_factors: tuple[
            tuple[float, float, float], tuple[float, float, float]
        ] = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ) -> None:
        """
        Args:
            data (list[dict]): List of dictionaries containing the paths to the images and masks.
            keys (list[str]): List of keys to be used for the masks.
            foreground_key (str): Key to be used for the foreground mask.
            image_key (str): Key to be used for the image.
            target_level (int): Target level to downsample the masks to.
            censor_key (str): Key to be used for the censor mask.
            label_key (str): Key to be used for the label mask.
            json_key (str): Key to be used for the json file.
            root_dir (str, optional): Root directory of the dataset. Defaults to "".
            transform (torchvision.transforms.Compose, optional): Transform to be applied to the image. Defaults to None.
            cache (str, optional): Whether and where to cache the dataset. Defaults to None (means no caching).
            normalization_factors (tuple[tuple[float, float, float], tuple[float, float, float]], optional): Normalization factors for the image. Defaults to ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).
        """
        super().__init__(
            data=[],
            keys=keys,
            foreground_key=foreground_key,
            image_key=image_key,
            label_key=label_key,
            root_dir=root_dir,
            censor_key=censor_key,
            normalization_factors=normalization_factors,
            json_key=json_key,
        )

        # self.data: list[dict[str, str | int | tuple]]  # = []
        self.target_level: int = target_level
        self.transform = transform
        # self.json_key = json_key

        self.cache = cache
        self.cache_file = cache_file

        assert self.image_key not in self.keys, "Image key must not be in self.keys"
        assert self.foreground_key in self.keys, "Foreground key must be in self.keys"
        assert self.label_key not in self.keys, "Label key must not be in self.keys"

        if self.cache and os.path.exists(self.cache_file):
            stdout_logger.info(f"Found chached Dataset at {self.cache_file}")
            self.data = self.from_json(self.cache_file)
            self.max_size_hw = self.data[0]["max_size_hw"]
        else:
            self._fill_dataset(data)
            if self.cache:
                self.to_json(self.cache_file)

    def _fill_dataset(self, data: list[dict]) -> None:
        max_size_hw: tuple[int, int] = (0, 0)
        # print(data)
        for file in data:
            # sample = {}
            extended_sample: dict = self._determine_bounding_box(file)
            max_size_hw = (
                # todo: define typed dict for allvalues in self.data & hope it propagates
                max(max_size_hw[0], extended_sample["size_wh"][1]),
                max(max_size_hw[1], extended_sample["size_wh"][0]),
            )
            self.data.append(extended_sample)

        for sample in self.data:
            sample["max_size_hw"] = max_size_hw

    def _determine_bounding_box(self, sample: dict[str, Any]) -> dict:
        img = CuImage(sample[self.image_key])
        mask = pyvips.Image.tiffload(sample[self.foreground_key])  # type: ignore
        self.level2maskpage, dimensions = self._write_lookup_table(img, mask)

        try:
            mask_page: int = self.level2maskpage[self.target_level]
        except KeyError as e:
            debug_logger.error(f"{e}")
            debug_logger.error(
                f"Target level {self.target_level} not found in mask. Current Mapping: {self.level2maskpage}"
            )
            debug_logger.error(f"file: {sample[self.foreground_key]}")
            debug_logger.error(f"file: {sample[self.image_key]}")
            debug_logger.error(f"Using highest level {max(self.level2maskpage.keys())}")
            mask_page = self.level2maskpage[max(self.level2maskpage.keys())]

        foreground_at_level = imread(
            sample[self.foreground_key], series=0, level=mask_page
        )
        print(sample[self.foreground_key])

        bounding_box: torch.Tensor = torchvision.ops.masks_to_boxes(
            torch.as_tensor(foreground_at_level).unsqueeze(0)
        )
        assert bounding_box.shape[0] == 1

        (x1, y1, x2, y2) = bounding_box[0].tolist()

        sample["case_id"] = self._get_case_id(sample)
        sample["stain_id"] = self._get_stain_id(sample)
        return {
            **sample,
            "location_wh": (x1, y1),
            "size_wh": (x2 - x1, y2 - y1),
            "img_level": self.target_level,
            "mask_page": mask_page,
        }

    def _load_tiffs(
        self, curr_sample, maskpage=0
    ) -> tuple[cucim.CuImage, dict[str, pyvips.Image]]:
        image: CuImage = CuImage(curr_sample[self.image_key])
        masks: dict[str, pyvips.Image] = {}
        for key in self.keys:
            if key in curr_sample.keys() and key != self.image_key:
                masks[key] = pyvips.Image.tiffload(  # type: ignore
                    curr_sample[key],
                    page=maskpage,
                )
            elif (
                key != self.image_key and key != self.foreground_key
            ):  # if annotation is missing, create empty mask indicating exactly that
                masks[key] = pyvips.Image.black(image.shape[1], image.shape[0])  # type: ignore
        return image, masks

    def __getitem__(self, idx):  # -> dict[str, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        curr_sample: dict = self.data[idx]

        image_full: CuImage
        masks_full: dict[str, pyvips.Image]
        assert (
            curr_sample["img_level"] == self.target_level
        ), "saved level is not the same as target level"
        image_full, masks_full = self._load_tiffs(
            curr_sample,
            curr_sample["mask_page"],
        )

        # cant save tuple to json, hence it isloaded as a list
        assert (
            type(curr_sample["location_wh"]) == list
        ), f"location_wh is not a list but a {str(type(curr_sample['location_wh']))}"
        temp_dict: dict[str, pyvips.Image] = {}
        for key, value in masks_full.items():
            temp_dict[key] = value.crop(  # type: ignore
                curr_sample["location_wh"][0],
                curr_sample["location_wh"][1],
                curr_sample["size_wh"][0],  # type: ignore
                curr_sample["size_wh"][1],
            )
        masks: dict[str, pyvips.Image] = temp_dict

        # get the bounding box for the image manually
        # We do not round up this number (decimals will be cut later) since we generate masks in the same manner
        (
            minx,
            miny,
            *_,
        ) = geojson_key_bounds(curr_sample[self.json_key], self.foreground_key)
        print("blupp1")
        print(minx, miny)
        print(curr_sample["location_wh"][0], curr_sample["location_wh"][1])
        print(image_full)
        image: cucim.CuImage = image_full.read_region(
            location=(
                # multiply by 4^level since cucim reads at level 0
                # prbably better to use the bounding box from shapely
                # curr_sample["location_wh"][0] * 4 ** self.target_level,
                # curr_sample["location_wh"][1] * 4 ** self.target_level,
                minx,
                miny,
            ),
            size=(
                # cast to int as per demand by our overlords
                int(curr_sample["size_wh"][0]),
                int(curr_sample["size_wh"][1]),
            ),
            level=self.target_level,
            batch_size = 10, drop_last = True, prefetch_factor = 1
        )

        print("blupp2")


        return_sample: dict[str, torch.Tensor] = {
            # image
            self.image_key:
            # np.expand_dims(np.einsum("hwc->chw",np.asarray(image)), axis=0)
            torch.einsum("hwc->chw", torch.as_tensor(np.asarray(image, dtype    = np.float32)))
        } | {
            # masks
            key: torch.as_tensor(  # np.expand_dims(value.numpy(), axis=0)
                np.asarray(value)
            ).unsqueeze(0)
            for key, value in masks.items()
        }

        # pad to size of largest (downsampled) image in dataset
        for key, value in return_sample.items():
            # we want to pad the images equally on each side.
            # add +1 to make sure we do not loose a pixel
            # only if the current image is smaller than the max size
            factor_x = (
                0
                if ((int(curr_sample["max_size_hw"][0]) - value.shape[1]) % 2 == 0)
                or (int(curr_sample["max_size_hw"][0]) == value.shape[1])
                else 1
            )
            factor_y = (
                0
                if ((int(curr_sample["max_size_hw"][1]) - value.shape[2]) % 2 == 0)
                or (int(curr_sample["max_size_hw"][1]) == value.shape[2])
                else 1
            )
            padding = (
                # floor division on one end
                int((curr_sample["max_size_hw"][1] - value.shape[2]) // 2),
                int((curr_sample["max_size_hw"][1] - value.shape[2]) // 2 + (factor_y)),
                int((curr_sample["max_size_hw"][0] - value.shape[1]) // 2),
                int((curr_sample["max_size_hw"][0] - value.shape[1]) // 2 + factor_x),
            )
            padding = (
                # floor division on one end
                int(curr_sample["max_size_hw"][1] - value.shape[2]),  # channel first
                0,
                int(curr_sample["max_size_hw"][0] - value.shape[1]),  # channel first
                0,
            )
            return_sample[key] = torch.nn.ConstantPad2d(
                padding,
                0,
            )(value)

            # ensure all images have the correct size, shift idx by 1 due to channel first
            assert return_sample[key].shape[1] == int(
                curr_sample["max_size_hw"][0]
            ), f"{return_sample[key].shape[1]} != {int(curr_sample['max_size_hw'][0])}"
            assert return_sample[key].shape[2] == int(
                curr_sample["max_size_hw"][1]
            ), f"{return_sample[key].shape[2]} != {int(curr_sample['max_size_hw'][1])}"

        # multiply image with foreground mask to set background to 0
        return_sample[self.image_key] = (
            return_sample[self.image_key] * return_sample[self.foreground_key]
        )

        # Apply supplied transformationson on both, scaling only on images aftwards
        if self.transform:
            if type(self.transform) == torchvision.transforms.Compose:
                return_sample[self.image_key] = self.transform(
                    return_sample[self.image_key]
                )
                for key, value in return_sample.items():
                    if key in self.keys:
                        return_sample[key] = self.transform(value)

            # Normalize Image
            # Note: Only do this for the image, not for the masks
            # convert to float32
            # return_sample[self.image_key] = return_sample[self.image_key].type(
            #     torch.float32
            # )
            # 1. scale from [0,255] to [0.0, 1.0]
            return_sample[self.image_key] = return_sample[self.image_key] / 255
            # 2. normalize to [-1, 1]
            return_sample[self.image_key] = torchvision.transforms.Normalize(
                self.mean, self.std
            )(
                return_sample[self.image_key]
            )  # .type(torch.float32)

        return_sample["data"] = torch.cat(
            [return_sample[key] for key in return_sample.keys()], dim=0
        )

        return {
            "data": return_sample["data"],
            self.label_key: curr_sample[self.label_key],
            self.censor_key: curr_sample[self.censor_key],
            "case_id": curr_sample["case_id"],
            "stain_id": curr_sample["stain_id"],
            "location": curr_sample["location_wh"],
        }

    def __len__(self) -> int:
        return len(self.data)

    def _write_lookup_table( self,
        cu_img: CuImage, mask: pyvips.Image
    ) -> tuple[dict[int, int], dict[int, tuple[int, int]]]:
        mask2page: dict[int, int] = {}
        dimensions: dict[int, tuple[int, int]] = {}

        for idx, (width, height) in enumerate(cu_img.resolutions["level_dimensions"]):
            for mask_page in range(mask.get_n_pages()):  # type: ignore
                tmp_mask = pyvips.Image.tiffload(mask.filename, page=mask_page)  # type: ignore
                if tmp_mask.width == width and tmp_mask.height == height:
                    mask2page[idx] = mask_page
                    dimensions[idx] = (width, height)

        return mask2page, dimensions
