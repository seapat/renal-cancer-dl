import logging
import os
import sys
from typing import Any

import cucim
import numpy as np
import tifffile
import torch
import torchvision
import zarr
from cucim import CuImage
from src.datasets.base_dataset import BaseWSIDataset
from src.misc.geojson_processing import geojson_key_bounds
from tifffile import imread

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
        transform1: None | torchvision.transforms.Compose = None,
        transform2: None | torchvision.transforms.Compose = None,
        cache: bool = False,
        cache_file: str = "",
        # normalization_factors: tuple[
        #     tuple[float, float, float], tuple[float, float, float]
        # ] = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
            transform1 (torchvision.transforms.Compose, optional): Transform to be applied to the image only. Defaults to None.
            transform1 (torchvision.transforms.Compose, optional): Transform to be applied to the stack of image and masks. Defaults to None.
            cache (str, optional): Whether and where to cache the dataset. Defaults to None (means no caching).
        """
        # normalization_factors (tuple[tuple[float, float, float], tuple[float, float, float]], optional): Normalization factors for the image. Defaults to ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).

        super().__init__(
            data=[],
            keys=keys,
            foreground_key=foreground_key,
            image_key=image_key,
            label_key=label_key,
            root_dir=root_dir,
            censor_key=censor_key,
            # normalization_factors=None,
            json_key=json_key,
            transform1=transform1,
            transform2=transform2,
        )
        self.target_level: int = target_level

        self.cache = cache
        self.cache_file = cache_file

        assert self.image_key not in self.keys, "Image key must not be in self.keys"
        assert self.label_key not in self.keys, "Label key must not be in self.keys"
        assert self.foreground_key in self.keys, "Foreground key must be in self.keys"

        print(f"Cache: {self.cache}")
        print(f"Cache file: {self.cache_file}")
        if self.cache and os.path.exists(self.cache_file):
            stdout_logger.info(f"Found cached Dataset at {self.cache_file}")
            self.data = self.from_json(self.cache_file)
            self.max_size_hw = self.data[0]["max_size_hw"]
        else:
            self._fill_dataset(data)
            if self.cache:
                self.to_json(self.cache_file)

    def _load_tiffs(self, curr_sample) -> tuple[cucim.CuImage, dict[str, Any]]:
        image: CuImage = CuImage(curr_sample[self.image_key])
        masks: dict[str, Any] = {}
        for key in self.keys:
            if key in curr_sample.keys() and key != self.image_key:
                masks[key] = tifffile.imread(curr_sample[key], aszarr=True)
            elif (
                key != self.image_key and key != self.foreground_key
            ):  # if annotation is missing, create empty mask indicating exactly that
                masks[key] = None
        return image, masks

    def _fill_dataset(self, data: list[dict]) -> None:
        max_size_hw: tuple[int, int] = (0, 0)
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

    def _write_lookup_table(
        self, sample
    ) -> tuple[dict[int, int], dict[int, tuple[int, int]]]:
        mask2page: dict[int, int] = {}
        dimensions: dict[int, tuple[int, int]] = {}

        cu_img = CuImage(sample[self.image_key])

        store = tifffile.imread(sample[self.foreground_key], aszarr=True)
        z_obj = zarr.open(store, mode="r")

        for idx, (width, height) in enumerate(cu_img.resolutions["level_dimensions"]):
            for mask_page in range(len(z_obj)):
                tmp_mask = z_obj[mask_page]

                if tmp_mask.shape[:2] == (height, width):
                    mask2page[idx] = mask_page
                    dimensions[idx] = (width, height)

        return mask2page, dimensions

    def _determine_bounding_box(self, sample: dict[str, Any]) -> dict:
        self.level2maskpage, dimensions = self._write_lookup_table(sample)

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
            "location_wh": tuple(int(value) for value in (x1, y1)),
            "size_wh": tuple(int(value) for value in (x2 - x1, y2 - y1)),
            "img_level": self.target_level,
            "mask_page": mask_page,
        }

    def __getitem__(self, idx) -> dict:  # -> dict[str, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        curr_sample: dict = self.data[idx]

        image_full: CuImage
        masks_full: dict[str, tifffile.tifffile.ZarrTiffStore]
        assert (
            curr_sample["img_level"] == self.target_level
        ), "saved level is not the same as target level"
        image_full, masks_full = self._load_tiffs(
            curr_sample,
        )

        # get the bounding box for the image manually
        # We do not round up this number (decimals will be cut later) since we generate masks in the same manner
        (
            minx,
            miny,
            *_,
        ) = geojson_key_bounds(curr_sample[self.json_key], self.foreground_key)

        # cant save tuple to json, hence it isloaded as a list
        assert (
            type(curr_sample["location_wh"]) == list
            or type(curr_sample["location_wh"]) == tuple
        ), f"location_wh is not a list but a {str(type(curr_sample['location_wh']))}"

        # TODO: START extract method
        # load masks
        temp_dict: dict[str, np.ndarray | object] = {}
        x, y = tuple(int(float_num) for float_num in curr_sample["location_wh"])
        w, h = tuple(int(float_num) for float_num in curr_sample["size_wh"])
        for key, value in masks_full.items():
            if value is None:
                temp_dict[key] = np.zeros(
                    shape=(
                        int(curr_sample["size_wh"][1]),
                        int(curr_sample["size_wh"][0]),
                    ),
                    dtype=np.uint8,
                )
            else:
                z_obj = zarr.open(value, mode="r")
                temp_dict[key] = z_obj[curr_sample["mask_page"]][y : y + h, x : x + w]
        masks: dict[str, object] = temp_dict
        # load image

        image: torch.Tensor = torch.einsum(
            "hwc->chw",
            torch.as_tensor(
                np.asarray(
                    image_full.read_region(
                        location=(
                            minx,
                            miny,
                        ),
                        size=(
                            w,
                            h,
                        ),
                        level=self.target_level,
                    )
                ),
                dtype=torch.float32,
            ),
        )
        # END extract method
        # image = np.asarray(image)

        if self.transform1:
            # Note: Only do this for the image, not for the masks
            # 1. scale from [0,255] to [0.0, 1.0]
            # NB: using transforms.toTensor() causes memory issues
            image = image / 255

            image = self.transform1(image)

        # create common dict
        return_sample: dict[str, torch.Tensor] = {
            self.image_key: image
            # torch.einsum("hwc->chw", torch.as_tensor(image))
        } | {key: torch.as_tensor(value).unsqueeze(0) for key, value in masks.items()}

        # TODO: START extract method
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
                # floor division can cut of a pixel, hence we add 1 to one end if that happens
                int((curr_sample["max_size_hw"][1] - value.shape[2]) // 2),
                int((curr_sample["max_size_hw"][1] - value.shape[2]) // 2 + (factor_y)),
                int((curr_sample["max_size_hw"][0] - value.shape[1]) // 2),
                int((curr_sample["max_size_hw"][0] - value.shape[1]) // 2 + factor_x),
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
        # END extract method

        # multiply image with foreground mask to set background to 0
        return_sample[self.image_key] = (
            return_sample[self.image_key] * return_sample[self.foreground_key]
        )

        return_sample["data"] = torch.cat(
            [return_sample[key] for key in return_sample.keys()], dim=0
        )

        if self.transform2:
            return_sample["data"] = self.transform2(return_sample["data"])

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

if __name__ == "__main__":
    from src.datasets.base_dataset import dataset_globber
    from torchvision import transforms

    config: dict[str, int | str | list[str | float] | float] = {
        "lr": 0.02,
        "num_epochs": 500,
        "seed": 42,
        "batch_size": 50,
        "target_level": 3,
        "num_workers": 9,
        "base_dir": "/data2/projects/DigiStrudMed_sklein/",
        "overfit": 1.0,
        "run_name": "blas",
        "data_split": [0.8, 0.15, 0.05],
        "annos_of_interest": [
            "Tissue",
            "Tumor_vital",
            "Angioinvasion",
            "Tumor_necrosis",
            "Tumor_regression",
        ],
    }

    level: int = int(config["target_level"])
    config["cache_path"] = (
        config["base_dir"]
        + f"downsampled_datasets/cached_DownsampleDataset_level_{level}.json"
    )

    input_dicts = dataset_globber(
        config["base_dir"] + "DigiStrucMed_Braesen/all_data/",
        config["base_dir"] + "survival_status.csv",
    )

    img_transform = transforms.Compose(
        [
            # transforms.ColorJitter(
            #     brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01
            # ),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

        ]
    )

    stack_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ]
    )

    dataset: DownsampleMaskDataset = DownsampleMaskDataset(
        input_dicts, #[ : int(len(input_dicts) * config["overfit"])],
        foreground_key="Tissue",
        image_key="image",
        label_key="surv_days",
        keys=config["annos_of_interest"],
        censor_key="uncensored",
        json_key="geojson",
        cache=True,
        cache_file=config["cache_path"],
        target_level=config["target_level"],
        transform1=img_transform,
        transform2=stack_transform,
    )

    batch = dataset[0]