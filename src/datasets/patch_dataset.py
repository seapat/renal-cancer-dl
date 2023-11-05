import logging
import os
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


class PatchMaskDataset(BaseWSIDataset):
    """Face Landmarks dataset."""

    def __init__(
        self,
        data: list[dict[str, str | int]],
        keys: list[str],
        foreground_key: str,
        image_key: str,
        label_key: str,
        censor_key: str,
        json_key: str,
        # extraction_level: int,
        patch_size: int = 512,
        root_dir: str = "",
        overlap=0,
        # target_level=0,        
        cache: bool = False,
        cache_file: str = "",
        transform1: None | torchvision.transforms.Compose = None,
        transform2: None | torchvision.transforms.Compose = None,
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
            transform1 (torchvision.transforms.Compose, optional): Transform to be applied to the image only. Defaults to None.
            transform1 (torchvision.transforms.Compose, optional): Transform to be applied to the stack of image and masks. Defaults to None.
        """
        super().__init__(
            data=[],
            keys=keys,
            foreground_key=foreground_key,
            image_key=image_key,
            label_key=label_key,
            root_dir=root_dir,
            censor_key=censor_key,
            transform1=transform1,
            transform2=transform2,
            json_key=json_key,
        )
        # extraction_level: Pyramid level of mask at which to extract patch locations for upsampling
        # target_level: level at which to extract patches. Defaults to 0
        self.regions: dict = {}

        self.mil = False

        self.patch_size: int = patch_size
        self.overlap: int = overlap

        self.cache = cache
        self.cache_file = cache_file

        assert self.image_key not in self.keys, "Image key must not be in self.keys"
        assert self.foreground_key in self.keys, "Foreground key must be in self.keys"
        assert self.label_key not in self.keys, "Label key must not be in self.keys"
        # assert self.extraction_level >= 0
        assert self.patch_size > 0, "Patch size must be greater than 0"

        if self.cache and os.path.exists(self.cache_file):
            stdout_logger.info(f"Found cached Dataset at {self.cache_file}")
            self.data = self.from_json(self.cache_file)
        else:
            self.data: list
            self.input_data: list[dict[str, str | int]] = list(data)
            for sample in self.input_data:
                start: float = time.time()
                print(f"Adding patches from {sample[self.image_key]}")
                patch_samples: list[dict[str, Any]] = self._determine_patch_locations(
                    sample
                )
                print(
                    f"Added {len(patch_samples)} patches! Time elapsed: {time.time() - start}"
                )
                self.data.extend(patch_samples)
            if self.cache:
                self.to_json(self.cache_file)


    def _determine_patch_locations(
        self, sample: dict[str, Any]
    ) -> list[dict[str, int | tuple[int, int] | list[int] | str]]:
        # masks and images are from the same case
        assert sample[self.image_key].split("/")[-1].removesuffix(".svs") == sample[
            self.foreground_key
        ].split("/")[-1].removesuffix(f"-{self.foreground_key}.tif") or sample[
            self.image_key
        ].split(
            "/"
        )[
            -1
        ].removesuffix(
            ".svs"
        ) == sample[
            self.foreground_key
        ].split(
            "/"
        )[
            -1
        ].removesuffix(
            f"-{self.foreground_key}.tiff"
        ), "Image and mask are not from the same case"

        shapes: dict[str, shapely.Geometry] = geojson_to_shapely(
            sample[self.json_key], sample[self.foreground_key]
        )
        polys: list[shapely.Polygon] = (
            shapes[self.foreground_key].geoms
            if isinstance(shapes[self.foreground_key], shapely.MultiPolygon)
            else [shapes[self.foreground_key]]
        )

        # height, width, dims = CuImage(sample[self.image_key]).size()

        patch_locations: list[tuple[int, int]] = [
            (
                # 1. shift location so that so that it becomes the center of the patch
                # 2. if left or top out of bounds, set to 0
                # 3. if right or botom out of bounds set shifted
                # min(max(int(point.x - (self.patch_size / 2)), 0), width - self.patch_size),
                # min(max(int(point.y - (self.patch_size / 2)), 0), height - self.patch_size)
                # I dont think we need to shift if we overlap right or bottom,
                # since we catch that later by padding those images
                # FIXME: this causes an overlap, w
                int(point.x - (self.patch_size / 2)),
                int(point.y - (self.patch_size / 2)),
            )
            for geom in polys
            for point in get_grid_locations(geom, self.patch_size, self.overlap)
        ]

        # # There is no gap between patches, so the distance between the first and the second patch should be the patch size
        # if len(patch_locations) > 1:
        #     soted_locs: list[tuple[int, int]] = sorted(patch_locations, key=lambda val: val[1])
        #     assert (
        #         soted_locs[1][1] - soted_locs[0][1]
        #         == self.patch_size - self.overlap
        #     ), f"Patch locations dont correspond to patch size (gaps) {soted_locs[1][1]} - {soted_locs[0][1]} = {soted_locs[1][1] - soted_locs[0][1]}"
        #     # We'll just assume that his also holds for the vertical direction

        # patch_locations: np.ndarray[Any, np.dtype[Any]] = np.array(patch_locations)

        sample["patch_size"] = self.patch_size
        sample["num_patches"] = len(patch_locations)
        sample["case_id"] = self._get_case_id(sample)
        sample["stain_id"] = self._get_stain_id(sample)
        return self._build_location_dictlist(sample, patch_locations)

    def _build_location_dictlist(
        self, sample: dict[str, Any], patch_locations: list[tuple[int, int]]
    ) -> list[dict[str, int | tuple[int, int] | list[int] | str]]:
        return [
            {
                **sample,
                "location": loc,
            }
            for loc in patch_locations
        ]

    def __len__(self) -> int:
        return len(self.data)

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

    def __getitem__(self, idx) -> dict[str, Any]:
        import time

        start: float = time.time()

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # alias the input
        curr_sample: dict = self.data[idx]

        # A mask of interest is used for patch extraction
        assert (
            self.foreground_key in curr_sample.keys()
        ), f"Foreground key {self.foreground_key} not in sample keys"

        # load the image and masks
        timer.info(f"Loading image and masks {time.time() - start}")
        image_full: cucim.CuImage
        masks_full: dict[str, pyvips.Image]
        image_full, masks_full = self._load_tiffs(curr_sample)
        timer.info(f"Loading done {time.time() - start}")

        # image does not occur in masks / in self.keys
        assert (
            self.image_key not in masks_full.keys()
        ), f"Image key {self.image_key} is also a mask key"

        # image.shape[0] is the height, image.shape[1] is the width
        for key, value in masks_full.items():
            assert image_full.shape[0:2] == [value.height, value.width]

        timer.info(f"Starting patch extraction for masks {time.time() - start}")
        image: cucim.CuImage
        mask_images: dict[str, Any]
        image, mask_images = self._extract_patches_from_locations(
            curr_sample, image_full, masks_full
        )
        timer.info(f"Size of image patch {image.shape} {time.time() - start}")
        timer.info(f"Patch extraction for image done {time.time() - start}")
        timer.info(f"Starting conversion to tensor {time.time() - start}")

        try:
            masks: dict[str, np.ndarray] = {
                key: np.ndarray(
                    buffer=value,
                    dtype=np.uint8,
                    shape=[curr_sample["patch_size"][1], curr_sample["patch_size"][0]],
                )
                for key, value in mask_images.items()
            }
        except Exception as e:
            print([len(i) for i in mask_images.values()])
            print(self.patch_size**2)
            print(f"{curr_sample}")
            print(f'{curr_sample["case_id"]} {curr_sample["stain_id"]}')
            print(f"{key} {value}")  # type: ignore
            print(f"width {value.width} height {value.height}")  # type: ignore

            raise e

        timer.info(f"Converting masks to numpy done {time.time() - start}")

        # image = torch.einsum("hwc->chw", torch.from_numpy(np.asarray(image)))
        image = torch.einsum("hwc->chw", torch.from_numpy(np.asarray(image)))
        if self.transform1:
            image = image / 255
            image = self.transform1(image)

        # read image and masks image->numpy->tensor and change to channel first       
        return_sample: dict[str, torch.Tensor] = {
            # image
            self.image_key:
            # np.expand_dims(np.einsum("hwc->chw",np.asarray(image)), axis=0)
            image
        } | {
            # masks
            key: torch.as_tensor(  # np.expand_dims(value.numpy(), axis=0)
                value  # np.asarray(value)
            ).unsqueeze(0)
            for key, value in masks.items()
        }

        timer.info(f"Converting to tensor done {time.time() - start}")

        timer.info(f"Starting padding {time.time() - start}")
        # pad the patches with zeros if they are too small
        for key, value in return_sample.items():
            if (
                value.shape[1] != self.patch_size
                or value.shape[2] != self.patch_size
                # and key in self.keys
            ):
                return_sample[key] = torch.nn.ConstantPad2d(
                    (
                        0,
                        self.patch_size - value.shape[2],
                        0,
                        self.patch_size - value.shape[1],
                    ),
                    0,
                )(value)
            assert return_sample[key].shape[1:3] == (
                self.patch_size,
                self.patch_size,
            ), f"Shape of {key} mask {return_sample[key].shape[1:3]} does not match shape of {(self.patch_size, self.patch_size)}"
        timer.info(f"Padding done {time.time() - start}")

        # multiply image with foreground mask to set background to 0
        return_sample[self.image_key] = (
            return_sample[self.image_key] * return_sample[self.foreground_key]
        )

        # stack patch per label onto image
        return_sample["data"] = torch.cat(
            [return_sample[key] for key in return_sample.keys()], dim=0
        )
        if self.transform2:
            return_sample["data"] = self.transform2(return_sample["data"])

        # copy over the label
        # return_sample[self.label_key] = curr_sample[self.label_key]
        timer.info(f"Return_sample keys {return_sample.keys()} {time.time() - start}")

        return {
            "data": return_sample["data"],
            self.label_key: curr_sample[self.label_key],
            self.censor_key: curr_sample[self.censor_key],
            "case_id": curr_sample["case_id"],
            "stain_id": curr_sample["stain_id"],
            "location": curr_sample["location"],
        }

    def _extract_patches_from_locations(
        self, curr_sample: dict, image_in: cucim.CuImage, masks_in: dict[str, Any]
    ) -> tuple[cucim.CuImage, dict[str, pyvips.Image]]:
        # logger.info(f"extracting patches for {curr_sample['Tissue']}")

        """NOTE:
        We create "regions" and "fetch" from that instead of using crop directly on the images.
        Reason: pyvips sets up some kind of pipeline when using crop in order to do multi-threaded processing
        This does not work well with num_workers of pytorch dataloaders. The pipeline is pointless as well since,
            it is recreated for every creation of a pyips image (ie. call of __getitem__).
        """

        # try to create nested dict of dicts to cache created regions and re-use them when extracting patches
        curr_case = curr_sample["case_id"]
        self.regions[curr_case] = {}

        # crop till the end of image if patch would overlap padded later with zeros
        temp_dict: dict[str, pyvips.Image] = {}
        for key, value in masks_in.items():
            if curr_sample["location"][1] + self.patch_size > value.height:
                y_fit_size: int = value.height - curr_sample["location"][1]
            else:
                y_fit_size = self.patch_size

            if curr_sample["location"][0] + self.patch_size > value.width:
                x_fit_size: int = value.width - curr_sample["location"][0]
            else:
                x_fit_size = self.patch_size

            # we use this to create a numpy array from the fetched region,
            # if we have to use a smaller region *due to out of bounds), we
            # save the actual region size here
            # then this is combined with the image patch (same size)
            # and padded afterwards
            curr_sample["patch_size"] = (x_fit_size, y_fit_size)

            assert (
                x_fit_size >= 0 and y_fit_size >= 0
            ), f"Patch size {self.patch_size} is too large for image {value.width} x \
                 {value.height} at location {curr_sample['location']}"

            debug_logger.debug(f"cropping {value.filename}")
            debug_logger.debug(f"image size {value.width} x {value.height}")
            debug_logger.debug(f"cropping location {curr_sample['location']}")
            debug_logger.debug(f"x_fit_size {x_fit_size} y_fit_size {y_fit_size}")

            # only create new region if it does not exist yet
            if key in self.regions[curr_case].keys():
                region = self.regions[curr_case][key]
            else:
                region = pyvips.Region.new(value)
                self.regions[curr_case][key] = region
            # _ region = pyvips.Region.new(value)

            # try:
            temp_dict[key] = region.fetch(  # value.crop
                curr_sample["location"][0],
                curr_sample["location"][1],
                x_fit_size,
                y_fit_size,
                # page 0 as seen above
            )
            #     if len(temp_dict[key]) > self.patch_size**2:  # type: ignore
            #         print(f"region {len(temp_dict[key])} is too large for patch size {self.patch_size ** 2    }")  # type: ignore
            #         print(value)
            #         print(f"curr_sample location {curr_sample['location']}")
            #         print(f"true width {value.width } true height {value.height}")
            #         print(f"x_fit_size {x_fit_size} y_fit_size {y_fit_size}")
            #         assert len(temp_dict[key]) == self.patch_size**2  # type: ignore
            # except Exception as e:
            #     print(curr_sample)
            #     print(value)
            #     print(e)

        masks: dict[str, pyvips.Image] = temp_dict

        if curr_sample["location"][1] + self.patch_size > image_in.shape[0]:
            y_fit_size = image_in.shape[0] - curr_sample["location"][1]
        else:
            y_fit_size = self.patch_size

        if curr_sample["location"][0] + self.patch_size > image_in.shape[1]:
            x_fit_size = image_in.shape[1] - curr_sample["location"][0]
        else:
            x_fit_size = self.patch_size

        image = image_in.read_region(
            (curr_sample["location"][0], curr_sample["location"][1]),
            (
                int(x_fit_size),
                int(y_fit_size),
            ),  # cast to int as per demand by our overlords
            level=0,  # self.target_level,
        )

        return image, masks
