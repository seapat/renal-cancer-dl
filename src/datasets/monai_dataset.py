# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import json
import os
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.data import PatchWSIDataset
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import iter_patch_position
from monai.data.wsi_reader import BaseWSIReader, WSIReader
from monai.transforms import ForegroundMask, Randomizable, apply_transform
from monai.utils import convert_to_dst_type, ensure_tuple_rep
from monai.utils.enums import CommonKeys, ProbMapKeys, WSIPatchKeys


import pyvips
from src.misc.pyvips_wrapper import PyVipsReader

__all__ = [
    # "PatchWSIDataset",
    "BinMaskedPatchWSIDataset"
]


class BinMaskedPatchWSIDataset(PatchWSIDataset):
    """
    This dataset extracts patches from whole slide images at the locations where foreground mask
    at a given level is non-zero.

    Args:
        data: the list of input samples including image, location, and label (see the note below for more details).
        masks: List of annotations names
        tissue_label: the label of the foreground mask.
        patch_size: the size of patch to be extracted from the whole slide image.
        patch_level: the level at which the patches to be extracted (default to 0).
        mask_level: the resolution level at which the mask is created.
        transform: transforms to be executed on input data.
        include_label: whether to load and include labels in the output
        center_location: whether the input location information is the position of the center of the patch
        additional_meta_keys: the list of keys for items to be copied to the output metadata from the input data
        reader: the module to be used for loading whole slide imaging. Defaults to cuCIM. If `reader` is

            - a string, it defines the backend of `monai.data.WSIReader`.
            - a class (inherited from `BaseWSIReader`), it is initialized and set as wsi_reader,
            - an instance of a class inherited from `BaseWSIReader`, it is set as the wsi_reader.

        kwargs: additional arguments to pass to `WSIReader` or provided whole slide reader class

    Note:
        The input data has the following form as an example:

        .. code-block:: python

            [
                {"image": "path/to/image1.tiff"},
                {"image": "path/to/image2.tiff", "size": [20, 20], "level": 2}
            ]

    """

    def __init__(
        self,
        data: Sequence,
        masks: Sequence,
        tissue_label: str = "Tissue",
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_level: Optional[int] = None,
        mask_level: int = 7,
        transform: Optional[Callable] = None,
        include_label: bool = False,
        center_location: bool = False,
        additional_meta_keys: Sequence[str] = (ProbMapKeys.LOCATION, ProbMapKeys.NAME),
        reader="cuCIM",
        **kwargs,
    ):
        super().__init__(
            data=[],
            patch_size=patch_size,
            patch_level=patch_level,
            transform=transform,
            include_label=include_label,
            center_location=center_location,
            additional_meta_keys=additional_meta_keys,
            reader=reader,
            **kwargs,
        )

        self.masks = masks
        self.tissue_label = tissue_label
        self.mask_level = mask_level

        # Create single sample for each patch (in a sliding window manner)
        self.data: list
        self.image_data = list(data)
        for sample in self.image_data:
            patch_samples = self._evaluate_patch_locations(sample)
            self.data.extend(patch_samples)

        self.mask_object_dict: Dict = {}


    def _write_lookup_tables(self, wsi):

        img = pyvips.Image.tiffload(self.data[0][self.image_key])
        mask = pyvips.Image.tiffload(self.data[0][self.foreground_key])
        
        image2page = {}
        mask2page = {}
        dimensions = {}

        for img_page in range(img.get_n_pages()):
            tmp_img = pyvips.Image.tiffload(img.filename, page=img_page)
            for mask_page in range(mask.get_n_pages()):
                tmp_mask = pyvips.Image.tiffload(mask.filename, page=mask_page)
                if (
                    tmp_mask.width == tmp_img.width
                    and tmp_mask.height == tmp_img.height
                ):

                    mask2page[img_page] = mask_page
                    image2page[mask_page] = img_page
                    dimensions[img_page] = np.array((tmp_mask.width, tmp_mask.height))

        image_level_to_page_n = {
            idx: value for idx, value in enumerate(image2page.values())
        }
        mask_level_to_page_n = {
            idx: value for idx, value in enumerate(mask2page.values())
        }

        dimensions = {idx: value for idx, value in enumerate(dimensions.values())}

        image_page = image_level_to_page_n[self.extraction_level]
        mask_page = mask_level_to_page_n[self.extraction_level]

        return image_level_to_page_n, mask_level_to_page_n, dimensions

    def _get_mask_object(self, sample: Dict, label: str):
        mask_path = sample[label]
        if mask_path not in self.mask_object_dict:
            self.mask_object_dict[mask_path] = PyVipsReader(sample[label])
        return self.mask_object_dict[mask_path]

    def _get_data(self, sample: Dict):
        # Don't store OpenSlide objects to avoid issues with OpenSlide internal cache
        if self.backend == "openslide":
            self.wsi_object_dict = {}
        wsi_obj = self._get_wsi_object(sample)
        location = self._get_location(sample)
        level = self._get_level(sample)
        size = self._get_size(sample)

        img_data, meta_data = self.wsi_reader.get_data(
            wsi=wsi_obj, location=location, size=size, level=level
        )

        mask_dict = {}
        for mask in self.masks:
            if mask in sample.keys():
                mask_obj = self._get_mask_object(sample, mask)
                mask_data = mask_obj.read_region(
                    location_at_0=location, extract_size=size, level=level
                )
                mask_dict[mask] = mask_data.numpy()  # this is shit
                print()

        return meta_data, img_data, mask_dict

    def _transform(self, index: int):
        # Get a single entry of data
        sample: Dict = self.data[index]

        # Extract patch image and associated metadata
        metadata, image, masks = self._get_data(sample)

        # Add additional metadata from sample
        for key in self.additional_meta_keys:
            metadata[key] = sample[key]

        metadata["Masks"] = list(masks.keys())

        # Create MetaTensor output for image
        output = (
            masks | {CommonKeys.IMAGE: MetaTensor(image, meta=metadata)}
            if CommonKeys.IMAGE not in masks.keys()
            else {"Error": "the masks dict contains the key 'image'!"}
        )
        # Include label in the output
        if self.include_label:
            output[CommonKeys.LABEL] = self._get_label(sample)

        # Apply transforms and return it
        return self.transform(output) if self.transform else output

    def _evaluate_patch_locations(self, sample: dict):
        """Calculate the location for each patch based on the mask at different resolution level

        samples is a dict with keys: image, location, level, size etc.

        """
        patch_size = self._get_size(sample)
        patch_level = self._get_level(sample)
        wsi_obj = self._get_wsi_object(sample)

        self._write_lookup_tables(wsi_obj)

        mask_level = 0

        mask_object: PyVipsReader = PyVipsReader(sample[self.tissue_label])
        mask_at_level: pyvips.Image = mask_object.read_level(self.mask_level)
        mask_ratio = int(mask_object.downsamples[self.mask_level])

        mask_np = mask_at_level.numpy()

        #  get all indices for non-zero pixels
        # np.vstack(mask_np.nonzero()).T
        # nonzero() returns tuple of two arrays, one for each dimension
        # vstack() turns them into a single array of shape (N, 2)
        # T transposes the array to shape (2, N), so each tuples is a location
        mask_locations = np.vstack(mask_np.nonzero())
        # get the first member for top bound and max for left bound
        start_pos = (
            mask_locations[0].min() * mask_ratio,
            mask_locations[1].min() * mask_ratio,
        )
        # get bottom and right bounds
        end_pos = (
            mask_locations[0].max() * mask_ratio,
            mask_locations[1].max() * mask_ratio,
        )

        # iter_patch_position lacks a end_pos argument, hence we compute the desired size beforehand
        fake_image_size = end_pos

        patch_locations = np.array(
            list(
                iter_patch_position(
                    image_size=fake_image_size,
                    patch_size=patch_size,
                    start_pos=start_pos,
                    overlap=0.0,
                    padded=True,
                )
            )
        )

        # fill out samples with location and metadata
        sample[WSIPatchKeys.SIZE.value] = patch_size
        sample[WSIPatchKeys.LEVEL.value] = patch_level
        sample[ProbMapKeys.NAME.value] = os.path.basename(sample[CommonKeys.IMAGE])
        sample[ProbMapKeys.COUNT.value] = len(patch_locations)
        sample[ProbMapKeys.SIZE.value] = np.array(
            self.wsi_reader.get_size(wsi_obj, self.mask_level)
        )
        return [
            {
                **sample,
                WSIPatchKeys.LOCATION.value: np.array(loc),
                ProbMapKeys.LOCATION.value: mask_loc,
            }
            for loc, mask_loc in zip(
                patch_locations, mask_locations.T
            )  # change mask location to coord pairs
        ]

    def to_json(self):
        """Converts the dataset to JSON format."""
        return json.dumps(self.data, indent=4, sort_keys=True)

    def get_sample(self, index):
        return self.data[index]


if __name__ == "__main__":
    import sys

    from monai.data import DataLoader

    print(
        f"sys.argv[1] = {sys.argv[1]} \n sys.argv[2] = {sys.argv[2]}"  # \n sys.argv[3] = {sys.argv[3]}"
    )

    annos_of_interest: list[str] = [
        "Tumor_vital",
        "diffuse tumor growth in soft tissue",
        "Angioinvasion",
        "Tumor_necrosis",
        "Tumor_regression",
        "Tissue",
    ]

    ds = BinMaskedPatchWSIDataset(
        [{"image": sys.argv[1], "Tissue": sys.argv[2]}],
        mask_level=0,
        patch_size=40000,
        reader="cuCIM",
        tissue_label="Tissue",
        masks=annos_of_interest,
    )
    print("dataset created")
    print(f"len(ds) = {len(ds)}")

    dl = DataLoader(ds, batch_size=1, num_workers=10)
    print(f"len(dl) = {len(dl)}")
    print()

    dl_iter = iter(dl)
    batch = next(dl_iter)
    print(batch.shape)
    print(batch)
