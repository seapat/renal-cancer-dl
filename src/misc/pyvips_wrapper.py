"""Inspired by Slideflow
https://github.com/jamesdolezal/slideflow/blob/04f083e791bddb91b40077bc942d3397eca03c4a/slideflow/slide/backends/pyvips.py#L44

"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyvips
# import slideflow


import logging
logging.basicConfig(level=logging.WARNING)

def numpy2pyramid(input_np: np.ndarray, path: str, tiff_tilesize: int = 512, zstd_level: int = 3):
    '''Converts a numpy array to a pyramid tiff file.
    Args:
        input_np (np.ndarray): numpy array to convert
        path (str): path to save the tiff file
        tiff_tilesize (int): tile size for the tiff file
        zstd_level (int, optional): zstd compression level. Defaults to 3.
    '''
    img = pyvips.Image.new_from_array(
    input_np, #.astype("bool"),
    interpretation='b-w',
    )


    # print(f"img.filename: {img.filename}, img.height: {img.height}, img.width: {img.width}")
    
    img.tiffsave( # type: ignore
    path,
    pyramid=True, # type: ignore
    tile=True, # type: ignore
    compression="zstd", # type: ignore
    level=zstd_level, # type: ignore
    subifd=False, # type: ignore # NOT found in SVS files
    tile_width=tiff_tilesize, # type: ignore
    tile_height=tiff_tilesize, # type: ignore
    bigtiff=True, # type: ignore
) 

VIPS_FORMAT_TO_DTYPE: dict[str, type] = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

class PyVipsReader:

    def __init__(
        self,
        path: str,
    ) -> None:
        """Wrapper for Libvips to preserve cross-compatible functionality."""

        self.path = path
        self.loaded_downsample_levels = {}  # type: Dict[int, "pyvips.Image"]

        self.dimensions: List[Tuple[int, int]] = []
        self.tile_dimensions: List[Tuple[int, int]] = [] 
        self.downsamples: List[float] = [] 

        level_image = pyvips.Image.openslideload(self.path, level=0)  # type: ignore
        self.level_count = int(level_image.get('openslide.level-count'))  # type: ignore
        for level in range(self.level_count):
            self.downsamples.append(float(level_image.get(f"openslide.level[{level}].downsample")))
            self.dimensions.append(
                (
                    int(level_image.get(f"openslide.level[{level}].width")),
                    int(level_image.get(f"openslide.level[{level}].height")),
                )
            )
            self.tile_dimensions.append(
                (
                    int(level_image.get(f"openslide.level[{level}].tile-width")),
                    int(level_image.get(f"openslide.level[{level}].tile-height")),
                )
            )

        loaded_image:pyvips.Image = self.read_level(level=0)

        # Load image properties
        self.properties = {}
        for field in loaded_image.get_fields():
            self.properties.update({field: loaded_image.get(field)})
    
    def read_level(
        self,
        level: int = 0,
        access=pyvips.enums.Access.RANDOM,
        **kwargs,
    ) -> pyvips.Image:
        """Read a pyramid level."""

        result = None
        if level in range(self.level_count):
            if level in self.loaded_downsample_levels:
                result =  self.loaded_downsample_levels[level]
            else:
                image: pyvips.Image = pyvips.Image.tiffload(  # type: ignore
                    self.path, access=access, page=level, **kwargs
                )
                self.loaded_downsample_levels.update({level: image})
                result =  image
            assert result.width, result.height == self.dimensions[level]
            return result
        else:
            raise ValueError(
                f"Unavailable level {level}, aviailable levels: 0-{self.level_count-1}"
            )

    def read_region(
        self,
        location_at_0: Tuple[int, int],
        level: int,
        extract_size: Tuple[int, int],
        flatten: bool = False,
        resize_factor: Optional[float] = None,
    ) -> "pyvips.Image":
        """Extracts a region from the image at the given downsample level.

        Args:
            location_at_0 (Tuple[int, int]): Top-left location of the region
                to extract, using base layer coordinates (x, y)
            downsample_level (int): Downsample level to read.
            extract_size (Tuple[int, int]): Size of the region to read
                (width, height) using downsample layer coordinates.

        Returns:
            pyvips.Image: VIPS image.
        """
        base_level_x, base_level_y = location_at_0
        extract_width, extract_height = extract_size
        downsample_factor = self.downsamples[level]
        downsample_x = int(base_level_x / downsample_factor)
        downsample_y = int(base_level_y / downsample_factor)
        image: pyvips.Image = self.read_level(level=level)
        region: pyvips.Image = image.crop(downsample_x, downsample_y, extract_width, extract_height)  # type: ignore
        # Final conversions
        if flatten and region.bands == 4:
            region = region.flatten()  # removes alpha # type: ignore
        if resize_factor is not None:
            region = region.resize(resize_factor)  # type: ignore
        return region

    def read_from_pyramid(
        self,
        top_left: Tuple[int, int],
        window_size: Tuple[int, int],
        target_size: Optional[Tuple[int, int]],
        flatten: bool = False,
    ) -> "pyvips.Image":
        """Reads a region from the image using base layer coordinates.
        Performance is accelerated by pyramid downsample layers, if available.

        Args:
            top_left (Tuple[int, int]): Top-left location of the region to
                extract, using base layer coordinates (x, y).
            window_size (Tuple[int, int]): Size of the region to read (width,
                height) using base layer coordinates.
            target_size (Tuple[int, int]): Resize the region to this target
                size (width, height).

        Returns:
            pyvips.Image: VIPS image. Dimensions will equal target_size unless
            the window includes an area of the image which is out of bounds.
            In this case, the returned image will be cropped.
        """
        target_downsample: float = window_size[0] / target_size[0] if target_size else 1
        target_level: int = self.downsamples.index(min(self.downsamples, key=lambda value:abs(value-target_downsample)))
        image: pyvips.Image = self.read_level(level=target_level)

        resize_factor = self.downsamples[target_level] / target_downsample
        image: pyvips.Image = image.resize(resize_factor)  # type: ignore
        image: pyvips.Image = image.crop(
            int(top_left[0] / target_downsample),
            int(top_left[1] / target_downsample),
            min(target_size[0], image.width),  # type: ignore
            min(target_size[1], image.height),  # type: ignore
        )  # type: ignore

        # Final conversions
        if flatten and image.bands == 4:
            image = image.flatten()  # removes alpha # type: ignore

        return image

    # We rather make the network itself robust to color variations
    # def sf_normalize(self,
    #     pv_img: np.ndarray, normalizer: slideflow.norm.StainNormalizer, normalizer_target: str
    # ) -> np.ndarray:
    #     """Normalizes an image using a StainNormalizer object.

    #     Args:
    #         img (np.ndarray): Image.
    #         normalizer (StainNormalizer): StainNormalizer object.

    #     Returns:
    #         np.ndarray: Normalized image.
    #     """
        
    #     normalizer = slideflow.norm.StainNormalizer(normalizer)  # type: ignore
    #     if normalizer_target is not None:
    #         normalizer.fit(normalizer_target)

    #     if normalizer:
    #         try:
    #             pv_img = normalizer.rgb_to_rgb(pv_img)
    #         except Exception as e:
    #             # The image could not be normalized,
    #             # which happens when a tile is primarily one solid color
    #             # return None
    #             log.debug(f"The image could not be normalized: {e}")
    #             raise e

    #     return normalizer.normalize(pv_img)

    @ staticmethod
    def convert_to(
        pv_img: pyvips.Image, convert: str = ""
    ) -> Union[np.ndarray, pyvips.Image]:
        if convert and convert.lower() in ("jpg", "jpeg"):
            return pv_img.jpegsave_buffer()  # type: ignore
        elif convert and convert.lower() == "png":
            return pv_img.pngsave_buffer()  # type: ignore
        elif convert == "numpy":
            return pv_img.numpy() # type: ignore
        else:
            raise ValueError(
                f'Unknown conversion type: {convert}. Accepted values are "jpg", "jpeg", "png", and "numpy"'
            )

            # np.ndarray(buffer=mask_data.write_to_memory(), dtype=np.uint8, shape=[mask_data.height, mask_data.width, mask_data.bands]) 

    @staticmethod
    def resize(
        img: np.ndarray, crop_width: int, target_px: int
    ) -> pyvips.Image:
        """Resizes and crops an image using libpyvips.resize()

        Args:
            img (np.ndarray): Image.
            crop_width (int): Height/width of image crop (before resize).
            target_px (int): Target size of final image after resizing.

        Returns:
            np.ndarray: Resized image.
        """
        img_data = np.ascontiguousarray(img).data
        vips_image = pyvips.Image.new_from_memory(
            img_data, crop_width, crop_width, bands=3, format="uchar"
        )
        vips_image: pyvips.Image = vips_image.resize(target_px / crop_width)  # type: ignore

        return vips_image  # type: ignore