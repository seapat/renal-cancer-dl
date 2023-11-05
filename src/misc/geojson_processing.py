import json
from glob import glob
import os

from rasterio import features
from shapely import geometry
# import openslide
from cucim import CuImage

from shapely.prepared import prep
from shapely.geometry import Point

import numpy as np

import logging
logging.basicConfig(encoding='utf-8', level=logging.DEBUG, force=True)
# logging.getLogger("pyvips").setLevel(logging.CRITICAL)
# logging.getLogger("rasterio").setLevel(logging.CRITICAL)
gjson_logger = logging.getLogger('gjson_logger')
gjson_logger.setLevel(level=logging.DEBUG)


def get_grid_locations(polygon, step_size, overlap=0, center=True) -> list[Point]:
    # inspird by https://stackoverflow.com/a/66017009
    valid_points = []

    # determine maximum edges
    # polygon = max(shapes["Tissue"].geoms, key=lambda a: a.area)
    latmin, lonmin, latmax, lonmax = map(
        # bound can be neg. if annotation outside of image, adjust so to be inside nad shiftable to centroid
        lambda x: max(x, (step_size / 2) if center else 0),
        polygon.bounds)

    # create prepared polygon
    prep_polygon = prep(polygon)

    # construct a rectangular mesh
    points = []
    for lat in np.arange(latmin, latmax, step_size - overlap):
        for lon in np.arange(lonmin, lonmax, step_size - overlap):
            # Cucim ignores sub-pixel precision
            # cutting decimals is faster than round and makes sure we dont have 1px gaps
            # points.append(Point((round(lat,4), round(lon,4))))
            points.append(Point(lat, lon))

    # validate if each point falls inside shape using
    # the prepared polygon
    # valid_points.extend(filter(prep_polygon.contains, points))

    # This adds the next patch evenen if it is not in the polygon anymore
    # Edges where the centroid lies outside but the patch still contains some tissue
    prev_added: bool = False
    for point in points:
        if prep_polygon.contains(point):
            valid_points.append(point)
            prev_added = True
        elif prev_added:
            valid_points.append(point)
            prev_added = False

    return valid_points

def geojson_to_shapely(gjson_file: str, target_annos: list[str]) -> dict[str, geometry.base.BaseGeometry]:
    """
    gjson_file: path to geojson file
    target_annos: list of annotations to extract
    output: dict of shapely shapes
    """

    # 1. READ JSON ARRAY from geojson file
    input_file = open(gjson_file)
    data = json.load(input_file)

    # 2. FILTER ANNOTATIONS FOR TISSUE AND TUMOR
    annos = []
    for dict_like in data:

        props = dict_like["properties"]
        if "classification" in props.keys():
            if props["objectType"] == "annotation":
                    if props["classification"]["name"] in target_annos:
                        annos.append(dict_like)
        else:
            gjson_logger.info("source: geojson_to_shapely")
            gjson_logger.info(f"file:  {gjson_file}")
            gjson_logger.info(f"Key 'classification' missing in props {props}")   
            continue


    # 3. CONVERT TO SHAPELY SHAPES
    shapes = {
        annotation["properties"]["classification"]["name"]: geometry.shape(
            annotation["geometry"]
        )
        for annotation in annos
    }

    return shapes

def geojson_annotations(gjson_file: str) -> list[str]:
    # 1. READ JSON ARRAY from geojson file
    input_file = open(gjson_file)
    data = json.load(input_file)
    # 2. FILTER ANNOTATIONS FOR TISSUE AND TUMOR
    annos = []
    for dict_like in data:
        props = dict_like["properties"]
        if "classification" in props.keys():
            if props["objectType"] == "annotation":
                annos.append(props["classification"]["name"])
        else:
            gjson_logger.info("source: geojson_annotations")
            gjson_logger.info(f"file:  {gjson_file}")
            gjson_logger.info(f"Key 'classification' missing in props {props}")   
            continue


    return annos

def geojson_to_mask(gjson_file, target_annos: list[str], img_hxw: tuple[int, int])-> dict[str, np.ndarray]:
    """
    gjson_file: path to geojson file
    target_annos: list of annotations to extract
    img_hxw: tuple of image dimensions (width, height)
    """

    # # 1. READ JSON ARRAY from geojson file
    # input_file = open(gjson_file)
    # data = json.load(input_file)

    # # 2. FILTER ANNOTATIONS FOR TISSUE AND TUMOR
    # annos = []
    # for dict_like in data:
    #     props = dict_like["properties"]
    #     if props["objectType"] == "annotation":
    #         if props["classification"]["name"] in target_annos:
    #             annos.append(dict_like)

    # # 3. CONVERT TO SHAPELY SHAPES
    # shapes = {
    #     annotation["properties"]["classification"]["name"]: geometry.shape(
    #         annotation["geometry"]
    #     )
    #     for annotation in annos
    # }

    shapes = geojson_to_shapely(gjson_file, target_annos)

    # 4. CONVERT TO MASK (TUMOR AND TISSUE)
    masks = {}
    # masks[case] = {}
    for name, curr_shape in shapes.items():
        mask = features.rasterize(
            [curr_shape],
            out_shape=img_hxw,
            all_touched=False,
        )#.astype(dtype=bool)
        masks[name] = mask

    return masks

def geojson_key_bounds(gjson_file, key: str)-> tuple[float, float, float, float]:
    """
    gjson_file: path to geojson file
    target_annos: list of annotations to extract
    key: name of the dict member for which to return the bounds

    retruns: tuple of bounds (minx, miny, maxx, maxy)
    """

    # # 1. READ JSON ARRAY from geojson file
    # input_file = open(gjson_file)
    # data = json.load(input_file)

    # # 2. FILTER ANNOTATIONS FOR TISSUE AND TUMOR
    # annos = []
    # for dict_like in data:
    #     props = dict_like["properties"]
    #     if props["objectType"] == "annotation":
    #         if props["classification"]["name"] == key:
    #             annos.append(dict_like)

    # # 3. CONVERT TO SHAPELY SHAPES
    # shapes = {
    #     annotation["properties"]["classification"]["name"]: geometry.shape(
    #         annotation["geometry"]
    #     )
    #     for annotation in annos
    # }
    shapes = geojson_to_shapely(gjson_file, [key])

    return shapes[key].bounds

def gjsondir_to_mask(json_dir, target_annos, img_dir):
    """
    json_dir: directory containing geojson files
    target_annos: list of annotations to extract
    image_dims: tuple of image dimensions (width, height)
    """

    images = sorted(glob(os.path.join(json_dir, "*.svs")))
    jsons = sorted(glob(os.path.join(img_dir, "*.json")))
    masks = {}
    for gjson, image in zip(jsons, images):

        _, filename = os.path.split(gjson)
        case, _ = os.path.splitext(filename)

        try:
            # Haematoxylin and eosin OR IHC
            case, stain_code = filename.split("~")[0].split(".")
        except:
            # Haemalun
            case, _, stain_code = filename.split("~")[0].split(".")

        # width, height = openslide.open_slide(image).dimensions
        width, height = CuImage(image).resolutions["level_dimensions"][0]
        case_masks = geojson_to_mask(gjson, target_annos, (height, width))
        masks[case] = case_masks

    return masks


if __name__ == "__main__":
    import sys

    print(
        f"sys.argv[1] = {sys.argv[1]} \n sys.argv[2] = {sys.argv[2]} \n sys.argv[3] = {sys.argv[3]}"
    )
    # json_file = "/Users/username/Downloads/geojson/"
    # target_annos = ["Tissue", "Tumor"]
    print(
        geojson_to_mask(
            sys.argv[1],
            # sys.argv[2], #[str(string) for string in sys.argv[2]],
            [
                x.strip()
                for x in sys.argv[2]
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .split(",")
            ],
            tuple(
                int(s) for s in sys.argv[3].replace("(", "").replace(")", "").split(",")
            ),
        )
    )
