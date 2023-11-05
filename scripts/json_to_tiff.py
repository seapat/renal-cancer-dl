from multiprocessing import Pool
import os
from glob import glob
import sys
from cucim import CuImage
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from pyvips_wrapper import numpy2pyramid
from geojson_processing import geojson_to_mask, geojson_annotations
import os
import functools
import json
import sys

sys.path.append("/") # oterwhise python does not finde the patch apparently

save_path = "/data2/projects/DigiStrudMed_sklein/DigiStrucMed_Braesen/all_data/"

import logging
logging.basicConfig(filename='json_to_tiff.log', encoding='utf-8', level=logging.INFO, force=True)
logging.getLogger("pyvips").setLevel(logging.CRITICAL)
logging.getLogger("rasterio").setLevel(logging.CRITICAL)
logging.getLogger('gjson_logger').addHandler(logging.StreamHandler())

logger = logging.getLogger('logger')
logger.setLevel(level=logging.ERROR)

def setup_pacquo():

    install_path: str = "/home/sklein/QuPath-0.4.3"
    os.system(f"paquo get_qupath --install-path {install_path} 0.4.3")
    try:
        import paquo.projects
    except:
        os.system("rm .paquo.toml")
        os.system(f"echo qupath_dir='{install_path}' >> .paquo.toml")

        import paquo.projects

def qpproj_to_json(gproj_list: str, output_dir: str):
    from paquo.projects import QuPathProject


    for project in gproj_list:
        print(f"opening {project}")

        # qupath expects certain file endings, backups etc. append & thus invalidate the ending
        if not project.endswith(".qpproj"):
            old_project = project
            temp = project.split(".qpproj")
            project = temp[0] + temp[1] + ".qpproj"
            print(f"renaming {old_project} to {project}")
            os.rename(old_project, project) if os.path.exists(old_project) else print(f"already renamed") if os.path.exists(project) else print(f"{old_project} does not exist")

        with QuPathProject(project, mode='r') as qp:

            print("opened", qp.name)

            # iterate over the images
            for image in qp.images:

                if not image.entry_path.exists():
                    print(f"File does not exist {image.image_name}")
                # else:
                    # print(f"File exist at {image.image_name}")

                annotations = image.hierarchy#.annotations
                name = image.image_name.removesuffix(".svs")

                json_path = os.path.join(save_path, name + ".json")
                if os.path.exists(json_path):
                    # print(f"File already exists at {json_path}")
                    continue
                else:
                    with open(json_path, "w", encoding='utf-8') as file:
                        json.dump(annotations.to_geojson(), file, indent=None, ensure_ascii=True, allow_nan=False, )


def main_mp(input_dir, annos_of_interest, rerun = False):
    geojsons: set[str] = set([file.removesuffix(".json")  for file in glob(os.path.join(input_dir, "*.json"))])
    images: set[str] = set([file.removesuffix(".svs")  for file in glob(os.path.join(input_dir, "*.svs"))])
    files: list[str] = list(geojsons.intersection(images))
    logger.debug(f"ALL FILES W/ MATCHES: {[os.path.basename(file) for file in files]}")

    # chunks = [files[x:x+100] for x in range(0, len(files), 100)]

    cpus = os.cpu_count() or 1 

    with Pool(10) as p:
        print(zip(files, [input_dir for _ in range(len(files))]))
        p.map(functools.partial(worker, input_dir=input_dir, annos_of_interest=annos_of_interest, rerun=rerun), files, 1)

def main(input_dir, annos_of_interest, rerun = False):
    # setup_pacquo()
    # gproj_list = glob(os.path.join(input_dir, "*.qpproj"))

    geojsons: set[str] = set([file.removesuffix(".json")  for file in glob(os.path.join(input_dir, "*.json"))])
    images: set[str] = set([file.removesuffix(".svs")  for file in glob(os.path.join(input_dir, "*.svs"))])

    # only use files where we have json files
    # only those appear in qupath projects
    files: list[str] = list(geojsons.intersection(images))


    logger.debug(f"ALL FILES W/ MATCHES: {[os.path.basename(file) for file in files]}")
    # logger.debug(f"not geojsons found for {[os.path.basename(file) for file in images.difference(geojsons)]}")
    for file in files:
        worker(file, input_dir, annos_of_interest, rerun=rerun)
   

    print("Done")

def worker(file, input_dir, annos_of_interest, rerun=False):
    #  for file in files:
        geojson, image = file + ".json", file + ".svs"
        # logger.debug(f"Processing {os.path.basename(geojson)} and {os.path.basename(image)}")

        height: int
        width: int
        channels: int
        height, width, channels = CuImage(image).shape
        try:
            present_annos = geojson_annotations(geojson)
            logger.info(f"present_annos: {present_annos} for {os.path.basename(geojson)}")

            for anno in annos_of_interest:
                if anno in present_annos:
                    # remove suffix not necessary?
                    target_file = image.removesuffix(".svs") + f"-{anno}.tif"
                    logger.debug(f"Target file: {os.path.basename(target_file)}, original file: {os.path.basename(image)    }")
                    if target_file not in glob(os.path.join(input_dir, "*.tif")) or rerun:
                        # print(f"Creating {target_file}")
                        logger.debug(f"target file does not exist, creating...")
                        try:
                            masks_np: dict = geojson_to_mask(geojson, [anno], (height, width))
                            # logger.debug(f"masks_np: {masks_np.keys()}, anno: {anno}")
                            if anno == "Tissue":
                                assert "Tissue" in masks_np.keys(), "Tissue mask is missing"
                            for anno, arr in masks_np.items():
                                assert arr.shape == (height, width)
                                # print(f"{anno}--{arr.shape}")
                                numpy2pyramid(input_np=arr, path=target_file, tiff_tilesize=512)
                                assert os.path.exists(target_file), f"File {target_file} does not exist"
                        except KeyError as e:
                            logger.info(f"Error on {geojson}")
                            logger.info(f"Error on {image}")
                            logger.info(f"Error on creating {target_file}")
                            logger.info(f"Error: {e}")                   
                            continue

                    else:
                        logger.debug(f"File already exists {target_file}")

        except KeyError as e:
            logger.error(f"KeyError for {os.path.basename(geojson)}")
            logger.error(f"Error: {e}")                   



if __name__ == "__main__":

    annos_of_interest: list[str] = [
        "Tissue",
        "Tumor_vital",
        "diffuse_tumor_growth_in_soft_tissue",
        "Angioinvasion",
        "Tumor_necrosis",
        "Tumor_regression",
    ]

    main_mp(sys.argv[1], annos_of_interest)
