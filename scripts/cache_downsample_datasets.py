from multiprocessing import Pool
import logging
import os
from datasets import DownsampleMaskDataset
from datasets import dataset_globber

# logging.basicConfig(filename=__name__, encoding='utf-8', level=logging.INFO, force=True)
logger = logging.getLogger("cache_downsample_dataset")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler("cache_downsample_dataset.log", mode="w"))


def create_downsampled_dataset(input_dicts, annos_of_interest, base_path, level):
    path = base_path + f"downsampled_datasets/cached_DownsampleDataset_level_{level}.json"

    logger.info(f"Starting creation of dataset at {path}")

    input_dataset = DownsampleMaskDataset(  # PatchMaskDataset(
        input_dicts,
        foreground_key="Tissue",
        image_key="image",
        label_key="surv_time",
        keys=annos_of_interest,
        censor_key="uncensored",
        json_key="geojson",
        cache=True,
        cache_file=path,
        # transform=True,
        target_level=level,
        # normalization_factors=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    )

    logger.info(
        f"Finished creating dataset with {len(input_dataset)} samples. Found at {path}"
    )


def main():
    annos_of_interest: list[str] = [
        "Tissue",
        "Tumor_vital",
        "diffuse_tumor_growth_in_soft_tissue",
        "Angioinvasion",
        "Tumor_necrosis",
        "Tumor_regression",
    ]

    levels = 4
    base_path = "/data2/projects/DigiStrudMed_sklein/"
    input_dir = base_path + "DigiStrucMed_Braesen/all_data/"

    input_dicts = dataset_globber(input_dir, base_path + "survival_status.csv")

    # logger.info(input_dicts[0])
    logger.info(f"Number of input dicts: {len(input_dicts)}")

    # finish this first
    # create_downsampled_dataset(input_dicts, annos_of_interest, base_path, 1)

    # for level in range(0,levels):
    with Pool(levels) as pool:
        # create_downsampled_dataset(input_dicts, annos_of_interest, base_path, level)
        # [pool.apply_async(create_downsampled_dataset) for i in range(4)]
        pool.starmap(
            create_downsampled_dataset,
            [
                (input_dicts, annos_of_interest, base_path, level)
                for level in [1, 2, 3, 4] # range(levels) #
            ],
        )


if __name__ == "__main__":
    main()
