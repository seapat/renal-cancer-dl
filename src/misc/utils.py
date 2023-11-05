
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
import os
import tensorflow


def path_to_name(path: str) -> str:
    """Returns name of a file, without extension,
    from a given full path string."""
    _file = path.split("/")[-1]
    if len(_file.split(".")) == 1:
        return _file
    else:
        return ".".join(_file.split(".")[:-1])


def path_to_ext(path: str) -> str:
    """Returns extension of a file path string."""
    _file = path.split("/")[-1]
    if len(_file.split(".")) == 1:
        return ""
    else:
        return _file.split(".")[-1]

def show(imgs, dpi=None):
    # from torchvision

    plt.rcParams["savefig.bbox"] = "tight"

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    fig.set_dpi(dpi) if dpi else dpi
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(
            np.asarray(img), interpolation="none", #cmap="gray", vmin=0, vmax=1
        )
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def visualize(image: np.ndarray, downsample: int = 1) -> None:
    """
    Visualize the image.

    Args:
        image: The image to visualize.
        downsample: The downsample factor.

    Returns:
        None

    """

    # from cucim
    # https://github.com/rapidsai/cucim/blob/26283ce6b86073ee77a2c4f7c3b6e31c2b9fa14c/notebooks/Supporting_Aperio_SVS_Format.ipynb
    dpi = 80.0 * downsample
    height, width, _ = image.shape
    plt.figure(figsize=(width / dpi, height / dpi))
    plt.axis('off')
    plt.imshow(image)
    # plt.close('all')

def get_case_id(path: str) -> int:
    """Returns case ID from a given path string."""
    return int(os.path.basename(path).split(".")[0].split("-")[2])



def save_network_img(model, batch):
    # batch = next(iter(data.train_dataloader()))
    input_names = ["actual_input"]
    output_names = ["output"]
    dynamic_axes_dict = {
        "actual_input": {0: "   ", 2: "img_x", 3: "img_y"},
        "output": {0: "features", 1: "hazard"},
    }
    torch.onnx.export(
        model,
        batch, #.to(model.device),
        f"{sys.argv[1]}.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        dynamic_axes=dynamic_axes_dict,
    )

def save_network(model, batch):
    # batch = next(iter(data.train_dataloader()))
    input_names = ["actual_input"]
    output_names = ["output"]
    dynamic_axes_dict = {
        "actual_input": {0: "   "},
        "output": {0: "features", 1: "hazard"},
    }
    torch.onnx.export(
        model,
        batch, #.to(model.device),
        f"{sys.argv[1]}.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        dynamic_axes=dynamic_axes_dict,
        opset_version = 11,
    )