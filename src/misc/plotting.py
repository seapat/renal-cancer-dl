import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torch import Tensor

from PIL import Image

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from matplotlib import colors
import matplotlib.legend_handler as lh
from matplotlib.lines import Line2D
import os
import matplotlib


def get_tab10_legend():
    # Get the tab10 colormap
    try:
        tab10 = matplotlib.colormaps["tab10"].colors
    except AttributeError:
        tab10 = matplotlib.cm.get_cmap("tab10").colors

    # Convert RGB to hex values
    colors_hex = [colors.rgb2hex(color) for color in tab10]
    custom_lines = [
        Line2D([0], [0], color=colors_hex[i], lw=4, alpha=0.6)
        for i in range(len(colors_hex))
    ]

    return custom_lines, colors_hex

# semantic_masks
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import torch


def semantic_masks(image, masks, titles: list[str] = [], fig_size: tuple = (10, 10)):
    if len(masks) > len(titles):
        titles = titles

    custom_lines, colors_hex = get_tab10_legend()

    drawn_masks = draw_segmentation_masks(image, masks, alpha=0.4, colors=colors_hex)

    fig, axs = plt.subplots()  # figsize=fig_size
    lines = axs.imshow(torch.einsum("chw -> hwc", drawn_masks))
    fig.legend(
        custom_lines,
        [x.replace("_", " ") for x in titles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    fig.tight_layout()
    plt.show()

    return fig, axs


# semantic_masks_no_tissue
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import torch


def semantic_masks_no_tissue(
    image, masks, titles: list[str] = [], fig_size: tuple = (10, 10)
):
    masks = masks[1:]
    if len(masks) > len(titles):
        titles = titles

    custom_lines, colors_hex = get_tab10_legend()

    drawn_masks = draw_segmentation_masks(image, masks, alpha=0.4, colors=colors_hex)

    fig, axs = plt.subplots(figsize=(6, 6))  # figsize=fig_size
    lines = axs.imshow(torch.einsum("chw -> hwc", drawn_masks))
    axs.legend(
        custom_lines,
        [x.replace("_", " ") for x in titles],
        loc="center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
    )
    axs.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )
    fig.tight_layout()
    # plt.show()

    return fig, axs

# plot the first image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms


def plot_image(
    image,
    mask,
    titles: list[str] = [],
):
    """Plot a single image and its associated mask"""

    fig, axs = plt.subplots(1, len(mask) + 1, figsize=(20, 20))
    axs[0].imshow(image.permute(1, 2, 0))
    for i in range(1, len(mask) + 1):
        axs[i].imshow(mask[i - 1].unsqueeze(-1), cmap="gray", vmin=0, vmax=1)  #
        plt.setp(axs[i].get_yticklabels(), visible=False)
    for ax in axs:
        ax.xaxis.tick_top()
    for ax, title in zip(axs, titles):
        ax.set_xlabel(title)

    fig.tight_layout()
    plt.show()

    return fig, axs


def split_and_denormalize_batch(sample):

    if len(sample['data'].shape) == 3:
        sample["data"] = sample["data"].squeeze()

    image, masks = sample["data"][:,:3], sample["data"][:,3:]

    print(image.shape, masks.shape)

    # unnormalize
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    image = image * std[None, :, None, None] + mean[None, :, None, None]
    # remove background, mask[0] == tissue
    image = image * masks[:,0]
    img = (image * 255).type(torch.uint8)
    # black -> white
    img[img == 0] = 255
    # to bool
    masks = masks.type(torch.bool)

    return img, masks


def split_and_denormalize(sample):
    image, masks = sample["data"].squeeze()[:3], sample["data"].squeeze()[3:]

    # unnormalize
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image = image * std[:, None, None] + mean[:, None, None]
    # remove background, mask[0] == tissue
    image = image * masks[0]
    img = (image * 255).type(torch.uint8)
    # black -> white
    img[img == 0] = 255
    # to bool
    masks = masks.type(torch.bool)

    return img, masks
