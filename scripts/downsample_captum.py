import os
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import (GradientShap, GuidedGradCam, InputXGradient,
                         IntegratedGradients, NoiseTunnel, Occlusion, Saliency)
from captum.attr import visualization as viz
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms

from src.misc.plotting import semantic_masks_no_tissue, split_and_denormalize
from src.datasets.base_dataset import dataset_globber
from src.datasets.downsample_dataset import DownsampleMaskDataset
from src.networks.pathology import CoxResNet

# def unwrap_model(inp):
    # return model(inp)[1]

class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[1]


def sir_plotalot(batch, config, wrapped_model, model):

    red_cmap = LinearSegmentedColormap.from_list(
        "custom red", [(0, "#ffffff"), (0.25, "#C7030D"), (1, "#C7030D")], N=256
    )
    green_cmap = LinearSegmentedColormap.from_list(
        "custom green", [(0, "#ffffff"), (0.25, "#007D0B"), (1, "#007D0B")], N=256
    )
    blue_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "blue"), (1, "blue")], N=256
    )
    duo_cmap = LinearSegmentedColormap.from_list('Random gradient 9655', (
        # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%209655=0:C7030D-40:BC3B3B-50:FFFFFF-60:8BBE90-100:007D0B
        (0.000, (0.780, 0.012, 0.051)),
        (0.400, (0.737, 0.231, 0.231)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.600, (0.545, 0.745, 0.565)),
        (1.000, (0.000, 0.490, 0.043))))

    image, masks = split_and_denormalize(batch)

    path = "../assets/captum/"
    basename = f"case{batch['case_id'].item()}-stain{batch['stain_id'].item()}-{'dead' if batch['uncensored'].item() else 'censored'}_{int(batch['surv_days'].item())}days"
    # basename = f"case{batch['case_id']}-stain{batch['stain_id']}-{'dead' if batch['uncensored'] else 'censored'}_{int(batch['surv_days'])}days"

    if os.path.exists(f"{path}Occlusion_pos_{basename}.png") or os.path.exists(f"{path}Occlusion_abs_{basename}.png"):
        return

    fig, axs = semantic_masks_no_tissue(
        image, masks, titles=config["annos_of_interest"][1:]
    )
    fig.savefig(dpi=300, fname=f"{path}masks_{basename}.png", transparent=True)

    ######################################################################################
    method = "integrated_gradients"
    integrated_gradients = IntegratedGradients(wrapped_model)
    attributions_ig = integrated_gradients.attribute(
        batch["data"].clone(),
        n_steps=50,
        internal_batch_size=1,
        # target=0,
    )
    fig_duo, ax_duo = viz.visualize_image_attr(
        np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image.cpu().detach().numpy(), (1, 2, 0)),
        method="blended_heat_map",
        cmap=blue_cmap,
        alpha_overlay=0.7,
        show_colorbar=True,
        sign="absolute_value",
        outlier_perc=2,
        use_pyplot=False,
    )
    fig_duo.savefig(dpi=300, fname=f"{path}{method}_abs_{basename}.png", transparent=True)

    ######################################################################################
    method = "guided_gradcam"
    n = 0
    while n<10:
        try:
            grad_cam = GuidedGradCam(wrapped_model, model.resnet.layer4[-1].conv3 ) # model.resnet.layer4[-1].conv3 #model.non_linear_downsample[0]
            attributions_gradcam = grad_cam.attribute(
                batch["data"].clone(),
                attribute_to_layer_input=True,
                interpolate_mode="area"
            )
            fig_abs, ax_abs = viz.visualize_image_attr(
                np.transpose(attributions_gradcam.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(image.cpu().detach().numpy(), (1, 2, 0)),
                method="blended_heat_map",
                cmap=blue_cmap,
                show_colorbar=True,
                alpha_overlay=0.7,
                sign="absolute_value",
                outlier_perc=2,
                use_pyplot=False,
                # title="Guided GradCam",
            )
            fig_abs.savefig(dpi=300, fname=f"{path}{method}_abs_{basename}.png", transparent=True)
            fig_pos, ax_pos = viz.visualize_image_attr(
                np.transpose(attributions_gradcam.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(image.cpu().detach().numpy(), (1, 2, 0)),
                method="blended_heat_map",
                cmap=green_cmap,
                show_colorbar=True,
                alpha_overlay=0.7,
                sign="positive",
                outlier_perc=2,
                use_pyplot=False,
            )
            fig_pos.savefig(dpi=300, fname=f"{path}{method}_pos_{basename}.png", transparent=True)
            fig_neg, ax_neg = viz.visualize_image_attr(
            np.transpose(attributions_gradcam.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(image.cpu().detach().numpy(), (1, 2, 0)),
            cmap=red_cmap,
            method="blended_heat_map",
            show_colorbar=True,
            alpha_overlay=0.7,
            sign="negative",
            outlier_perc=2,
            use_pyplot=False,
            )
            fig_neg.savefig(dpi=300, fname=f"{path}{method}_neg_{basename}.png", transparent=True)
            break
        except:
            n += 1
            continue

    ######################################################################################
    method = "saliency"

    saliency = Saliency(wrapped_model)
    sal_attribution = saliency.attribute(
        torch.cat([image, masks.type(torch.float)], dim=0).unsqueeze(0)
    )
    fig_sal, ax_sal = viz.visualize_image_attr(
        np.transpose(sal_attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image.cpu().detach().numpy(), (1, 2, 0)),
        method="blended_heat_map",
        cmap=blue_cmap,
        alpha_overlay=0.7,
        show_colorbar=True,
        sign="absolute_value",
        outlier_perc=2,
        use_pyplot=False,
    )
    fig_sal.savefig(dpi=300, fname=f"{path}{method}_{basename}.png", transparent=True)

    ######################################################################################
    method = "inputXgradient"

    inxgrad = InputXGradient(wrapped_model)
    inxgrad_attr = inxgrad.attribute(
        torch.cat([image, masks.type(torch.float)], dim=0).unsqueeze(0)
    )
    inxgrad_attr.shape
    fig_ixg, ax_ixg = viz.visualize_image_attr(
        np.transpose(inxgrad_attr.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image.cpu().detach().numpy(), (1, 2, 0)),
        method="blended_heat_map",
        cmap=blue_cmap,
        show_colorbar=True,
        alpha_overlay=0.7,
        sign="absolute_value",
        outlier_perc=2,
        use_pyplot=False,
    )
    fig_ixg.savefig(dpi=300, fname=f"{path}{method}_pos_{basename}.png", transparent=True)

    ######################################################################################
    # method = "Occlusion"

    # occl = Occlusion(wrapped_model)
    # occl_attr = occl.attribute(
    #     torch.cat([image, masks.type(torch.float)], dim=0).unsqueeze(0),
    #     sliding_window_shapes=(8, 50, 50),
    #     strides=(8, 50, 50),
    #     show_progress=True,
    # )
    # occl_attr.shape
    # fig_pos, ax_pos = viz.visualize_image_attr(
    #     np.transpose(occl_attr.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    #     np.transpose(image.cpu().detach().numpy(), (1, 2, 0)),
    #     method="blended_heat_map",
    #     # cmap=blue_cmap,
    #     show_colorbar=True,
    #     sign="absolute_value",
    #     outlier_perc=10,
    #     use_pyplot=False,
    # )
    # fig_pos.savefig(dpi=300, fname=f"{path}{method}_abs_{basename}.png", transparent=True)


if __name__=="__main__":
    config = {
        "seed": 42,
        "batch_size": 1,
        "target_level": 3,
        "num_workers": 10,
        "base_dir": "/media/sciobiome/DATA/sklein_tmp/",
        "overfit": 1.0,
        "data_split": [0.0, 0.0, 1.0],
        "annos_of_interest": [
            "Tissue",
            "Tumor_vital",
            "Angioinvasion",
            "Tumor_necrosis",
            "Tumor_regression",
        ],
        "cache_path": "/media/sciobiome/DATA/sklein_tmp/cached_DownsampleDataset_level_3.json",
    }

    input_dicts = dataset_globber(
        config["base_dir"] + "data", #"Scans-fuer-QupathProjekt-RCC-3Faelle-10062021/",
        config["base_dir"] + "survival_status.csv",
    )

    img_transform = transforms.Compose(
        [
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0),
        ]
    )
    stack_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ]
    )

    config["total_samples"] = len(input_dicts)
    config["inputs"] = input_dicts

    dataset: DownsampleMaskDataset = DownsampleMaskDataset(
        input_dicts,
        foreground_key="Tissue",
        image_key="image",
        label_key="surv_days",
        keys=config["annos_of_interest"],
        censor_key="uncensored",
        json_key="geojson",
        # cache=True,
        # cache_file=config["cache_path"],
        target_level=config["target_level"],
        transform1=img_transform,
        transform2=stack_transform,
    )

    train_ds, val_ds, test_ds = random_split(
        dataset,
        config["data_split"],
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    config = config | {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }

    dataloader_test = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    model = CoxResNet(input_channels=8, feature_size=1000)
    # load model from file
    model.load_state_dict(
        torch.load(
            config["base_dir"] + "2023-04-14_lr_0.01_2_fc_layer_CI.pth",
            map_location=torch.device("cpu"),
        )["model_state_dict"]
    )
    wrapped_model = WrappedModel(model)

    print(len(dataloader_test))

    for batch in dataloader_test:
        sir_plotalot(batch, config, wrapped_model, model)
        