import argparse
from datetime import datetime
import logging
import os
import random
import sys
import numpy as np

import torch
from torch.optim import AdamW, Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch import nn, optim

from accelerate import Accelerator, DistributedType
from accelerate.utils import (
    ProjectConfiguration,
    pad_across_processes,
    extract_model_from_parallel,
    set_seed,
)

from src.datasets.base_dataset import dataset_globber
from src.datasets.downsample_dataset import DownsampleMaskDataset
from src.networks.pathology import CoxResNet, CoxEffNet
from src.misc.metrics import (
    ConcordanceIndex,
    ConcordanceMetric,
    concordance_index,
    brier_score,
)
from src.misc.losses import CoxNNetLoss, FastCPH

from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from accelerate.logging import get_logger

from accelerate.tracking import (
    WandBTracker,
    TensorBoardTracker,
)

torch.autograd.set_detect_anomaly(False, check_nan=True)

def save_model(
    args,
    model,
    optimizer,
    best_metric,
    epoch,
    val_losses_epoch_mean,
    output_dir,
    accelerator,
    scheduler=None,
):
    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, output_dir)
        # create args.outputdir if not exists
        os.makedirs(args.output_dir, exist_ok=True)

    best_model = extract_model_from_parallel(model)
    best_model = accelerator.unwrap_model(best_model)
    accelerator.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": best_model.state_dict(),
            "lr_scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None
            else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_concordance_index": best_metric,
            "best_loss": val_losses_epoch_mean,
        },
        output_dir + ".pth",
    )

def setup_data(
    config: dict, acc_logger=None,
) -> tuple[DataLoader, DataLoader, DataLoader, DownsampleMaskDataset]:
    """Download datasets and create dataloaders

    Parameters
    ----------
    config: needs to contain `base_dir`, `batch_size`, `batch_size`, and `num_workers`
    """

    input_dicts = dataset_globber(
        config["base_dir"] + "DigiStrucMed_Braesen/all_data/",
        config["base_dir"] + "survival_status.csv",
    )
    img_transform = transforms.Compose(
        [
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0
            ),
        ]
    )

    stack_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.RandomRotation((0, 180))
        ]
    )
    

    overfit: float = config["overfit"]
    if overfit < 1.0:
        assert overfit > 0, "overfit must be a percentage of [0.0,1.0]"
        input_dicts: list[dict] = random.sample(input_dicts, int(len(input_dicts) * config["overfit"]))


    if acc_logger is not None:
        acc_logger.info(f"Value of overfit {overfit}")
        acc_logger.info(
            f"Overfitting on {overfit * 100}% of the data, no. of samples: {len(input_dicts)}"
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
        cache=True,
        cache_file=config["cache_path"],
        target_level=config["target_level"],
        transform1=img_transform,
        transform2=stack_transform,
    )

    train_ds, val_ds, test_ds = random_split(
        dataset,
        config["data_split"],
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )

    config = config | {"train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds)}

    dataloader_train = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        sampler=None,
    )
    dataloader_eval = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    dataloader_test = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    return dataloader_train, dataloader_eval, dataloader_test, dataset


def train(config, accelerator, args):
    seed = int(config["seed"])

    set_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    accelerator.init_trackers(
        config["experiment_name"], 
        {
            # convert some of the config to strings for tensorboard
            k: str(v)
            if not isinstance(v, int | float | str | bool | torch.Tensor)
            else v
            for k, v in config.items()
        }
    )

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    num_epochs = int(config["num_epochs"])


    # TODO: method extract: setup_logger
    acc_logger = get_logger(
        __name__,
        log_level="DEBUG",
    )
    log_file_handler = logging.FileHandler(
        filename=f"{config['experiment_name']}.log",
        mode="w",
    )
    acc_logger.logger.addHandler(log_file_handler)
    acc_logger.logger.addHandler(logging.StreamHandler(sys.stdout))
    # End of extract

    # Instantiate loss
    criterion = FastCPH()

    train_dataloader, eval_dataloader, test_dataloader, _ = setup_data(config, acc_logger)
    model = CoxEffNet(input_channels=8, feature_size=1000,)#act=nn.Sigmoid()) # type: ignore
    model = accelerator.prepare(model)  # FSDP: do this before init of optimizer

    # TODO: method extract "setup optimizer"
    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=config["lr"])

    # Instantiate scheduler
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=lr,
    #     total_steps=num_epochs * len(train_dataloader) // config["gradient_accumulation_steps"],
    #     # steps_per_epoch=len(train_dataloader),
    #     # epochs=num_epochs,
    #     anneal_strategy="cos",
    #     final_div_factor=25,
    # )

    swa_model = None
    if args.swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = optim.swa_utils.SWALR(
            optimizer, swa_lr=0.05, anneal_epochs=10, anneal_strategy="cos"
        )
        swa_start = args.swa
    # End of extract

    to_accelerate = [
        optimizer,
        train_dataloader,
        eval_dataloader,
        # scheduler,
    ]

    (
        optimizer,
        train_dataloader,
        eval_dataloader,
        # scheduler,
    ) = accelerator.prepare(
        *to_accelerate, device_placement=[True for _ in to_accelerate]
    )

    # accelerator.register_for_checkpointing(scheduler)

    # TODO: Method extract: "train"
    starting_epoch = 0

    # TODO: method extract "run_epoch"
    best_metric = 0.0
    best_val_loss = 0.0
    total_steps = 1

    for epoch in range(starting_epoch, num_epochs):
        ##################################################################################
        # TODO: method extract "run_train"
        # disable regularization if overfitting
        train_losses: list[float] = []
        train_risks: list[float] = []
        train_times: list[float] = []
        train_events: list[float] = []

        model.train(not config['overfit'] < 1.0) # FIXME: might cause issues
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                (inputs, times, events) = list(batch.values())[:3]

                _, risks = model(inputs)

                assert len(risks) == len(inputs), f"len(risks)={len(risks)} != len(inputs)={len(inputs)}"

                train_risks.extend(risks.flatten().tolist())
                train_times.extend(times.flatten().tolist())
                train_events.extend(events.flatten().tolist())

                loss: torch.Tensor = criterion(risks, (times, events))
                accelerator.log(
                    {"loss/train_loss": loss.item(), "iter/train_step": total_steps},
                    step=total_steps,
                )
                train_losses.append(loss.item())

                accelerator.backward(loss)
                optimizer.step()
                # if scheduler is not None and not accelerator.optimizer_step_was_skipped:
                #     scheduler_step()
                
                ## Reduc on plateau
                # scheduler.step(loss)

                optimizer.zero_grad()
            total_steps += step

        ##################################################################################
        # TODO: method extract "run_val"
        model.eval()
        val_losses: list[float] = []
        val_risks: list = []
        val_times: list = []
        val_events: list = []
        total_steps += 1
        for step, batch in enumerate(eval_dataloader):
            (inputs, times, events) = list(batch.values())[:3]

            with torch.no_grad():
                _, risks = model(inputs)
                val_risks.extend(risks.flatten().tolist())
                val_times.extend(times.flatten().tolist())
                val_events.extend(events.flatten().tolist())

                loss: torch.Tensor = criterion(risks, (times, events))
                accelerator.log(
                    {"loss/val_loss": loss.item(), "iter/val_step": total_steps},
                    step=total_steps,
                )
                val_losses.append(loss.item())
            total_steps += step

        ##################################################################################
        train_losses_epoch_mean = torch.tensor(train_losses).nanmean()
        train_cidx = concordance_index(train_risks, train_times, train_events)
        val_losses_epoch_mean = torch.tensor(val_losses).nanmean()
        eval_cidx = concordance_index(val_risks, val_times, val_events)

        acc_logger.info(f"Train-Loss Epoch {epoch}: {train_losses_epoch_mean.item()}")
        acc_logger.info(f"Train-CI   Epoch {epoch}: {train_cidx}")
        acc_logger.info(f"Val-Loss   Epoch {epoch}: {val_losses_epoch_mean.item()}")
        acc_logger.info(f"Val-CI     Epoch {epoch}: {eval_cidx}")

        accelerator.log(
            {
                "loss/train_loss_epoch": train_losses_epoch_mean.item(),
                "loss/val_loss_epoch": val_losses_epoch_mean.item(),

                "performance/concordance_index_train": train_cidx,
                "performance/concordance_index_val": eval_cidx,

                "lr/learning_rate": optimizer.param_groups[0]["lr"],
                "iter/epoch": epoch,
            },
            step=epoch,
        )
        ##################################################################################
        # Ending Epoch
        acc_logger.info(f'learning_rate: {optimizer.param_groups[0]["lr"]}')
        # lr_scheduler.step(train_losses_epoch_mean)

        # Checkpointing
        if eval_cidx > best_metric:
            best_metric = eval_cidx
            output_dir = (
                f"{config['date']}_{config['run_name']}_{config['job_type']}_CI"
            )
            save_model(
                args,
                model,
                optimizer,
                best_metric,
                epoch,
                val_losses_epoch_mean,
                output_dir,
                accelerator,
                # scheduler,
            )

        # Checkpointing
        if val_losses_epoch_mean > best_val_loss:
            best_val_loss = val_losses_epoch_mean
            output_dir = (
                f"{config['date']}_{config['run_name']}_{config['job_type']}_CPH"
            )
            save_model(
                args,
                model,
                optimizer,
                best_metric,
                epoch,
                val_losses_epoch_mean,
                output_dir,
                accelerator,
                # scheduler,
            )

        output_dir = f"epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.save_state(output_dir)

        # Average the model
        if epoch > args.swa and swa_model is not None:
            swa_model.update_parameters(model)

    ######################################################################################
    accelerator.end_training()
    accelerator.free_memory()

    if args.swa and swa_model is not None:
        # update_bn() does not work on our datase
        #   -> loop over the batches manually and call forward
        # torch.optim.swa_utils.update_bn(train_dataloader, swa_model)
        for batch in train_dataloader:
            (inputs, survtime, censor) = list(batch.values())[:3]
            with torch.no_grad():
                swa_model(inputs)

    # return eval_cidx

def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--job_type",
        type=str,
        required=True,
        help="Name of the job for wandb",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=False,
        help="Name of the run for wandb",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",  # THIS WAS 48 WHEN THE RUN WORKED
        type=int,
        default=1,
        help="The number of minibatches to be ran before gradients are accumulated.",
    )
    parser.add_argument(
        "--swa",
        type=int,
        default=0,
        help="How many epochs to wait before starting to average the model. default: 0, i.e. no averaging.e. no averaging)",
    )
    parser.add_argument(
        "--overfit",
        type=float,
        default=1.0,
        help="percentage of samples to use for overfitting (default: 1.0)",
    )
    args = parser.parse_args()
    ######################################################################################

    config: dict = {
        "lr": 1e-2,
        "num_epochs": 500,
        "seed": 42,
        "batch_size": 3,
        "target_level": 3,
        "num_workers": 10,
        "base_dir": "/data2/projects/DigiStrudMed_sklein/",
        "overfit": args.overfit or 1.0,
        "experiment_name": os.path.split(__file__)[-1].split(".")[0],
        "data_split": [0.5,0.5,0.0] if args.overfit < 1.0 else [0.8, 0.15, 0.05],
        "annos_of_interest": [
            "Tissue",
            "Tumor_vital",
            "Angioinvasion",
            "Tumor_necrosis",
            "Tumor_regression",
        ],
        "grad_accum_steps": args.gradient_accumulation_steps
        or 12,  # we have 24 batches in train loader
        "date": datetime.now().strftime("%Y-%m-%d"),
    } | vars(args)
    level: int = int(config["target_level"])
    config["cache_path"] = (
        config["base_dir"]
        + f"downsampled_datasets/cached_DownsampleDataset_level_{level}.json"
    )

    proj_conf = ProjectConfiguration(
        project_dir=config["base_dir"] + "huggingface/",
        logging_dir=config["base_dir"] + "huggingface/logs",
        automatic_checkpoint_naming=True,
        total_limit=0,
        iteration=0,
    )

    print(vars(proj_conf))

    ######################################################################################
    tensorboard_logger: TensorBoardTracker = TensorBoardTracker(
        run_name=config["experiment_name"],
        logging_dir=proj_conf.logging_dir + "/tensorboard",
    )
    wandb_logger: WandBTracker = WandBTracker(
        run_name="Thesis",  # actually project name
        dir=proj_conf.logging_dir,
        group=config["experiment_name"],  # filename
        job_type=config["job_type"],  # current experiment
        name=args.run_name or config[
            "date"
        ],  # if 'run_name' in config.keys() else None, # name of the run
        magic=True,
        save_code=True,
        mode="online",
        sync_tensorboard=True,
    )

    # Initialize accelerator
    accelerator = Accelerator(
        cpu=False,
        mixed_precision="bf16",
        gradient_accumulation_steps=config["grad_accum_steps"],
        log_with=[wandb_logger, tensorboard_logger],
        project_dir=config["base_dir"] + "huggingface/",
        project_config=proj_conf,
        # FIXME: maybe set this to false so we do not have to consider grad accums and just step on every iter?
        # step_scheduler_with_optimizer=ASKD,  # loss fluctuates a lot, so we only step the scheduler after each epoch
    )
    with accelerator.autocast():
        train(config, accelerator, args)


if __name__ == "__main__":
    main()
