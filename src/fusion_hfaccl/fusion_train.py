import argparse
import logging
import os
import random
import sys
from datetime import datetime, timedelta

import numpy as np
import torch
from accelerate import Accelerator, DistributedType 
from accelerate.logging import get_logger
from accelerate.tracking import TensorBoardTracker, WandBTracker
from accelerate.utils import (
    ProjectConfiguration,
    extract_model_from_parallel,
    pad_across_processes,
    set_seed,
    DistributedDataParallelKwargs
)
from torch import nn, optim
from torch.optim import Adam, AdamW, lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.datasets.base_dataset import dataset_globber
from src.datasets.downsample_dataset import DownsampleMaskDataset
from src.datasets.nested_dataset import MultimodalDataset
from src.datasets.rcc_dataset import RCCDataset
from src.downsample_hfaccl import hugg_downsample
from src.misc.losses import CoxNNetLoss, FastCPH
from src.misc.metrics import brier_score, concordance_index
from src.networks.fusion import MultiSurv
from src.networks.pathology import CoxResNet

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


def setup_data(config, accl_logger):
    *_, wsi_dataset = hugg_downsample.setup_data(config, accl_logger)
    # rcc_dataset = RCCDataset(config["rcc_path"], config["surv_path"], sparse=len(wsi_dataset))

    dataset = MultimodalDataset(wsi_dataset, config["rcc_path"], config["surv_path"])

    train_ds, val_ds, test_ds = random_split(
        dataset,
        config["data_split"],
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )

    # set up data loaders
    dataloader_train = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
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

    config = config | {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }
    accl_logger.info(
        f'"train_size": {len(train_ds)}, "val_size": {len(val_ds)}, "test_size": {len(test_ds)}'
    )

    return dataloader_train, dataloader_eval, dataloader_test, dataset


def train(config, accelerator, args):
    seed = int(config["seed"])

    set_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TODO: method extract: setup_logger
    acc_logger = get_logger(
        __name__,
        log_level="DEBUG",
    )
    log_file_handler = logging.FileHandler(
        filename=f"{config['experiment_name']}_{config['method']}.log",
        mode="w",
    )
    acc_logger.logger.addHandler(log_file_handler)
    acc_logger.logger.addHandler(logging.StreamHandler(sys.stdout))
    # End of extract

    train_dataloader, eval_dataloader, test_dataloader, _ = setup_data(
        config, acc_logger
    )

    accelerator.init_trackers(
        config["experiment_name"],
        {
            # convert some of the config to strings for tensorboard
            k: str(v)
            if not isinstance(v, int | float | str | bool | torch.Tensor)
            else v
            for k, v in config.items()
        },
    )

    criterion = FastCPH()

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = MultiSurv(
        data_modalities=["wsi", "rcc"],
        fusion_method=config["method"],
        device=accelerator.device,
        modality_feature_size=1000,
    )
    model = accelerator.prepare(model)  # FSDP: do this before init of optimizer

    # Instantiate loss
    # TODO: method extract "setup optimizer"
    # Instantiate optimizer
    unwrapped = accelerator.unwrap_model(model)
    optimizer = AdamW(
        params=[
            {"params": unwrapped.nanostring_submodel.parameters(), "lr": 1e-5},
            {"params": unwrapped.wsi_submodel.parameters(), "lr": 1e-3},
        ],
        lr=config["lr"],
    )

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

    for epoch in range(starting_epoch, int(config["num_epochs"])):
        ##################################################################################

        # TODO: method extract "run_train"
        # disable regularization if overfitting
        train_losses: list[float] = []
        train_risks: list[float] = []
        train_times: list[float] = []
        train_events: list[float] = []

        model.train(not config["overfit"] < 1.0)  # FIXME: might cause issues
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                sample, (times, events), sample_id, stain_id, genes = batch

                features, risks = model(sample)

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

        acc_logger.info(f"train_risks: {train_risks}")
        acc_logger.info(f"train_times: {train_times}")
        acc_logger.info(f"train_events: {train_events}")

        train_cidx = concordance_index(train_risks, train_times, train_events)
        train_losses_epoch_mean = torch.tensor(train_losses).nanmean()

        ##################################################################################
        # TODO: method extract "run_val"
        model.eval()
        val_losses: list[float] = []
        val_risks: list[float] = []
        val_times: list[float] = []
        val_events: list[float] = []
        total_steps += 1
        for step, batch in enumerate(eval_dataloader):
            sample, (times, events), sample_id, stain_id, genes = batch

            with torch.no_grad():
                features, risks = model(sample)

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

        acc_logger.info(f"val_risks: {val_risks}")
        acc_logger.info(f"val_times: {val_times}")
        acc_logger.info(f"val_events: {val_events}")

        eval_cidx = concordance_index(val_risks, val_times, val_events)
        val_losses_epoch_mean = torch.tensor(val_losses).nanmean()
        ##################################################################################
        acc_logger.info(f"Train-Loss Epoch {epoch}: {train_losses_epoch_mean.item()}")
        acc_logger.info(f"Train-CI   Epoch {epoch}: {train_cidx}")
        acc_logger.info(f"Val-Loss Epoch {epoch}: {val_losses_epoch_mean.item()}")
        acc_logger.info(f"Val-CI   Epoch {epoch}: {eval_cidx}")

        accelerator.log(
            {
                "performance/concordance_index_train": train_cidx,
                "performance/concordance_index_val": eval_cidx,
                "loss/train_loss_epoch": train_losses_epoch_mean.item(),
                "loss/val_loss_epoch": val_losses_epoch_mean.item(),
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
            output_dir = f"{config['date']}_{config['method']}_{config['date']}_CI"
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
            output_dir = f"{config['date']}_{config['method']}_{config['date']}_CPH"
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

        output_dir = f"{config['experiment_name']}_{config['method']}_epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.save_state(output_dir)

        # Average the model
        if epoch > args.swa and swa_model is not None:
            swa_model.update_parameters(model)

    ######################################################################################

    model.eval()
    test_losses: list[float] = []
    test_risks: list[float] = []
    test_times: list[float] = []
    test_events: list[float] = []
    total_steps += 1
    for step, batch in enumerate(test_dataloader):
        sample, (times, events), sample_id, stain_id, genes = batch

        with torch.no_grad():
            features, risks = model(sample)

        test_risks.extend(risks.flatten().tolist())
        test_times.extend(times.flatten().tolist())
        test_events.extend(events.flatten().tolist())

        loss: torch.Tensor = criterion(risks, (times, events))
        test_losses.append(loss.item())
        total_steps += step

    test_cidx = concordance_index(test_risks, test_times, test_events)
    test_losses_mean = torch.tensor(test_losses).nanmean()
    ##################################################################################
    acc_logger.info(f"Test-Loss: {test_losses_mean.item()}")
    acc_logger.info(f"Test-CI:   {test_cidx}")

    accelerator.log(
        {
            "performance/concordance_index_test": test_cidx,
            "loss/test_loss": test_losses_mean.item(),
        },
    )

    if args.swa and swa_model is not None:
        # update_bn() does not work on our datase
        #   -> loop over the batches manually and call forward
        # torch.optim.swa_utils.update_bn(train_dataloader, swa_model)
        for batch in train_dataloader:
            sample, (times, events), sample_id, stain_id = batch
            with torch.no_grad():
                swa_model(sample)

    # return eval_cidx


def objective(trial, config, accelerator, args):
    config["lr"] = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    # config['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    pass


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="If passed, will train on the CPU."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    # parser.add_argument(
    #     "--job_type",
    #     type=str,
    #     required=True,
    #     help="Name of the job for wandb",
    # )
    # parser.add_argument(
    #     "--run_name",
    #     type=str,
    #     required=False,
    #     help="Name of the run for wandb",
    # )
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
        "lr": 1e-3,
        "num_epochs": 300,
        "seed": 42,
        "batch_size": 3,
        "target_level": 3,
        "num_workers": 15,
        "base_dir": "/data2/projects/DigiStrudMed_sklein/",
        "overfit": args.overfit or 1.0,
        "experiment_name": os.path.split(__file__)[-1].split(".")[0],
        "data_split": [0.5, 0.5, 0.0] if args.overfit < 1.0 else [0.8, 0.15, 0.05],
        "annos_of_interest": [
            "Tissue",
            "Tumor_vital",
            "Angioinvasion",
            "Tumor_necrosis",
            "Tumor_regression",
        ],
        "grad_accum_steps": args.gradient_accumulation_steps
        or 12,  # we have 24 batches in train loader
        "date": (datetime.now()).strftime("%Y-%m-%d"), #  - timedelta(days=1)
    } | vars(args)
    level: int = int(config["target_level"])
    config["cache_path"] = (
        config["base_dir"]
        + f"downsampled_datasets/cached_DownsampleDataset_level_{level}.json"
    )
    config["rcc_path"] = config["base_dir"] + "DigiStrucMed_Braesen/NanoString_RCC/"
    config["surv_path"] = config["base_dir"] + "survival_status.csv"

    proj_conf = ProjectConfiguration(
        project_dir=config["base_dir"] + "huggingface/",
        logging_dir=config["base_dir"] + "huggingface/logs",
        automatic_checkpoint_naming=True,
        total_limit=0,
        iteration=0,
    )
    print(vars(proj_conf))
    methods = ["sum", "prod"] #["max","embrace", "attention", "kronecker", "cat", "sum", "prod"] #["kronecker", "embrace", "cat", "max", "sum", "prod", "attention"]
    ######################################################################################
    for method in methods:
        config["method"] = method 

        tensorboard_logger: TensorBoardTracker = TensorBoardTracker(
            run_name=config["experiment_name"],
            logging_dir=proj_conf.logging_dir + "/tensorboard",
        )
        wandb_logger: WandBTracker = WandBTracker(
            run_name="Thesis",  # actually project name
            dir=proj_conf.logging_dir,
            group=config["experiment_name"], # + "_test_run",  # filename
            job_type=config["date"],
            name=config["method"],
            magic=True,
            save_code=True,
            mode="online",
            sync_tensorboard=True,
        )

        kwargs = []
        kwargs.append(DistributedDataParallelKwargs(find_unused_parameters = True))

        # Initialize accelerator
        accelerator = Accelerator(
            cpu=args.cpu,
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=config["grad_accum_steps"],
            log_with=[wandb_logger, tensorboard_logger],
            project_dir=config["base_dir"] + "huggingface/",
            project_config=proj_conf,
            # FIXME: maybe set this to false so we do not have to consider grad accums and just step on every iter?
            # step_scheduler_with_optimizer=ASKD,  # loss fluctuates a lot, so we only step the scheduler after each epoch
            kwargs_handlers=kwargs
        )

        with accelerator.autocast():
            train(config, accelerator, args)
            accelerator.end_training()
            accelerator.free_memory() 
            torch.cuda.empty_cache()
            import time
            time.sleep(60*5)


if __name__ == "__main__":
    main()
