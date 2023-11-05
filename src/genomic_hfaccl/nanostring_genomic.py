import argparse
from datetime import datetime
import logging
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Subset
from accelerate import Accelerator
from src.misc.losses import FastCPH
from src.misc.metrics import concordance_index
from src.datasets.rcc_dataset import RCCDataset, rcc_to_csv
from src.networks.genomic import SNNet, simpleNet
from accelerate.tracking import (
    WandBTracker,
    TensorBoardTracker,
)
from accelerate.utils import (
    ProjectConfiguration,
    pad_across_processes,
    extract_model_from_parallel,
    set_seed,
)
from accelerate.logging import get_logger
from src.misc.utils import save_network
import tensorflow

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


def setup_data(config, accelerator, accl_logger):
    dataset = RCCDataset(config["rcc_path"], config["surv_path"])

    train_ds = Subset(dataset, range(config['val_set_size'], len(dataset)))
    val_ds = Subset(dataset, range(0, config['val_set_size']))
    test_ds = []

    # train_ds, val_ds, test_ds = random_split(
        # dataset,
        # config["val_set_size"],
        # generator=torch.Generator().manual_seed(int(config["seed"])),
    # )

    # set up data loaders
    dataloader_train = DataLoader(
        train_ds,
        batch_size=len(train_ds),#config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    dataloader_eval = DataLoader(
        val_ds,
        batch_size=len(val_ds),#config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    # dataloader_test = DataLoader(
    #     test_ds,
    #     batch_size=config["batch_size"],
    #     shuffle=False,
    #     num_workers=config["num_workers"],
    # )

    config = config | {"train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds)}
    accl_logger.info(f'"train_size": {len(train_ds)}, "val_size": {len(val_ds)}, "test_size": {len(test_ds)}')

    return dataloader_train, dataloader_eval, None#dataloader_test

##########################################################################################
def train(config, accelerator):
    seed = int(config["seed"])
    set_seed(seed)
    torch.manual_seed(seed)

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

    dataloader_train, dataloader_eval, dataloader_test = setup_data(
        config, accelerator, acc_logger
    )

    # set up model
    model = SNNet(input_dim=750, feature_size=250, elu=False, init_max=False, dropout_rate=0.25)

    # set up optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = FastCPH()
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

    # move model to device and set it to train mode
    (
        model,
        optimizer,
        dataloader_train,
        dataloader_eval,
        dataloader_test,
    ) = accelerator.prepare(
        model, optimizer, dataloader_train, dataloader_eval, dataloader_test
    )
    model.train()

    best_metric = 0.0
    best_val_loss = 0.0
    total_steps = 1

    # train loop
    for epoch in range(config["num_epochs"]):
        ##################################################################################
        # train step
        train_losses: list[float] = []
        train_risks: list[float] = []
        train_times: list[float] = []
        train_events: list[float] = []

        model.train()
        for step, batch in enumerate(dataloader_train):
            inputs, times, events, sample_id, genes = batch

            _, risks = model(inputs)
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
            optimizer.zero_grad()

            total_steps += step

        
        ##################################################################################
        # validation step
        val_losses: list[float] = []
        val_risks: list = []
        val_times: list = []
        val_events: list = []

        model.eval()
        for step, batch in enumerate(dataloader_eval):
            inputs, times, events, sample_id, genes = batch

            with torch.no_grad(): 
                _, risks = model(inputs)

            loss: torch.Tensor = criterion(risks, (times, events))
            val_risks.extend(risks.flatten().tolist())
            val_times.extend(times.flatten().tolist())
            val_events.extend(events.flatten().tolist())
            accelerator.log(
                {"loss/val_loss": loss.item(), "iter/val_step": total_steps},
                step=total_steps,
            )
            val_losses.append(loss.item())
            total_steps += step

        ##################################################################################
        train_losses_epoch_mean = torch.tensor(train_losses).nanmean()
        val_losses_epoch_mean = torch.tensor(val_losses).nanmean()

        train_cidx = concordance_index(train_risks, train_times, train_events)

        eval_cidx = concordance_index(val_risks, val_times, val_events)


        acc_logger.info(f"Train-Loss Epoch {epoch}: {train_losses_epoch_mean.item()}")
        acc_logger.info(f"Train-CI   Epoch {epoch}: {train_cidx}")
        acc_logger.info(f"Val-Loss   Epoch {epoch}: {val_losses_epoch_mean.item()}")
        acc_logger.info(f"Val-CI     Epoch {epoch}: {eval_cidx}")
        acc_logger.info(f'learning_rate: {optimizer.param_groups[0]["lr"]}')

        accelerator.log(
            {
                # val
                "performance/concordance_index_train": train_cidx,
                "loss/train_loss_epoch": train_losses_epoch_mean.item(),
                # train
                "performance/concordance_index_val": eval_cidx,
                "loss/val_loss_epoch": val_losses_epoch_mean.item(),
                # general
                "lr/learning_rate": optimizer.param_groups[0]["lr"],
                "iter/epoch": epoch,
            },
            step=epoch,
        )
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
        # if accelerator.is_main_process:
            # accelerator.save_state(config['output_dir'])    
    
    # data = torch.rand(inputs.shape)

    # import pdb; pdb.set_trace()
    dummy = accelerator.unwrap_model(model)
    save_network(
        dummy,
        batch[0],
    )


    accelerator.end_training()
    accelerator.free_memory()

##########################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
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
        "lr": 1e-4,
        "num_epochs": 2500,
        "seed": 42,
        "num_workers": 5,
        "base_dir": "/media/sciobiome/DATA/sklein_tmp/",# "/data2/projects/DigiStrudMed_sklein/",
        "experiment_name": os.path.split(__file__)[-1].split(".")[0],
        "val_set_size": 6,
        "annos_of_interest": [
            "Tissue",
            "Tumor_vital",
            "Angioinvasion",
            "Tumor_necrosis",
            "Tumor_regression",
        ],
        "date": datetime.now().strftime("%Y-%m-%d"),
    } | vars(args)
    config["rcc_path"] = config["base_dir"] + "DigiStrucMed_Braesen/NanoString_RCC/"
    config["surv_path"] = config["base_dir"] + "survival_status.csv"

    ######################################################################################

    proj_conf = ProjectConfiguration(
        project_dir=config["base_dir"] + "huggingface/",
        logging_dir=config["base_dir"] + "huggingface/logs",
        automatic_checkpoint_naming=True,
        total_limit=5,
        iteration=0,
    )

    tensorboard_logger: TensorBoardTracker = TensorBoardTracker(
        run_name=config["experiment_name"],
        logging_dir=proj_conf.logging_dir + "/tensorboard",
    )
    wandb_logger: WandBTracker = WandBTracker(
        run_name="Thesis",  # actually project name
        dir=proj_conf.logging_dir,
        group=config["experiment_name"],  # filename
        job_type=config["job_type"],  # current experiment
        name=args.run_name
        or config[
            "date"
        ], # name of the run
        magic=True,
        save_code=True,
        mode="online",
        sync_tensorboard=True,
    )

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        # gradient_accumulation_steps=config["grad_accum_steps"],
        log_with=[wandb_logger, tensorboard_logger],
        project_dir=config["base_dir"] + "huggingface/" + config["experiment_name"] + "/",
        project_config=proj_conf,
        # FIXME: maybe set this to false so we do not have to consider grad accums and just step on every iter?
        # step_scheduler_with_optimizer=ASKD,  # loss fluctuates a lot, so we only step the scheduler after each epoch
    )
    with accelerator.autocast():
        train(config, accelerator)
