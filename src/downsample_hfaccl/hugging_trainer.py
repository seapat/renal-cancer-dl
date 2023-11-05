class HuggingTrainer:
    def __init__(self, config, accelerator):
        self.config = config
        self.accelerator = accelerator
        self.best_metric = 0.0
        self.best_val_loss = 0.0
        self.setup_data(self)

        set_seed(seed)
        torch.manual_seed(seed)
        # random.seed(seed)
        # np.random.seed(seed)

        # Instantiate the metric
        metric = ConcordanceIndex()

        # Instantiate loss
        self.criterion = FastCPH()

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = setup_data(config)
        # Instantiate the model (we build the model here so that the seed also control new weights initialization)
        model = CoxResNet(input_channels=8, freeze=False)
        self.model = accelerator.prepare(model) # FSDP: do this before init of optimizer 

        optimizer, lr_scheduler = self.setup_optimizer()

        to_accelerate = [
            optimizer,
            train_dataloader,
            eval_dataloader,
            lr_scheduler,
        ]
        (
            optimizer,
            train_dataloader,
            eval_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            *to_accelerate, device_placement=[True for _ in to_accelerate]
        )

        accelerator.register_for_checkpointing(self.lr_scheduler)

        self.train()


    def setup_data(self) -> tuple[DataLoader, DataLoader, DataLoader]:
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

        if overfit < 1.0:
            assert overfit > 0, "overfit must be a percentage of [0.0,1.0]"
            input_dicts =  input_dicts[ : int(len(input_dicts) * config["overfit"])],
            
            
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
        return dataloader_train, dataloader_eval, dataloader_test

    def setup_optimizer(self):
        optimizer = AdamW(params=model.parameters(), lr=lr)  # what does AdamW do different?

        # Instantiate scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=1e-4,
            # cooldown=50,
        )
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_dataloader),
            epochs=num_epochs,
            anneal_strategy="cos",
            final_div_factor=25,
        )

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
            lr_scheduler,
            criterion,
            metric,
        ]

        (
            optimizer,
            train_dataloader,
            eval_dataloader,
            lr_scheduler,
            criterion,
            metric,
        ) = accelerator.prepare(
            *to_accelerate, device_placement=[True for _ in to_accelerate]
        )

        accelerator.register_for_checkpointing(lr_scheduler)

    def setup_logging(self):
        # convert some of the config to strings for tensorboard
        self.accelerator.init_trackers(
            config["run_name"],
            {
                k: str(v)
                if not isinstance(v, int | float | str | bool | torch.Tensor)
                else v
                for k, v in config.items()
            },
        )

        self.acc_logger = get_logger(
            __name__,
            log_level="DEBUG",
        )
        log_file_handler = logging.FileHandler(
            filename=f"{config['run_name']}.log",
            mode="w",
        )
        self.acc_logger.logger.addHandler(log_file_handler)
        self.acc_logger.logger.addHandler(logging.StreamHandler(sys.stdout))
        
    def _load_checkpoint(self):

        if config.resume_from_checkpoint:
            if config.resume_from_checkpoint is not None or config.resume_from_checkpoint != "":
                self.acc_logger.info(f"Resumed from checkpoint: {config.resume_from_checkpoint}")
                self.accelerator.load_state(config.resume_from_checkpoint)
                path = os.path.basename(config.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[
                    -1
                ]  # Sorts folders by date modified, most recent checkpoint is the last
            training_difference = os.path.splitext(path)[0]
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1

            return starting_epoch

    def hugger(self, accelerator, config):


        for epoch in range(starting_epoch, num_epochs):
            self._run_train(accelerator, config)

            c_idx, brier, loss = self._run_val(accelerator, config)

            acc_logger.info(f'learning_rate: {optimizer.param_groups[0]["lr"]}')

            # Checkpointing
            if eval_cidx > best_metric:
                best_metric = eval_cidx
                output_dir = f"{config['date']}_{args.name}_CI"
                self._save_model(
                    args,
                    model,
                    optimizer,
                    lr_scheduler,
                    best_metric,
                    epoch,
                    val_losses_epoch_mean,
                    output_dir,
                    accelerator,
                )

            # Checkpointing
            if val_losses_epoch_mean > best_val_loss:
                best_val_loss = val_losses_epoch_mean
                output_dir = f"{config['date']}_{args.name}_CPH"
                self._save_model(
                    args,
                    model,
                    optimizer,
                    lr_scheduler,
                    best_metric,
                    epoch,
                    val_losses_epoch_mean,
                    output_dir,
                    accelerator,
                )

            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.save_state(output_dir)

            # Average the model
            if epoch > args.swa and swa_model is not None:
                swa_model.upda

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


    def _run_train(self, accelerator, config):
        model.train()
        train_losses: list[float] = []
        train_risks: list[float] = []
        train_times: list[float] = []
        train_events: list[float] = []

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                (inputs, times, events) = list(batch.values())[:3]

                risks = model(inputs)

                train_risks.extend(risks.flatten().tolist())
                train_times.extend(times.flatten().tolist())
                train_events.extend(events.flatten().tolist())

                loss: torch.Tensor = criterion(risks, (times, events))
                accelerator.log({"loss/train_loss": loss.item(), "iter/train_step": step}, step=step)
                train_losses.append(loss.item())

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step(loss)
                optimizer.zero_grad()

        train_metric = concordance_index(train_risks, train_times, train_events)
        train_losses_epoch_mean = torch.tensor(train_losses).nanmean()
        train_brier = brier_score(torch.as_tensor(train_events), torch.as_tensor(train_risks))
        acc_logger.info(f"Train-Loss Epoch {epoch}: {train_losses_epoch_mean.item()}")
        acc_logger.info(f"Train-CI   Epoch {epoch}: {train_metric}")
        accelerator.log(
            {
                "lr/learning_rate": optimizer.param_groups[0]["lr"],
                "performance/concordance_index_train": train_metric,
                "performance/brier_train": train_brier,
                "loss/train_loss_epoch": train_losses_epoch_mean.item(),
            },
            step=epoch,
        )

    def _run_val(self, accelerator, config):
        model.eval()
        val_losses: list[float] = []
        val_risks: list = []
        val_times: list = []
        val_events: list = []
        for step, batch in enumerate(eval_dataloader):
            (inputs, times, events) = list(batch.values())[:3]

            with torch.no_grad():
                risks = model(inputs)
                val_risks.extend(risks.flatten().tolist())
                val_times.extend(times.flatten().tolist())
                val_events.extend(events.flatten().tolist())

                loss: torch.Tensor = criterion(risks, (times, events))
                accelerator.log({"loss/val_loss": loss.item(), "iter/val_step": step}, step=step)
                val_losses.append(loss.item())

        eval_cidx = concordance_index(val_risks, val_times, val_events)
        eval_brier = brier_score(torch.as_tensor(val_events), torch.as_tensor(val_risks))
        val_losses_epoch_mean = torch.tensor(val_losses).nanmean()

        acc_logger.info(f"Val-Loss Epoch {epoch}: {val_losses_epoch_mean.item()}")
        acc_logger.info(f"Val-CI     Epoch {epoch}: {eval_cidx}")
        accelerator.log(
            {
                "performance/concordance_index_val": eval_cidx,
                "performance/brier_val": eval_brier,
                "loss/val_loss_epoch": val_losses_epoch_mean.item(),
                "iter/epoch": epoch,
            },
            step=epoch,
        )

        return eval_cidx, eval_brier, val_losses_epoch_mean

        def _save_model(
            args,
            model,
            optimizer,
            lr_scheduler,
            best_metric,
            epoch,
            val_losses_epoch_mean,
            output_dir,
            accelerator,
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
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_concordance_index": best_metric,
                    "best_loss": val_losses_epoch_mean,
                },
                output_dir + ".pth",
            )
