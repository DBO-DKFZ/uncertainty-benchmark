import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import registered Datamodules
from src.datamodules.camelyon_datamodules import *

# Import registered Litmodules
from src.pl_modules.basemodules import METRIC_MODE
from src.pl_modules.resnetmodules import *

from src.utils.BetterCLI import BetterCli
import jsonargparse

from src.utils.system import find_max_version, find_save

from eval import run_eval


def main(**kwargs):
    def add_arguments_to_parser(parser: jsonargparse.ArgumentParser):

        parser.add_argument("--name", required=True)
        parser.add_argument("--version", type=Optional[int], default=None)

        parser.add_argument("--checkpoint")
        parser.add_argument("--save_dir", default=os.environ["EXPERIMENT_LOCATION"])
        parser.add_argument("--checkpoint_metric", default="val_f1")
        parser.add_argument("--eval", type=Union[bool, str], default=False)
        parser.add_argument("--uncertainty_method", type=str, default=None)
        parser.add_argument("--output", type=str, default=None)  # Used to write eval predicitions into custom folder
        parser.add_argument("--augmentations", type=str, default=None)

    cli = BetterCli(
        run=False,
        instantiate=False,
        add_arguments_function=add_arguments_to_parser,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None, "env_prefix": None},
        auto_class=True,
        save_config_overwrite=True,  # Needed to perform multiple test runs
    )

    cli.instantiate_classes(trainer=False)
    config = cli.config

    # Custom setup of trainer
    add_callbacks = []
    add_trainer_args = {}

    experiment_base_path = Path(config.save_dir) / config.name
    if config["version"] is None:
        version = find_max_version(experiment_base_path) + 1
    else:
        version = config["version"]

    version_dir = experiment_base_path / f"version_{version}"
    checkpoint_dir = version_dir / "checkpoints"

    tensorboard_logger = TensorBoardLogger(str(experiment_base_path), name=None, version=version, sub_dir="logs")
    csv_logger = CSVLogger(str(experiment_base_path), name=None, version=version)

    print("Writing logs to " + str(version_dir))
    loggers = [tensorboard_logger, csv_logger]

    assert config.model.opti_metric in METRIC_MODE.keys()
    assert config.checkpoint_metric in METRIC_MODE.keys()

    model_checkpoint = ModelCheckpoint(
        version_dir / "checkpoints",
        filename="{epoch}-{" + str(config.checkpoint_metric) + ":.4f}",
        monitor=config.checkpoint_metric,
        mode=METRIC_MODE[config.checkpoint_metric],
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=False,
        save_weights_only=True,
    )

    # Dont save if using optuna
    if not "optuna_trial" in kwargs:
        add_callbacks.extend([model_checkpoint])
    else:
        add_trainer_args["enable_checkpointing"] = False

    if "optuna_trial" in kwargs:
        import optuna

        optuna_callback = optuna.integration.PyTorchLightningPruningCallback(
            kwargs["optuna_trial"], monitor=kwargs["optuna_target"]
        )

        add_callbacks.extend([optuna_callback])

    # Instantiate Trainer
    cli.instantiate_trainer(
        add_callbacks=add_callbacks,
        logger=loggers,
        # num_sanity_val_steps=0, # Moved to config
        # reload_dataloaders_every_n_epochs=1, # Moved to config
        # weights_save_path=checkpoint_dir,
        **add_trainer_args,
    )
    trainer = cli.trainer
    model = cli.model
    datamodule = cli.datamodule

    ckpt_save = config.get("checkpoint")

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_save)

    # Support for optuna
    if "optuna_trial" in kwargs:
        return trainer.callback_metrics[kwargs["optuna_target"]].item()

    best_model = model_checkpoint.best_model_path

    # Finalize by copying best save to final.ckpt. This marks the experiment as completed.
    if best_model is not None and trainer.is_global_zero:
        os.link(best_model, checkpoint_dir / "final.ckpt")
        best_model = checkpoint_dir / "final.ckpt"

    if cli.config["eval"] or cli.config["eval"] == "eval" and trainer.is_global_zero:
        run_eval(cli, version_dir, best_model)


if __name__ == "__main__":
    main()
