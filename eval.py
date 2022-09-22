import itertools
import os
import sys
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import registered Datamodules
from src.datamodules.camelyon_datamodules import *

# Import registered Litmodules
from src.pl_modules.basemodules import *
from src.pl_modules.resnetmodules import *

# Import custom callbacks
from src.pl_modules.basemodules import CustomWriter
from src.pl_modules.temperature_scaling import TemperatureScalingCallback

from src.utils.BetterCLI import BetterCli

from src.utils.system import find_max_version, find_save


def run_eval(cli: BetterCli, version_dir: Path, model_checkpoint: Union[str, Path]):

    # Clean up cli, just to be sure. Always make predictions on full dataset.
    if hasattr(cli, "config_init"):
        cli.config_init["trainer.limit_train_batches"] = 1.0
        cli.config_init["trainer.limit_train_batches"] = 1.0
        cli.config_init["trainer.limit_train_batches"] = 1.0
        cli.config_init["trainer.fast_dev_run"] = False
        cli.config_init["data.tumor_threshold"] = 0.0

    cli.config["trainer.limit_train_batches"] = 1.0
    cli.config["trainer.limit_train_batches"] = 1.0
    cli.config["trainer.limit_train_batches"] = 1.0
    cli.config["trainer.fast_dev_run"] = False
    cli.config["data.tumor_threshold"] = 0.0

    # Make directories to write logs and predictions
    eval_dir = version_dir / "logs"
    if cli.config["output"] is not None:
        output_dir = version_dir / cli.config["output"]
    else:
        output_dir = version_dir / "predictions"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Remove lighning_logs subdirectory
    tensorboard_logger = TensorBoardLogger(save_dir=str(eval_dir), name=None, version="")
    csv_logger = CSVLogger(save_dir=str(eval_dir), name=None, version="")

    writer_callback = CustomWriter(output_dir=output_dir, write_interval="epoch", name=Path(model_checkpoint).stem)

    loggers = [tensorboard_logger, csv_logger]
    callbacks = [writer_callback]

    cli.instantiate_classes(trainer=False)
    datamodule = cli.datamodule

    datamodule.setup("predict")
    val_dataloader = datamodule.val_dataloader()

    # Difference between Cam16 and Cam17
    test_dataloaders = datamodule.test_dataloader()
    if type(test_dataloaders) in [list, tuple]:
        test_id_dataloader, test_ood_dataloader = test_dataloaders
    else:
        test_id_dataloader = test_dataloaders
        test_ood_dataloader = None

    if cli.config["uncertainty_method"] == "temperature_scaling":
        ts_callback = TemperatureScalingCallback(validation_loader=val_dataloader)
        callbacks.append(ts_callback)

    # Instantiate Trainer
    cli.instantiate_trainer(add_callbacks=callbacks, logger=loggers)
    trainer = cli.trainer

    # ckpt_save = config.get("checkpoint")
    model = cli.model.load_from_checkpoint(model_checkpoint)
    if cli.config["augmentations"] is not None:
        model.set_transforms(cli.config["augmentations"])

    print(f"Writing predictions to {output_dir}")

    # Used by logging (LitTileClassifier and EnsembleTileClassifier) and optionally CustomWriter
    model.metric_suffix = "val"
    trainer.test(model, dataloaders=val_dataloader)
    model.metric_suffix = "id"
    trainer.test(model, dataloaders=test_id_dataloader)

    if test_ood_dataloader is not None:
        model.metric_suffix = "ood"
        trainer.test(model, dataloaders=test_ood_dataloader)

    # When training on centers [0, 1, 3], evaluate OOD specifically on center 2 and center 4
    if cli.config["data_class"] == "Camelyon17BaseDataModule" and cli.config["data.id_centers"] == [0, 1, 3]:
        datamodule = Camelyon17BaseDataModule(
            path=cli.config["data.path"],
            batch_size=cli.config["data.batch_size"],
            num_workers=cli.config["data.num_workers"],
            tumor_threshold=cli.config["data.tumor_threshold"],
            id_centers=cli.config["data.id_centers"],
            ood_centers=2,
            sampling_factor=cli.config["data.sampling_factor"],
            val_subset=cli.config["data.val_subset"],
            transformlib=cli.config["data.transformlib"],
        )
        datamodule.setup("predict")
        test_id_dataloader, test_ood_dataloader = datamodule.test_dataloader()
        model.metric_suffix = "ood2"
        trainer.test(model, dataloaders=test_ood_dataloader)

        datamodule = Camelyon17BaseDataModule(
            path=cli.config["data.path"],
            batch_size=cli.config["data.batch_size"],
            num_workers=cli.config["data.num_workers"],
            tumor_threshold=cli.config["data.tumor_threshold"],
            id_centers=cli.config["data.id_centers"],
            ood_centers=4,
            sampling_factor=cli.config["data.sampling_factor"],
            val_subset=cli.config["data.val_subset"],
            transformlib=cli.config["data.transformlib"],
        )
        datamodule.setup("predict")
        test_id_dataloader, test_ood_dataloader = datamodule.test_dataloader()
        model.metric_suffix = "ood4"
        trainer.test(model, dataloaders=test_ood_dataloader)


def whole_slide_eval(cli: BetterCli, version_dir: Path, model_checkpoint: Union[str, Path]):

    # Clean up cli, just to be sure. Always make predictions on full dataset.
    if hasattr(cli, "config_init"):
        cli.config_init["trainer.limit_train_batches"] = 1.0
        cli.config_init["trainer.limit_train_batches"] = 1.0
        cli.config_init["trainer.limit_train_batches"] = 1.0
        cli.config_init["trainer.fast_dev_run"] = False
        cli.config_init["data.tumor_threshold"] = 0.0
        cli.config.pop("data.path_Cam17", None)

    cli.config["trainer.limit_train_batches"] = 1.0
    cli.config["trainer.limit_train_batches"] = 1.0
    cli.config["trainer.limit_train_batches"] = 1.0
    cli.config["trainer.fast_dev_run"] = False
    cli.config["data.tumor_threshold"] = 0.0
    cli.config.pop("data.path_Cam17", None)

    # Make directories to write logs and predictions
    eval_dir = version_dir / "logs"
    output_dir = version_dir / "predictions"
    output_dir.mkdir(exist_ok=True, parents=True)

    writer_callback = CustomWriter(output_dir=output_dir, write_interval="epoch", name=Path(model_checkpoint).stem)

    callbacks = [writer_callback]

    cli.instantiate_classes(trainer=False)
    datamodule = cli.datamodule

    datamodule.setup("predict")

    # Difference between Cam16 and Cam17
    test_dataloaders = datamodule.test_dataloader()
    if type(test_dataloaders) in [list, tuple]:
        test_id_dataloader, test_ood_dataloader = test_dataloaders
    else:
        test_id_dataloader = test_dataloaders
        test_ood_dataloader = None

    # Instantiate Trainer
    cli.instantiate_trainer(add_callbacks=callbacks, logger=[])
    trainer = cli.trainer

    # ckpt_save = config.get("checkpoint")
    model = cli.model.load_from_checkpoint(model_checkpoint)
    if cli.config["augmentations"] is not None:
        model.set_transforms(cli.config["augmentations"])

    print(f"Writing predictions to {output_dir}")

    # Used by logging (LitTileClassifier and EnsembleTileClassifier) and optionally CustomWriter
    model.metric_suffix = "whole_id"
    results = trainer.predict(model, dataloaders=test_id_dataloader)
    writer_callback.write_on_epoch_end(cli.trainer, model, results)

    if test_ood_dataloader is not None and test_ood_dataloader.dataset is not None:
        model.metric_suffix = "whole_ood"
        results = trainer.predict(model, dataloaders=test_ood_dataloader)
        writer_callback.write_on_epoch_end(cli.trainer, model, results)


def main(**kwargs):

    p = argparse.ArgumentParser()
    p.add_argument("--evaluation_base_dir", type=str, nargs="+", default=False)
    p.add_argument("--version", type=str, nargs="+", default=False)
    p.add_argument("--checkpoint", default="final")

    a, rest = p.parse_known_args()

    if a.evaluation_base_dir is not False:

        path_list = a.evaluation_base_dir
        ckpt = a.checkpoint
        assert ckpt in ["last", "final", "last.ckpt", "final.ckpt"]
        if not ".ckpt" in ckpt:
            ckpt += ".ckpt"

        if not (isinstance(path_list, list) or isinstance(path_list, tuple)):
            path_list = [path_list]

        # Turn into Paths objects
        for i in range(len(path_list)):
            path_list[i] = Path(path_list[i])
            assert path_list[i].exists()

        # Build Cartesian Product
        if a.version is not False:
            versions = a.version
            if not (isinstance(versions, list) or isinstance(versions, tuple)):
                versions = [versions]

            tmp_path_list = []
            for exp_dir, version in itertools.product(path_list, versions):
                tmp_path_list.append(exp_dir / f"version_{version}")
                assert tmp_path_list[-1].exists()

            path_list = tmp_path_list

        found_ckpts = []

        for exp_path in path_list:
            exp_path = Path(exp_path)
            assert exp_path.exists(), f"Path {exp_path} does not exist."

            tmp_ckpts = [Path(i) for i in exp_path.glob(f"**/{ckpt}")]
            found_ckpts.extend(tmp_ckpts)

        # Search for a config for every checkpoint
        found_confs = []
        for ckpt in found_ckpts:
            assert ckpt.parents[0].stem == "checkpoints"

            config_file = ckpt.parents[1] / "logs" / "config.yaml"
            assert config_file.exists()

            found_confs.append(config_file)

        save_sysargv = sys.argv.copy()

        for ckpt, conf in zip(found_ckpts, found_confs):

            print(f"Evaluating {conf}")
            sys.argv = (
                [sys.argv[0]] + ["--config", str(conf), "--checkpoint", str(ckpt)] + rest
            )  # [sys.argv[0]] + ["--config", str(conf), "--checkpoint", str(ckpt)] + sys.argv[1:]

            main_eval()

            sys.argv = save_sysargv
    else:
        main_eval()


def main_eval():
    def add_arguments_to_parser(parser):

        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--uncertainty_method", type=str, default=None)

        parser.add_argument("--eval_lvl", choices=["tile", "slide"], default="tile")
        parser.add_argument("--skip_existing", action="store_true", default=False)
        parser.add_argument("--output", type=str, default=None)
        parser.add_argument("--augmentations", type=str, default=None)

        # --- leftovers
        parser.add_argument("--name", required=True)
        parser.add_argument("--version", type=Optional[int])
        parser.add_argument("--checkpoint_metric")
        parser.add_argument("--save_dir", default=os.environ["EXPERIMENT_LOCATION"])
        parser.add_argument("--eval")

    cli = BetterCli(
        run=False,
        instantiate=False,
        add_arguments_function=add_arguments_to_parser,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
        auto_class=True,
        save_config_callback=None,
    )

    config = cli.config
    config_path = Path(str(cli.config["config"][0]))
    model_checkpoint = Path(config.checkpoint)

    assert config_path.parts[-2] == "logs"
    version_dir = config_path.parents[1]

    if not model_checkpoint.exists():
        raise RuntimeError("Provided .ckpt does not exist")

    if config["eval_lvl"] == "tile":
        run_eval(cli, version_dir, model_checkpoint)
    else:
        whole_slide_eval(cli, version_dir, model_checkpoint)


if __name__ == "__main__":
    main()
