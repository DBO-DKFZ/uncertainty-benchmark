import argparse
import copy
import sys
import warnings
from typing import *
import os
from pytorch_lightning.callbacks import Callback

import pytorch_lightning
import torch
from jsonargparse import Namespace
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import (
    LightningCLI,
    LightningArgumentParser,
    _populate_registries,
    OPTIMIZER_REGISTRY,
    LR_SCHEDULER_REGISTRY,
)
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities.cloud_io import get_filesystem

from .betterargparse import ArgumentParser as b_ArgumentParser
import jsonargparse

from pathlib import Path

import random

from .system import find_max_version


def _get_chosen_data_model():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class")  # , choices=list(pytorch_lightning.utilities.cli.MODEL_REGISTRY.names))
    parser.add_argument("--data_class")  # , choices=list(pytorch_lightning.utilities.cli.DATAMODULE_REGISTRY.names))
    # Necessary for some reason. I really dont get it.
    parser.add_argument("--data")  # , choices=list(pytorch_lightning.utilities.cli.DATAMODULE_REGISTRY.names))
    parser.add_argument("--model")  # , choices=list(pytorch_lightning.utilities.cli.DATAMODULE_REGISTRY.names))

    parser.add_argument("--config", action="append", nargs="*")

    parsed_args, rest = parser.parse_known_args()
    parsed_args = vars(parsed_args)
    config_model_name = None
    parsed_model_name = parsed_args.get("model_class")

    config_data_name = None
    parsed_data_name = parsed_args.get("data_class")

    if parsed_args["config"]:
        import yaml

        config_data_name = None
        config_model_name = None

        for config in parsed_args["config"]:
            config = config[0]
            if not Path(config).exists():
                raise RuntimeError(f"Config file {str(config)} does not exists!")

            with open(config, "r") as stream:
                file = yaml.safe_load(stream)
                t_model_name = file.get("model_class", None)
                t_data_name = file.get("data_class", None)

                config_data_name = t_data_name if t_data_name is not None else config_data_name
                config_model_name = t_model_name if t_model_name is not None else config_model_name

    if config_model_name is not None and parsed_model_name is not None and config_model_name != parsed_model_name:
        warnings.warn(
            "Passed two different model_name in CLI and config file! Make sure that object parameters fit the given class. CLI will overwrite config!"
        )
        # raise RuntimeError("Passed two different model_name in CLI and config file!")

    if config_data_name is not None and parsed_data_name is not None and config_data_name != parsed_data_name:
        warnings.warn(
            "Passed two different data_name in CLI and config file! Make sure that object parameters fit the given class. CLI will overwrite config!"
        )
        # raise RuntimeError("Passed two different data_name in CLI and config file!")

    assert parsed_model_name in pytorch_lightning.utilities.cli.MODEL_REGISTRY.names + [None]
    assert config_model_name in pytorch_lightning.utilities.cli.MODEL_REGISTRY.names + [None]

    assert parsed_data_name in pytorch_lightning.utilities.cli.DATAMODULE_REGISTRY.names + [None]
    assert config_data_name in pytorch_lightning.utilities.cli.DATAMODULE_REGISTRY.names + [None]

    chosen_model = None
    chosen_data = None

    if config_model_name is not None:
        chosen_model = config_model_name
    if parsed_model_name is not None:
        chosen_model = parsed_model_name

    if config_data_name is not None:
        chosen_data = config_data_name
    if parsed_data_name is not None:
        chosen_data = parsed_data_name

    if chosen_model is not None:
        chosen_model = pytorch_lightning.utilities.cli.MODEL_REGISTRY[chosen_model]
    if chosen_data is not None:
        chosen_data = pytorch_lightning.utilities.cli.DATAMODULE_REGISTRY[chosen_data]

    return chosen_model, chosen_data


class SaveConfigCallback(Callback):
    """Saves a LightningCLI config to the log_dir when training starts.

    Args:
        parser: The parser object used to parse the configuration.
        config: The parsed configuration that will be saved.
        config_filename: Filename for the config file.
        overwrite: Whether to overwrite an existing config file.
        multifile: When input is multiple config files, saved config preserves this structure.

    Raises:
        RuntimeError: If the config file already exists in the directory to avoid overwriting a previous run
    """

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str,
        safe_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        multifile: bool = False,
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
        self.multifile = multifile
        self.safe_path = safe_path

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:

        if len(trainer.loggers) == 0:
            warnings.warn(
                "BetterCLI was used, but no logger was passed to the trainer. CLI will not be able to save the config."
            )
            return

        relevant_logger = trainer.loggers[0]

        if isinstance(relevant_logger, TensorBoardLogger):
            log_dir = relevant_logger.log_dir
        else:
            log_dir = relevant_logger.save_dir

        assert log_dir is not None
        config_path = os.path.join(log_dir, self.config_filename)
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    " or set `LightningCLI(save_config_overwrite=True)` to overwrite the config file."
                )

        # save the file on rank 0
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            with open(str(log_dir + "/program_call.txt"), "w") as outfile:
                outfile.write(" ".join(sys.argv) + "\n")


class BetterCli(LightningCLI):
    def __init__(
        self,
        model_class: Optional[Union[Type[LightningModule], Callable[..., LightningModule]]] = None,
        datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
        add_arguments_function: Optional[Callable] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_filename: str = "config.yaml",
        save_config_overwrite: bool = True,
        save_config_multifile: bool = False,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = True,
        description: str = "pytorch-lightning trainer command line tool",
        env_prefix: str = "PL",
        env_parse: bool = False,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        parse: bool = True,
        instantiate: bool = True,
        run: bool = True,
        auto_registry: bool = False,
        auto_class: bool = False,
    ) -> None:
        """Receives as input pytorch-lightning classes (or callables which return pytorch-lightning classes), which
        are called / instantiated using a parsed configuration file and / or command line args.

        Parsing of configuration from environment variables can be enabled by setting ``env_parse=True``.
        A full configuration yaml would be parsed from ``PL_CONFIG`` if set.
        Individual settings are so parsed from variables named for example ``PL_TRAINER__MAX_EPOCHS``.

        For more info, read :ref:`the CLI docs <common/lightning_cli:LightningCLI>`.

        .. warning:: ``LightningCLI`` is in beta and subject to change.

        Args:
            model_class: An optional :class:`~pytorch_lightning.core.lightning.LightningModule` class to train on or a
                callable which returns a :class:`~pytorch_lightning.core.lightning.LightningModule` instance when
                called. If ``None``, you can pass a registered model with ``--model=MyModel``.
            datamodule_class: An optional :class:`~pytorch_lightning.core.datamodule.LightningDataModule` class or a
                callable which returns a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` instance when
                called. If ``None``, you can pass a registered datamodule with ``--data=MyDataModule``.
            add_arguments_function: An optional callable, that takes one single argument, a :class:`pytorch_lightning.utilities.cli.LightningArgumentParser`,
                as input and can modify it. Example applications are linking arguments and adding new arguments.
            save_config_callback: A callback class to save the training config.
            save_config_filename: Filename for the config file.
            save_config_overwrite: Whether to overwrite an existing config file.
            save_config_multifile: When input is multiple config files, saved config preserves this structure.
            trainer_class: An optional subclass of the :class:`~pytorch_lightning.trainer.trainer.Trainer` class or a
                callable which returns a :class:`~pytorch_lightning.trainer.trainer.Trainer` instance when called.
            trainer_defaults: Set to override Trainer defaults or add persistent callbacks. The callbacks added through
                this argument will not be configurable from a configuration file and will always be present for
                this particular CLI. Alternatively, configurable callbacks can be added as explained in
                :ref:`the CLI docs <common/lightning_cli:Configurable callbacks>`.
            seed_everything_default: Value for the :func:`~pytorch_lightning.utilities.seed.seed_everything`
                seed argument. If set to ''True'' a random seed will be chosen.
            description: Description of the tool shown when running ``--help``.
            env_prefix: Prefix for environment variables.
            env_parse: Whether environment variable parsing is enabled.
            parser_kwargs: Additional arguments to instantiate each ``LightningArgumentParser``.
            subclass_mode_model: Whether model can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            subclass_mode_data: Whether datamodule can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            run: Whether subcommands should be added to run a :class:`~pytorch_lightning.trainer.trainer.Trainer`
                method. If set to ``False``, the trainer and model classes will be instantiated only.
            auto_registry: Whether to automatically fill up the registries with all defined subclasses.
            auto_class: Whether to automatically detect the model_class and datamodule_class arguments in the config and CLI if present.
        """
        self.save_config_callback = save_config_callback
        self.save_config_filename = save_config_filename
        self.save_config_overwrite = save_config_overwrite
        self.save_config_multifile = save_config_multifile
        self.trainer_class = trainer_class
        self.trainer_defaults = trainer_defaults or {}
        self.seed_everything_default = seed_everything_default

        _populate_registries(auto_registry)

        if auto_class:
            tmp_model_class, tmp_datamodule_class = _get_chosen_data_model()

            if model_class and tmp_model_class:
                raise RuntimeError(
                    f"Found model_class {tmp_model_class} with auto_registry but you also provided {model_class} as input."
                )

            if datamodule_class and tmp_datamodule_class:
                raise RuntimeError(
                    f"Found data_class {tmp_datamodule_class} with auto_registry but you also provided {datamodule_class} as input."
                )

            model_class = tmp_model_class or model_class
            datamodule_class = tmp_datamodule_class or datamodule_class

        self.model_class = model_class
        # used to differentiate between the original value and the processed value
        self._model_class = model_class or LightningModule
        self.subclass_mode_model = (model_class is None) or subclass_mode_model

        self.datamodule_class = datamodule_class
        # used to differentiate between the original value and the processed value
        self._datamodule_class = datamodule_class or LightningDataModule
        self.subclass_mode_data = (datamodule_class is None) or subclass_mode_data

        self._add_argument_function = add_arguments_function

        main_kwargs, subparser_kwargs = self._setup_parser_kwargs(
            parser_kwargs or {},  # type: ignore  # github.com/python/mypy/issues/6463
            {"description": description, "env_prefix": env_prefix, "default_env": env_parse},
        )
        self.setup_parser(run, main_kwargs, subparser_kwargs)

        if parse:
            self.parse_arguments(self.parser)

            if instantiate:

                self.before_instantiate_classes()
                self.instantiate_classes()

                if run:
                    self.run_trainer()

    def parse_arguments(self, parser: Optional[LightningArgumentParser]) -> None:
        if parser is None:
            parser = self.parser

        super().parse_arguments(parser)

        self.subcommand = self.config.get("subcommand")

    def instantiate_classes(self, trainer=True, model=True, data=True) -> None:
        """Instantiates the classes and sets their attributes. Also sets the seed."""
        self.set_seed()
        config = copy.copy(self.config)

        if not model:
            raise RuntimeError("Does not work properly")
            self.parser.groups.pop("model", None)
            config.pop("model", None)
            config.pop("model_class", None)
            for member in self.parser.required_args:
                if "model." in member:
                    self.parser.required_args.pop(member)
        if not data:
            raise RuntimeError("Does not work properly")
            self.parser.groups.pop("data", None)
            config.pop("data", None)
            config.pop("data_class", None)
            for member in self.parser.required_args.copy():
                if "data." in member:
                    self.parser.required_args.remove(member)
        self.config_init = self.parser.instantiate_classes(config)
        if data:
            self.datamodule = self._get(self.config_init, "data")
        if model:
            self.model = self._get(self.config_init, "model")
        self._add_configure_optimizers_method_to_model(self.subcommand)
        if trainer:
            self.instantiate_trainer()

    def instantiate_trainer(
        self,
        add_callbacks: Union[
            Optional[Type[pytorch_lightning.callbacks.Callback]],
            Optional[Sequence[Type[pytorch_lightning.callbacks.Callback]]],
        ] = None,
        **kwargs: Any,
    ) -> Trainer:
        """Instantiates the trainer.
        Args:
            callbacks: A sequence of custom callbacks that can be added to the trainer
            kwargs: Any custom trainer arguments.
        """
        extra_callbacks = [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        if add_callbacks is not None:
            extra_callbacks.extend(add_callbacks)

        trainer_config = {**self._get(self.config_init, "trainer"), **kwargs}

        self.trainer = self._instantiate_trainer(trainer_config, extra_callbacks)

    def run_trainer(self, **function_kwargs):
        """Runs the provided subcommand"""
        if self.subcommand is not None:
            self._run_subcommand(self.subcommand, **function_kwargs)
        else:
            raise RuntimeError("No subcommand provided.")

    def _run_subcommand(self, subcommand: str, **kwargs) -> None:
        """Run the chosen subcommand."""
        before_fn = getattr(self, f"before_{subcommand}", None)
        if callable(before_fn):
            before_fn()

        default = getattr(self.trainer, subcommand)
        fn = getattr(self, subcommand, default)
        fn_kwargs = self._prepare_subcommand_kwargs(subcommand)
        fn(**fn_kwargs, **kwargs)

        after_fn = getattr(self, f"after_{subcommand}", None)
        if callable(after_fn):
            after_fn()

    def set_seed(self):
        """Sets the seed."""
        seed = self._get(self.config, "seed_everything")
        if seed is not None:
            seed_everything(seed, workers=True)

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds default arguments to the parser."""

        default_seed = self.seed_everything_default

        if isinstance(self.seed_everything_default, bool):
            if self.seed_everything_default:
                default_seed = random.randint(0, int(torch.iinfo(torch.int32).max))
            else:
                default_seed = None

        parser.add_argument(
            "--seed_everything",
            type=Optional[int],
            default=default_seed,
            help="Set to an int to run seed_everything with this value before classes instantiation",
        )

        parser.add_argument("--model_class", type=Optional[str], help="Class of the added ")
        parser.add_argument("--data_class")

    def _add_arguments(self, parser: LightningArgumentParser) -> None:
        # default + core + custom arguments
        self.add_default_arguments_to_parser(parser)
        self.add_core_arguments_to_parser(parser)
        self.add_arguments_to_parser(parser)
        if self._add_argument_function is not None:
            self._add_argument_function(parser)
        # add default optimizer args if necessary
        if not parser._optimizers:  # already added by the user in `add_arguments_to_parser`
            parser.add_optimizer_args(OPTIMIZER_REGISTRY.classes)
        if not parser._lr_schedulers:  # already added by the user in `add_arguments_to_parser`
            parser.add_lr_scheduler_args(LR_SCHEDULER_REGISTRY.classes)
        self.link_optimizers_and_lr_schedulers(parser)
