from jsonargparse import ArgumentParser

import inspect

import argparse
import glob
import inspect
import logging
import os
import re
import sys
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
from unittest.mock import patch

from jsonargparse.formatters import (
    DefaultHelpFormatter,
    empty_help,
    formatter_context,
    get_env_var,
)
from jsonargparse.jsonnet import ActionJsonnet
from jsonargparse.jsonschema import ActionJsonSchema
from jsonargparse.loaders_dumpers import (
    check_valid_dump_format,
    dump_using_format,
    get_loader_exceptions,
    loaders,
    load_value,
    load_value_context,
    yaml_load,
)
from jsonargparse.namespace import (
    is_meta_key,
    Namespace,
    split_key,
    split_key_leaf,
    strip_meta,
)
from jsonargparse.signatures import is_pure_dataclass, SignatureArguments
from jsonargparse.typehints import ActionTypeHint, LazyInitBaseClass
from jsonargparse.actions import (
    ActionParser,
    ActionConfigFile,
    _ActionSubCommands,
    _ActionPrintConfig,
    _ActionConfigLoad,
    _ActionLink,
    _is_branch_key,
    _find_action,
    _find_action_and_subcommand,
    _find_parent_action,
    _find_parent_action_and_subcommand,
    _is_action_value_list,
    filter_default_actions,
    parent_parsers,
)
from jsonargparse.optionals import (
    argcomplete_support,
    fsspec_support,
    omegaconf_support,
    get_config_read_mode,
    import_jsonnet,
    import_argcomplete,
    import_fsspec,
)
from jsonargparse.util import (
    identity,
    ParserError,
    usage_and_exit_error_handler,
    change_to_path_dir,
    Path,
    LoggerProperty,
    _lenient_check_context,
    lenient_check,
)


ArgumentParser = ArgumentParser


def parse_known_args(self, args=None, namespace=None):
    caller = inspect.getmodule(inspect.stack()[1][0]).__package__
    if args is None:
        args = sys.argv[1:]
    else:
        args = list(args)
        if not all(isinstance(a, str) for a in args):
            self.error(f"All arguments are expected to be strings: {args}")

    if namespace is None:
        namespace = Namespace()

    if caller == "argcomplete":
        namespace.__class__ = Namespace
        namespace = self.merge_config(self.get_defaults(skip_check=True), namespace).as_flat()

    try:
        with patch("argparse.Namespace", Namespace), _lenient_check_context(
            caller
        ), ActionTypeHint.subclass_arg_context(self), load_value_context(self.parser_mode):
            namespace, args = self._parse_known_args(args, namespace)
    except (argparse.ArgumentError, ParserError) as ex:
        self.error(str(ex), ex)

    return namespace, args


ArgumentParser.parse_known_args = parse_known_args
