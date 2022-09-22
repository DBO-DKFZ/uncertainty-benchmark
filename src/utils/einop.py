import functools
from typing import TypeVar, Union, Callable

import einops
from einops.parsing import EinopsError, ParsedExpression
import torch

A = TypeVar("A")


@functools.lru_cache(256)
def _match_einop(pattern: str, reduction=None, **axes_lengths: int):
    """Find the corresponding operation matching the pattern"""

    if not "->" in pattern:
        parse_test = ParsedExpression("pattern")
        return "check"

    left, rght = pattern.split("->")
    left = ParsedExpression(left)
    rght = ParsedExpression(rght)

    default_op = "rearrange"
    op = default_op

    for index in left.identifiers:
        if index not in rght.identifiers:
            op = "reduce"
            break

    for index in rght.identifiers:
        if index not in left.identifiers:
            if op != default_op:
                raise EinopsError("You must perform a reduce and repeat separately: {}".format(pattern))
            op = "repeat"
            break

    return op


def einop(tensor: A, pattern: str, reduction=None, **axes_lengths: int) -> A:
    """Perform either reduce, rearrange, or repeat depending on pattern"""
    op = _match_einop(pattern, reduction, **axes_lengths)

    if op == "check":
        return eincheck(tensor, pattern, **axes_lengths)

    if op == "rearrange":
        if reduction is not None:
            raise EinopsError(
                'Got reduction operation but there is no dimension to reduce in pattern: "{}"'.format(pattern)
            )
        return einops.rearrange(tensor, pattern, **axes_lengths)
    elif op == "reduce":
        if reduction is None:
            raise EinopsError("Missing reduction operation for reduce pattern: {}".format(pattern))
        return einops.reduce(tensor, pattern, reduction, **axes_lengths)
    elif op == "repeat":
        if reduction is not None:
            raise EinopsError("Do not pass reduction for repeat pattern: {}".format(pattern))
        return einops.repeat(tensor, pattern, **axes_lengths)
    else:
        raise ValueError(f"Unknown operation: {op}")


def eincheck(tensor: A, pattern: str, **axes_lengths: int):
    """Returns false if the pattern does not match"""

    if "->" in pattern:
        raise EinopsError("Got -> Operator in pattern, however you should only provide the left-hand side.")

    new_pattern = pattern + " -> " + pattern

    try:
        einops.rearrange(tensor, new_pattern, **axes_lengths)
        return True
    except EinopsError as e:
        return False


########### Change the torch libary.


def _instance_einop(self, pattern: str, reduction: Union[Callable, str] = None, **axes_lengths: int) -> A:
    return einop(self, pattern, reduction, **axes_lengths)


def _instance_rearrange(self, pattern: str, **axes_lengths: int) -> A:
    return einops.rearrange(self, pattern, **axes_lengths)


def _instance_repeat(self, pattern: str, **axes_lengths: int) -> A:
    return einops.repeat(self, pattern, **axes_lengths)


def _instance_reduce(self, pattern: str, reduction: Union[Callable, str] = None, **axes_lengths: int) -> A:
    return einops.reduce(self, pattern, reduction, **axes_lengths)


def _instance_eincheck(self, pattern: str, **axes_lengths: int) -> bool:
    return eincheck(self, pattern, **axes_lengths)


torch.Tensor.einop = _instance_einop
torch.Tensor.einrange = _instance_rearrange
torch.Tensor.einreduce = _instance_rearrange
torch.Tensor.einrepeat = _instance_rearrange
torch.Tensor.eincheck = _instance_eincheck


########### Make module callable. Requires some python hacks
########### Now you can do from utils import einop and not from utils.einop import einop
# import sys

# class EinopModule(sys.modules[__name__].__class__):
#    def __call__(self, tensor: A, pattern: str, reduction=None, **axes_lengths: int) -> A:
#        return einop(tensor, pattern, reduction, **axes_lengths)

# sys.modules[__name__].__class__ = EinopModule

##########
