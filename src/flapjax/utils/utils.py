from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import fields, is_dataclass
from typing import Any, Protocol, TypeVar

import jax
from jax import Array, tree_util
from jax import numpy as jnp

from flapjax.utils.print_utils import jax_print


class SupportsPytree(Protocol):
    def _dynamic_names(self) -> Sequence[str]: ...

    def _static_names(self) -> Sequence[str]: ...


T = TypeVar("T", bound=SupportsPytree)


def make_pytree[T: SupportsPytree](cls: type[T]) -> type[T]:
    """
    Convert an object to a pytree structure.
    :param cls: Class to be converted to a pytree.
    """

    def flatten_func(self: T) -> tuple[tuple[Any], tuple[Any]]:
        children = tuple(getattr(self, field) for field in self._dynamic_names())
        aux_data = tuple(getattr(self, field) for field in self._static_names())
        return children, aux_data

    def unflatten_func(aux_data: tuple[Any], children: tuple[Any]) -> T:
        obj = cls.__new__(cls)  # Create an uninitialized instance
        for field_name, value in zip(cls._dynamic_names(), children):
            setattr(obj, field_name, value)
        for field_name, value in zip(cls._static_names(), aux_data):
            setattr(obj, field_name, value)
        return obj

    tree_util.register_pytree_node(cls, flatten_func, unflatten_func)
    return cls


def check_type(obj: Any, type_: type) -> None:
    if not isinstance(obj, type_):
        raise TypeError(f"Expected {type_}, but got {obj}.")


def shallow_as_dict(obj):
    if not is_dataclass(obj):
        raise TypeError("object must be a dataclass")
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def index_to_arr(
    index: int | Array | Sequence[int] | slice | None, n_entries: int
) -> Array:
    r"""
    Convert an input index to an Array index.
    :param index: Index to apply to a sequence
    :param n_entries: Number of entries in the full un-indexed sequence
    :return: Array index corresponding to the input index
    """
    if isinstance(index, slice):
        return jnp.arange(n_entries)[index]
    elif isinstance(index, Sequence):
        return jnp.array(index)
    elif isinstance(index, Array):
        return index
    elif isinstance(index, int):
        return jnp.array([index])
    elif index is None:
        return jnp.arange(n_entries)
    else:
        raise TypeError("index must be a slices, sequence of ints, or Array")


def nested_list_to_tuple(a: list[Any]) -> tuple[Any, ...]:
    r"""
    Convert nested lists to nested tuples.
    """

    def inner_func(a_):
        try:
            return tuple(nested_list_to_tuple(i) for i in a_)
        except TypeError:
            return a_

    return inner_func(a)


def conditional_profile[U](
    func: Callable[..., U],
    n_loops: int | None,
    func_name: str,
    arg_name: str,
) -> Callable[..., tuple[U, float | None, float | None]]:
    r"""
    Function wrapper which optionally profiles it.
    :param func: Function to be profiled.
    :param n_loops: Number of times to run the function, where the average time will be taken. If None, no profiling is
    done.
    :param func_name: Name of the function to be profiled, used for console printing.
    :param arg_name: Name of the argument to be profiled, used for console printing.
    :return: New function which returns the same value as `func`, alongside the compile time and average run time of
    `func`, which are substituted for None when no profiling is done.
    """

    if n_loops is None:
        return lambda *args, **kwargs: (func(*args, **kwargs), None, None)
    else:
        if not isinstance(n_loops, int) or n_loops <= 0:
            raise ValueError("n_loops must be a positive int")
        loops: int = n_loops

        def inner_func(*args_: Any, **a_: Any) -> tuple[U, float, float]:
            # warm-up: includes compile + first run
            start_time = time.time()
            out = jax.block_until_ready(func(*args_, **a_))
            compile_run_time = time.time() - start_time

            # timed run on the compiled function
            start_time = time.time()
            for _ in range(loops):
                jax.block_until_ready(func(*args_, **a_))

            run_time = (time.time() - start_time) / loops
            compile_time = compile_run_time - run_time

            jax_print(
                f"| Function: {func_name:<14} Argument: {arg_name:<16} Compile time: {compile_time:<8.4f} Run time: {run_time:<8.4f} |",
                verbose_level="normal",
            )

            return out, compile_time, run_time

        return inner_func
