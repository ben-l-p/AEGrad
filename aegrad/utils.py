from __future__ import annotations

from typing import Any, Sequence
from typing import Protocol, TypeVar
from dataclasses import fields, is_dataclass
from jax import tree_util


class SupportsPytree(Protocol):
    def _dynamic_names(self) -> Sequence[str]: ...

    def _static_names(self) -> Sequence[str]: ...


T = TypeVar("T", bound=SupportsPytree)


def _make_pytree(cls: type[T]) -> type[T]:
    """
    Convert an object to a pytree structure_dv.
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


def _check_type(obj: Any, type_: type) -> None:
    if not isinstance(obj, type_):
        raise TypeError(f"Expected {type_}, but got {obj}.")
    return


def _shallow_asdict(obj):
    if not is_dataclass(obj):
        raise TypeError("object must be a dataclass")
    return {f.name: getattr(obj, f.name) for f in fields(obj)}
