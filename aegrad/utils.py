from typing import Any, Sequence
from typing import Callable, Protocol, TypeVar
from jax import tree_util
from dataclasses import fields, is_dataclass
from functools import wraps

def replace_self(func: Callable[..., object]) -> Callable[..., None]:
    # the beauty and the pain behind this codebase
    @wraps(func)
    def wrapper(*args, **kwargs) -> None:
        args[0].__dict__.update(
            func(*args, **kwargs).__dict__
        )  # self is always the first argument
    return wrapper


class SupportsPytree(Protocol):
    def _dynamic_names(self) -> Sequence[str]: ...
    def _static_names(self) -> Sequence[str]: ...


T = TypeVar("T", bound=SupportsPytree)

def make_pytree(cls: type[T]) -> type[T]:
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


def check_type(obj: Any, type_: type | Sequence[type]) -> None:
    try:
        len(type_)
    except TypeError:
        type_ = (type_,)

    for t in type_:
        if isinstance(obj, t):
            return

    raise TypeError(f"Expected {type_}, but got {obj}.")


def shallow_asdict(obj):
    if not is_dataclass(obj):
        raise TypeError("object must be a dataclass")
    return {f.name: getattr(obj, f.name) for f in fields(obj)}