from typing import Any, Sequence
from typing import Callable, Protocol, TypeVar
from jax import tree_util
from dataclasses import fields, is_dataclass

def replace_self(func: Callable[..., object]) -> Callable[..., None]:
    # the beauty and the pain behind this codebase
    def wrapper(*args, **kwargs) -> None:
        args[0].__dict__.update(
            func(*args, **kwargs).__dict__
        )  # self is always the first argument

    return wrapper


class SupportsPytree(Protocol):
    def flatten_func(self) -> tuple[tuple[Any], tuple[Any]]: ...

    @classmethod
    def unflatten_func(cls, aux_data: tuple[Any], children: tuple[Any]) -> object: ...


T = TypeVar("T", bound=SupportsPytree)


def make_pytree(cls: type[T]) -> type[T]:
    """
    Convert an object to a pytree structure.
    :param cls: Class to be converted to a pytree.
    """
    tree_util.register_pytree_node(cls, cls.flatten_func, cls.unflatten_func)
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