from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import Literal

import jax


class Colour(Enum):
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36


type VerbosityLevel = Literal["silent", "warning", "normal", "verbose"]

_VERBOSITY_RANK: dict[VerbosityLevel, int] = {
    "silent": 0,
    "warning": 1,
    "normal": 2,
    "verbose": 3,
}


def map_verbosity_level(level: VerbosityLevel) -> int:
    try:
        return _VERBOSITY_RANK[level]
    except KeyError:
        raise ValueError(f"Invalid verbosity level: {level}") from None


VERBOSITY_LEVEL: VerbosityLevel = "normal"  # default value


def set_verbosity(level: VerbosityLevel) -> None:
    global VERBOSITY_LEVEL
    VERBOSITY_LEVEL = level


@contextmanager
def verbosity(level: VerbosityLevel):
    r"""
    Context manager to temporarily change the verbosity level.
    :param level: Custom verbosity to use in context.
    """
    old = VERBOSITY_LEVEL
    set_verbosity(level)
    try:
        yield
    finally:
        set_verbosity(old)


def make_color(text: str, color: Colour) -> str:
    # info on colouring from https://vascosim.medium.com/how-to-print-colored-text-in-python-52f6244e2e30
    return f"\033[{color.value}m{text}\033[0m"


def warn(message: str, **kwargs) -> None:
    if map_verbosity_level(VERBOSITY_LEVEL) >= map_verbosity_level("warning"):
        jax_print(
            make_color(f"Warning: {message}", color=Colour.YELLOW),
            verbose_level="warning",
            **kwargs,
        )


def warn_if_32_bit() -> None:
    if not jax.config.read("jax_enable_x64"):
        warn(
            'Running with 32-bit floating point precision. Using 64-bit with "jax.config.update("jax_enable_x64", '
            'True)" is recommended'
        )


def jax_print(
    message: str, verbose_level: VerbosityLevel = "verbose", **kwargs
) -> None:
    if map_verbosity_level(VERBOSITY_LEVEL) >= map_verbosity_level(verbose_level):
        jax.debug.print(message, **kwargs)


def print_table_line(
    inner_width: int, verbose_level: VerbosityLevel = "normal"
) -> None:
    jax_print("+" + "-" * inner_width + "+", verbose_level=verbose_level)


def print_table_title(
    title: str, inner_width: int, verbose_level: VerbosityLevel = "normal"
) -> None:
    usable_width = inner_width - len(title) - 2
    left_length = usable_width // 2
    right_length = usable_width - left_length
    jax_print(
        "\n+" + "-" * left_length + f" {title} " + "-" * right_length + "+",
        verbose_level=verbose_level,
    )
