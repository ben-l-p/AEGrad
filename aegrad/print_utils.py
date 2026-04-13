from __future__ import annotations
import time
from functools import wraps
from enum import Enum
import jax


class Colour(Enum):
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36


class VerbosityLevel(Enum):
    SILENT = 0
    WARNING = 1
    NORMAL = 2
    VERBOSE = 3


VERBOSITY_LEVEL = VerbosityLevel.NORMAL  # default value


def set_verbosity(level: VerbosityLevel) -> None:
    global VERBOSITY_LEVEL
    VERBOSITY_LEVEL = level


def make_color(text: str, color: Colour) -> str:
    # info on colouring from https://vascosim.medium.com/how-to-print-colored-text-in-python-52f6244e2e30
    return f"\033[{color.value}m{text}\033[0m"


def warn(message: str, **kwargs) -> None:
    if VERBOSITY_LEVEL.value >= VerbosityLevel.WARNING.value:
        jax_print(
            make_color(f"Warning: {message}", color=Colour.YELLOW),
            verbose_level=VerbosityLevel.WARNING,
            **kwargs,
        )


def warn_if_32_bit() -> None:
    if not jax.config.read("jax_enable_x64"):
        warn(
            'Running with 32-bit floating point precision. Using 64-bit with "jax.config.update("jax_enable_x64", '
            'True)" is recommended'
        )


def jax_print(
        message: str, verbose_level: VerbosityLevel = VERBOSITY_LEVEL.VERBOSE, **kwargs
) -> None:
    if VERBOSITY_LEVEL.value >= verbose_level.value:
        jax.debug.print(message, **kwargs)
