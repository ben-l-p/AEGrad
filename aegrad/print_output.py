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

VERBOSITY_LEVEL = VerbosityLevel.NORMAL

def set_verbosity(level: VerbosityLevel) -> None:
    global VERBOSITY_LEVEL
    VERBOSITY_LEVEL = level


def make_color(text: str, color: Colour) -> str:
    # info on coloring from https://vascosim.medium.com/how-to-print-colored-text-in-python-52f6244e2e30
    return f'\033[{color.value}m{text}\033[0m'

def warn(message: str) -> None:
    if VERBOSITY_LEVEL.value >= VerbosityLevel.WARNING.value:
        print(make_color(f"Warning: {message}", Colour.YELLOW))

def jax_print(message: str, verbose_level = VERBOSITY_LEVEL.VERBOSE, **kwargs) -> None:
    if VERBOSITY_LEVEL.value >= verbose_level.value:
        jax.debug.print(message, **kwargs)

def print_with_time(init_message: str, final_message: str, verbose_level = VerbosityLevel.NORMAL):
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if VERBOSITY_LEVEL.value >= verbose_level.value:
                print(make_color(init_message, Colour.BLUE))
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                print(make_color(final_message.format(end_time - start_time), Colour.GREEN))
            else:
                result = func(*args, **kwargs)
            return result
        return inner
    return decorator