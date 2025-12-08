import time
from functools import wraps
from enum import Enum
from warnings import warn

class Colour(Enum):
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36


def make_color(text: str, color: Colour) -> str:
    # info on coloring from https://vascosim.medium.com/how-to-print-colored-text-in-python-52f6244e2e30
    return f'\033[{color.value}m{text}\033[0m'

def warn(message: str) -> None:
    print(make_color(f"Warning: {message}", Colour.YELLOW))

def print_with_time(init_message: str, final_message: str):
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            print(make_color(init_message, Colour.BLUE))
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(make_color(final_message.format(end_time - start_time), Colour.GREEN))
            return result
        return inner
    return decorator