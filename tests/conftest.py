import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

import pytest
import jax
from aegrad.utils.print_utils import set_verbosity

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True, scope="session")
def silence_output():
    set_verbosity("warning")
