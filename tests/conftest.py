import pytest
import jax
from aegrad.utils.print_utils import set_verbosity, VerbosityLevel

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True, scope="session")
def silence_output():
    set_verbosity(VerbosityLevel.WARNING)
