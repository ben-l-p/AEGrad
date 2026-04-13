import pytest
from print_utils import set_verbosity, VerbosityLevel


@pytest.fixture(autouse=True, scope="session")
def silence_output():
    set_verbosity(VerbosityLevel.WARNING)
