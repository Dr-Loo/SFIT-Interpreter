import numpy as np
import pytest

@pytest.fixture(scope="session", autouse=True)
def _seed_and_printopts():
    np.random.seed(7)
    np.set_printoptions(precision=6, suppress=True)
    yield
