import pytest
import numpy as np


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "long: mark test as long-running; skipped with --skiplong option"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--skiplong", action="store_true", default=False, help="skip long-running tests"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--skiplong'):
        return
    skip = pytest.mark.skip(reason="skipping long-running tests")
    for item in items:
        if "long" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def stable_rng():
    """
    To make use of algorithmic improvements in numpy's random number generators,
    bctpy uses numpy.random.default_rng() when creating RNGs.
    As such, results are not guaranteed to be deterministic between numpy versions.
    Therefore, for testing purposes we explicitly set the BitGenerator class.
    """
    seed = 1991
    return np.random.Generator(np.random.PCG64(seed))
