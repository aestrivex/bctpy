import pytest


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
