import pytest

from .load_samples import *
import numpy as np
import bct


SEED = 1


@pytest.mark.xfail(reason="unfixed bug #68")
def test_null_model_und_sign():
    # Regression test for bug fixed in b02a306
    x = load_sample(thres=.4)

    bct.null_model_und_sign(x)


@pytest.mark.xfail(reason="unfixed bug #68")
def test_null_model_dir_sign():
    # Regression test for counterpart to the undirected bug
    x = load_directed_sample(thres=.4)
    bct.null_model_dir_sign(x)


def test_randmio_und_seed():
    x = load_sample(thres=0.4)
    swaps = 5
    ref, _ = bct.randmio_und(x, swaps, seed=SEED)
    test_same, _ = bct.randmio_und(x, swaps, seed=SEED)
    test_diff, _ = bct.randmio_und(x, swaps, seed=SEED*2)

    assert np.allclose(ref, test_same)
    assert not np.allclose(ref, test_diff)
