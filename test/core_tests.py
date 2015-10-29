from load_samples import *
import bct
import numpy as np


def test_threshold_proportional():
    x = load_sample()
    x = bct.threshold_proportional(x, .5, copy=True)
    assert np.allclose(np.sum(x), 22548.51206965)


def test_threshold_proportional_nocopy():
    x = load_sample()
    bct.threshold_proportional(x, .3, copy=False)
    assert np.allclose(np.sum(x), 15253.75425406)


def test_threshold_proportional_directed():
    x = load_directed_sample()
    bct.threshold_proportional(x, .28, copy=False)
    assert np.sum(x) == 3410
    # assert np.allclose( np.sum(x), 32852.72485433 )


def test_threshold_absolute():
    x = load_sample()
    x = bct.threshold_absolute(x, 2.1)
    assert np.allclose(np.sum(x), 13280.17768104)


def test_strengths_und():
    x = load_sample()
    s = bct.strengths_und(x)
    assert np.allclose(np.sum(x), 38967.38702018)


def test_degrees_und():
    x = load_sample()
    s = bct.degrees_und(bct.threshold_proportional(x, .26))
    assert np.sum(s) == 4916


def test_binarize():
    x = load_sample()
    s = bct.binarize(bct.threshold_proportional(x, .41))
    assert np.sum(s) == 7752


def test_normalize():
    x = load_sample()
    s = bct.normalize(bct.threshold_proportional(x, .79))
    assert np.allclose(np.sum(s), 3327.96285964)


def test_invert():
    x = load_sample()
    s = bct.invert(bct.threshold_proportional(x, .13))
    assert np.allclose(np.sum(s), 790.43107587)
