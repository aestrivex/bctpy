from .load_samples import load_sample
import numpy as np
import bct


def test_glob_eff():
    x = load_sample(thres=.4)
    geff = bct.efficiency_wei(x)
    print(geff, 1.8784)
    assert np.allclose(geff, 1.8784, atol=1e-4)


def test_loc_eff():
    x = load_sample(thres=.4)
    leff_o = bct.efficiency_wei(x, local='original')
    print(np.sum(leff_o), 315.6225)
    assert np.allclose(np.sum(leff_o), 315.6225, atol=0.1)

    leff = bct.efficiency_wei(x, local=True)
    print(np.sum(leff), 268.5581)
    assert np.allclose(np.sum(leff), 268.5581, atol=0.1)


def test_glob_eff_bin():
    x = load_sample(thres=.4)
    geff = bct.efficiency_bin(x)

    y = bct.binarize(x)
    geff2 = bct.efficiency_bin(y)

    print(geff, geff2, 0.6999)
    assert np.allclose(geff, 0.6999, atol=1e-4)
    assert np.allclose(geff2, 0.6999, atol=1e-4)


def test_loc_eff_bin():
    x = load_sample(thres=.4)
    leff = bct.efficiency_bin(x, local=True)

    y = bct.binarize(x)
    leff2 = bct.efficiency_bin(y, local=True)

    print(np.sum(leff), np.sum(leff2), 105.5111)
    assert np.allclose(np.sum(leff), 105.5111, atol=0.1)
    assert np.allclose(np.sum(leff2), 105.5111, atol=0.1)
