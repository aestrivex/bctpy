from load_samples import *
import numpy as np
import bct


def test_breadthdist():
    x = load_sample(thres=.02)
    r, d = bct.breadthdist(x)
    d[np.where(np.isinf(d))] = 0
    print(np.sum(r), np.sum(d))
    assert np.sum(r) == 5804
    assert np.sum(d) == 30762


def test_reachdist():
    x = load_sample(thres=.02)
    r, d = bct.reachdist(x)
    d[np.where(np.isinf(d))] = 0
    print(np.sum(r), np.sum(d))
    assert np.sum(r) == 5804
    assert np.sum(d) == 30762

    bx = bct.binarize(x, copy=False)
    br, bd = bct.reachdist(bx)
    bd[np.where(np.isinf(bd))] = 0
    print(np.sum(br), np.sum(bd))
    assert np.sum(br) == 5804
    assert np.sum(bd) == 30762


def test_distance_bin():
    x = bct.binarize(load_sample(thres=.02), copy=False)
    d = bct.distance_bin(x)
    d[np.where(np.isinf(d))] = 0
    print(np.sum(d))
    assert np.sum(d) == 30506  # deals with diagonals differently


def test_distance_wei():
    x = load_sample(thres=.02)
    d, e = bct.distance_wei(x)
    d[np.where(np.isinf(d))] = 0
    print(np.sum(d), np.sum(e))

    assert np.allclose(np.sum(d), 155650.1, atol=.01)
    assert np.sum(e) == 30570

def test_charpath():
    x = load_sample(thres=.02)
    d, e = bct.distance_wei(x)
    l, eff,ecc,radius,diameter = bct.charpath(d)

    assert np.any(np.isinf(d))
    assert not np.isnan(radius)
    assert not np.isnan(diameter)


