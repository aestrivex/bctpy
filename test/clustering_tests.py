from load_samples import *
import numpy as np
import bct


def test_cluscoef_wu():
    x = load_sample(thres=.23)
    cc = bct.clustering_coef_wu(x)
    print(np.sum(cc), 187.95878414)
    assert np.allclose(np.sum(cc), 187.95878414)


def test_transitivity_wu():
    x = load_sample(thres=.23)
    t = bct.transitivity_wu(x)
    print(t, 1.32927829)
    assert np.allclose(t, 1.32927829)

# test signed clustering so that the cuberoot functionality is tested
# there is no equivalent matlab functionality


def test_cluscoef_signed():
    x = load_signed_sample(thres=.85)
    cc = bct.clustering_coef_wu(x)
    print(np.imag(np.sum(cc)), 0)
    assert np.imag(np.sum(cc)) == 0


def test_transitivity_signed():
    x = load_signed_sample(thres=.85)
    t = bct.transitivity_wu(x)
    print(np.imag(t), 0)
    assert np.imag(t) == 0

# test functions dealing with components on very sparse dataset


def test_component():
    x = load_sparse_sample()
    c1, cs1 = bct.get_components(x)

    print(np.max(c1), 19)
    assert np.max(c1) == 19

    print(np.max(cs1), 72)
    assert np.max(cs1) == 72


def test_consensus():
    x = load_sample(thres=.38)
    ci = bct.consensus_und(x, .1, reps=50)
    print(np.max(ci), 4)
    assert np.max(ci) == 4
    _, q = bct.modularity_und(x, kci=ci)
    print(q, 0.27)
    assert np.allclose(q, 0.27, atol=.01)


def test_cluscoef_wd():
    x = load_directed_low_modularity_sample(thres=.45)
    cc = bct.clustering_coef_wd(x)
    print(np.sum(cc), 289.30817909)
    assert np.allclose(np.sum(cc), 289.30817909)


def test_transitivity_wd():
    x = load_directed_low_modularity_sample(thres=.45)
    t = bct.transitivity_wd(x)
    print(t, 1.30727748)
    assert np.allclose(t, 1.30727748)


def test_cluscoef_bu():
    x = bct.binarize(load_sample(thres=.17), copy=False)
    cc = bct.clustering_coef_bu(x)
    print(np.sum(cc), 60.1016)
    assert np.allclose(np.sum(cc), 60.10160458)


def test_transitivity_bu():
    x = bct.binarize(load_sample(thres=.17), copy=False)
    t = bct.transitivity_bu(x)
    print(t, 0.42763)
    assert np.allclose(t, 0.42763107)


def test_cluscoef_bd():
    x = load_binary_directed_low_modularity_sample(thres=.41)
    cc = bct.clustering_coef_bd(x)
    print(np.sum(cc), 113.31145)
    assert np.allclose(np.sum(cc), 113.31145155)


def test_transitivity_bd():
    x = load_binary_directed_low_modularity_sample(thres=.41)
    t = bct.transitivity_bd(x)
    print(t, 0.50919)
    assert np.allclose(t, 0.50919493)


def test_agreement_weighted():
    # this function is very hard to use or interpret results from
    pass
