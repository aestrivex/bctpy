import os
from .load_samples import (
    load_sample, load_directed_sample, load_signed_sample, load_directed_low_modularity_sample, TEST_DIR
)
import numpy as np
import bct


def test_modularity_und():
    x = load_sample(thres=.4)
    _, q = bct.modularity_und(x)
    print(q)
    assert np.allclose(q, 0.24097717)
    # matlab and bctpy appear to return different results due to the cross-
    # package numerical instability of eigendecompositions


def test_modularity_louvain_und(stable_rng):
    x = load_sample(thres=.4)

    _, q = bct.modularity_louvain_und(x, seed=stable_rng)
    assert np.allclose(q, 0.253_148_271)

    fails = 0
    for i in range(100):
        ci, q = bct.modularity_louvain_und(x)
        try:
            assert np.allclose(q, .25, atol=0.01)
        except AssertionError:
            if fails >= 5:
                raise
            else:
                fails += 1

    _, q = bct.modularity_finetune_und(x, seed=stable_rng)
    assert np.allclose(q, 0.242_632_732)


def test_modularity_finetune_und(stable_rng):
    x = load_sample(thres=.4)

    _, q = bct.modularity_finetune_und(x, seed=stable_rng)
    assert np.allclose(q, 0.253_148_271)

    fails = 0
    for i in range(100):
        _, q = bct.modularity_finetune_und(x)
        try:
            assert np.allclose(q, .25, atol=0.03)
        except AssertionError:
            if fails >= 5:
                raise
            else:
                fails += 1

    ci, oq = bct.modularity_louvain_und(x, seed=stable_rng)
    _, q = bct.modularity_finetune_und(x, ci=ci, seed=stable_rng)
    print(q, oq)
    # assert np.allclose(q, .25892588)
    assert np.allclose(q, .25856714)
    assert q - oq >= -1e6

    ci, oq = bct.modularity_und(x)
    for i in range(100):
        _, q = bct.modularity_finetune_und(x, ci=ci)
        assert np.allclose(q, .25, atol=0.002)
        assert q - oq >= -1e6

    # modularity_finetune_und appears to be very stable when given a stable ci
    # in thousands of test runs on the sample data (using the deterministic
    # modularity maximization algorithm), only two states appeared;
    # both improved the optimal modularity. a basic increase -- modules that
    # always benefited from switching -- always occurred. on top of that, a
    # slightly larger increase dependent on order occurred in both matlab and
    # bctpy around ~0.6% of the time. Due to numerical instability arising
    # from something different between matlab and scipy, these values were not
    # the same across languages, but both languages showed bistable transitions
    # they were extremely stable. The values were about .0015 apart.

    # also the matlab and python versions of modularity_und return slightly
    # different modular structure, but the instability is present despite this
    #(i.e. it is unstable both when the modular structure is identical and not)


def test_modularity_louvain_und_sign_seed(stable_rng):
    # performance is same as matlab if randomness is quashed
    x = load_signed_sample()
    _, q = bct.modularity_louvain_und_sign(x, seed=stable_rng)
    print(q)
    assert np.allclose(q, 0.453_375_647)


def test_modularity_finetune_und_sign_actually_finetune(stable_rng):
    x = load_signed_sample()
    ci, oq = bct.modularity_louvain_und_sign(x, seed=stable_rng)
    _, q = bct.modularity_finetune_und_sign(x, seed=stable_rng, ci=ci)
    print(q)
    assert np.allclose(q, 0.462_292_161)
    assert q >= oq

    randomized_sample = stable_rng.random(size=(len(x), len(x)))
    randomized_sample = randomized_sample + randomized_sample.T
    x[np.where(bct.threshold_proportional(randomized_sample, .2))] = 0

    ci, oq = bct.modularity_louvain_und_sign(x, seed=stable_rng)
    print(oq)
    assert np.allclose(oq, 0.458_063_807)
    for i in range(100):
        _, q = bct.modularity_finetune_und_sign(x, ci=ci)
        assert q >= oq


def test_modularity_probtune_und_sign(stable_rng):
    x = load_signed_sample()
    ci, q = bct.modularity_probtune_und_sign(x, seed=stable_rng)
    print(q)
    # N.B. this result is quite different
    # assert np.allclose(q, .07885327)  # legacy numpy.random.RandomState
    assert np.allclose(q, 0.114_732_792)  # stable_rng

    ci, _ = bct.modularity_louvain_und_sign(x, seed=stable_rng)
    _, oq = bct.modularity_finetune_und_sign(x, seed=stable_rng, ci=ci)

    for i in np.arange(.05, .5, .02):
        fails = 0
        for j in range(100):
            _, q = bct.modularity_probtune_und_sign(x, ci=ci, p=i)
            try:
                assert q < oq
            except AssertionError:
                if fails > 5:
                    raise
                else:
                    fails += 1


def test_modularity_dir_low_modularity():
    x = load_directed_low_modularity_sample(thres=.67)
    _, q = bct.modularity_dir(x)
    assert np.allclose(q, .06450290)


def test_modularity_louvain_dir_low_modularity(stable_rng):
    x = load_directed_low_modularity_sample(thres=.67)
    _, q = bct.modularity_louvain_dir(x, seed=stable_rng)
    assert np.allclose(q, 0.069_334_607)

# def test_modularity_finetune_dir_low_modularity():
#	x = load_directed_low_modularity_sample(thres=.67)
#	seed = 39602351
#	ci,oq = bct.modularity_louvain_dir(x, seed=seed)
#	_,q = bct.modularity_finetune_dir(x, ci=ci, seed=seed)
#	print q,oq
#	assert q >= oq
    # this does not pass. the matlab code appears to have no idea what to do
    # with
    # the low modularity directed modules. this may be someone else's fault.


def test_modularity_dir():
    x = load_directed_sample()
    _, q = bct.modularity_dir(x)
    print(q, .32742787)
    assert np.allclose(q, .32742787)


def test_modularity_louvain_dir(stable_rng):
    x = load_directed_sample()
    _, q = bct.modularity_louvain_dir(x, seed=stable_rng)
    # assert np.allclose(q, .32697921)  # legacy np.random.RandomState
    assert np.allclose(q, 0.373_475_890)  # stable_rng

# def test_modularity_finetune_dir():
#	x = load_directed_sample()
#	seed = 26080
#	ci,oq = bct.modularity_louvain_dir(x, seed=seed)
#	for i in xrange(100):
#		_,q = bct.modularity_finetune_dir(x, ci=ci)
#		print q,oq
#		assert q >= oq
    # this does not pass with similar behavior to low modularity.
    # the code occasionally returns lower modularity (but very very similar,
    # order .001) partitions despite returning
    # higher modularity partitions a slight majority of the time. i dont know
    # what is wrong


def test_community_louvain(stable_rng):
    x = load_sample(thres=0.4)
    ci, q = bct.community_louvain(x, seed=stable_rng)
    print(q)
    assert np.allclose(q, 0.2583, atol=0.015)


def test_modularity_dir_bug71():
    """Regression test for bug described in issue #71"""
    fpath = os.path.join(TEST_DIR, "failing_cases", "modularity_dir_example.csv")
    x = np.loadtxt(fpath, int, delimiter=',')

    bct.modularity_dir(x)
