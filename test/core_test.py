from .load_samples import load_sample
import bct
import numpy as np


def test_assortativity_wu_sign():
    x = load_sample(thres=.1)
    ass_pos, _ = bct.local_assortativity_wu_sign(x)

    print(ass_pos, .2939)
    assert np.allclose(np.sum(ass_pos), .2939, atol=.0001)


def test_core_periphery_dir():
    x = load_sample(thres=.1)
    c, q = bct.core_periphery_dir(x)
    assert np.sum(c) == 57
    assert np.sum(np.cumsum(c)) == 4170
    assert np.allclose(q, .3086, atol=.0001)

def test_clique_communities():
    x = load_sample(thres=.23)

    print(np.sum(bct.binarize(x)))

    cis = bct.clique_communities(x, 9)
    print(cis.shape, np.max(np.sum(cis, axis=0)))
    print(np.sum(cis, axis=1))
    assert np.sum(cis) == 199
    assert 4 == 8
