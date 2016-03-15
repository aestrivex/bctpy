from load_samples import *
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
