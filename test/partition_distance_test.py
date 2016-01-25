from load_samples import *
import numpy as np
import bct


def test_partition_distance():
    q = load_sample_group_qball()
    d = load_sample_group_dsi()

    q = np.mean(q, axis=2)
    d = np.mean(d, axis=2)

    qi, _ = bct.modularity_und(q)
    di, _ = bct.modularity_und(d)

    vi, mi = bct.partition_distance(qi, di)

    print(vi, mi)
    assert np.allclose(vi, 0.1964, atol=0.01)
    assert np.allclose(mi, 0.6394, atol=0.01)
