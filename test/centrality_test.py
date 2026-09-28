from .load_samples import (
    load_sample, load_signed_sample, load_sparse_sample,
    load_directed_low_modularity_sample, load_binary_directed_low_modularity_sample
)
import numpy as np
import bct


def test_gateway_coef():
    x = load_sample(thres=.41)
    ci, _ = bct.modularity_und(x)
    gp, gn = bct.gateway_coef_sign(x, ci)
    gpb, gnb = bct.gateway_coef_sign(x, ci, centrality_type='betweenness')
    assert np.allclose(np.sum(gp), 89.6140)
    assert np.allclose(np.sum(gpb), 89.7742)
    assert np.all(gn == 0)
    assert np.all(gnb == 0)



def test_gateway_coef_handles_later_small_communities():
    W = np.ones((6, 6)) - np.eye(6)
    ci = np.repeat([1, 2, 3], 2)

    gp, gn = bct.gateway_coef_sign(W, ci)

    assert np.all(np.isfinite(gp))
    assert np.allclose(gp, gp[0])
    assert np.all(gn == 0)


def test_gateway_coef_uses_neighbor_node_indices():
    W = np.array([
        [0, 1, 4, 0, 0, 0],
        [1, 0, 0, 2, 0, 0],
        [4, 0, 0, 1, 3, 0],
        [0, 2, 1, 0, 0, 5],
        [0, 0, 3, 0, 0, 1],
        [0, 0, 0, 5, 1, 0],
    ], dtype=float)
    ci = np.array([1, 2, 1, 2, 1, 2])

    gp, _ = bct.gateway_coef_sign(W, ci)

    assert np.allclose(
        gp,
        [0.62248994, 0.57664286, 0.81934232, 0.81934232, 0.59285185, 0.67035485],
    )
