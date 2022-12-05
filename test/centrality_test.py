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
    assert np.allclose(np.sum(gp), 87.0141)
    assert np.allclose(np.sum(gpb), 87.0742)
    assert np.all(gn == 0)
    assert np.all(gnb == 0)
