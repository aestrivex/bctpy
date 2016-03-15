from load_samples import *
import numpy as np
import bct


def test_nbs_dsi_qbi():
    q = load_sample_group_qball()
    d = load_sample_group_dsi()
    _nbs_helper(q, d, .5, atol=0.3)


def test_nbs_paired_dsi_qbi():
    pass


def test_nbs_dsi_fmri():
    d = load_sample_group_dsi()
    f = load_sample_group_fmri()
    assert f.shape == (219, 219, 8)
    _nbs_helper(d, f, .03, atol=0.03)


def test_nbs_paired_dsi_fmri():
    pass


def _nbs_helper(x, y, expected_pval, atol=.05, thresh=.1, ntrials=25,
                paired=False):
    # comment

    pval, _, _ = bct.nbs_bct(x, y, thresh, k=ntrials, paired=paired)
    print(pval, expected_pval)
    assert np.allclose(pval, expected_pval, atol=atol)
