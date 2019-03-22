import pytest
import numpy as np

import bct

from .load_samples import load_sample


@pytest.mark.long
def test_link_communities():
    x = load_sample(thres=0.4)
    seed = 949389104
    M = bct.link_communities(x)
    assert np.max(M) == 1
