from load_samples import *
import numpy as np
import bct

def test_pc():
    x = load_sample(thres=.4)
    #ci,q = bct.modularity_und(x)
    ci = np.load('mats/sample_partition.npy')

    pc = np.load('mats/sample_pc.npy')

    pc_ = bct.participation_coef(x, ci)
    print zip(pc, pc_)

    assert np.allclose(pc, pc_, atol=0.02)
