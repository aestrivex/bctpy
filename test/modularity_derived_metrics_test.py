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

def test_zi():
    x = load_sample(thres=.4)
    ci = np.load('mats/sample_partition.npy')

    zi = np.load('mats/sample_zi.npy')

    zi_ = bct.module_degree_zscore(x, ci)
    print zip(zi,zi_)

    assert np.allclose(zi, zi_, atol=0.05)
    #this function does the same operations but varies by a modest quantity
    #because of the matlab and numpy differences in population versus
    #sample standard deviation. i tend to think that using the population
    #estimator is acceptable in this case so i will allow the higher
    #tolerance.
