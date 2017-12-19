from load_samples import *
import numpy as np
import bct


def test_pc():
    x = load_sample(thres=.4)
    # ci,q = bct.modularity_und(x)
    ci = np.load(mat_path('sample_partition.npy'))

    pc = np.load(mat_path('sample_pc.npy'))

    pc_ = bct.participation_coef(x, ci)
    print(list(zip(pc, pc_)))

    assert np.allclose(pc, pc_, atol=0.02)


def participation_test():
    W = np.eye(3)
    ci = np.array([1, 1, 2])

    assert np.allclose(bct.participation_coef(W, ci), [0, 0, 0])
    assert np.allclose(bct.participation_coef_sign(W, ci)[0], [0, 0, 0])

    W = np.ones((3, 3))
    assert np.allclose(bct.participation_coef(W, ci), [
                       0.44444444, 0.44444444, 0.44444444])
    assert np.allclose(bct.participation_coef_sign(W, ci)[0], [
                       0.44444444, 0.44444444, 0.44444444])

    W = np.eye(3)
    W[0, 1] = 1
    W[0, 2] = 1
    assert np.allclose(bct.participation_coef(W, ci), [0.44444444, 0, 0])
    assert np.allclose(bct.participation_coef_sign(W, ci)
                       [0], [0.44444444, 0, 0])

    W = np.eye(3)
    W[0, 1] = -1
    W[0, 2] = -1
    W[1, 2] = 1
    assert np.allclose(bct.participation_coef_sign(W, ci)[0], [0.,  0.5,  0.])


def gateway_test():
    x = load_sample(thres=.1)
    ci = np.load(mat_path('sample_partition.npy'))

    g_pos, _ = bct.gateway_coef_sign(x, ci)

    print(np.sum(g_pos), 43.4382)
    assert np.allclose(np.sum(g_pos), 43.4382, atol=.001)

    g_pos_bet, _ = bct.gateway_coef_sign(x, ci, centrality_type='betweenness')

    print(np.sum(g_pos_bet), 43.4026)
    assert np.allclose(np.sum(g_pos_bet), 43.4026, atol=.001)
    

def test_zi():
    x = load_sample(thres=.4)
    ci = np.load(mat_path('sample_partition.npy'))

    zi = np.load(mat_path('sample_zi.npy'))

    zi_ = bct.module_degree_zscore(x, ci)
    print(list(zip(zi, zi_)))

    assert np.allclose(zi, zi_, atol=0.05)
    # this function does the same operations but varies by a modest quantity
    # because of the matlab and numpy differences in population versus
    # sample standard deviation. i tend to think that using the population
    # estimator is acceptable in this case so i will allow the higher
    # tolerance.

# TODO this test does not give the same results, why not


def test_shannon_entropy():
    x = load_sample(thres=0.4)
    ci = np.load(mat_path('sample_partition.npy'))
    # ci, q = bct.modularity_und(x)
    hpos, _ = bct.diversity_coef_sign(x, ci)
    print(np.sum(hpos))
    print(hpos[-1])
    assert np.allclose(np.sum(hpos), 102.6402, atol=.01)
