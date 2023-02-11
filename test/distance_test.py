from .load_samples import load_sample
import numpy as np
import bct


def test_breadthdist():
    x = load_sample(thres=.02)
    r, d = bct.breadthdist(x)
    d[np.where(np.isinf(d))] = 0
    print(np.sum(r), np.sum(d))
    assert np.sum(r) == 5804
    assert np.sum(d) == 30762


def test_reachdist():
    x = load_sample(thres=.02)
    r, d = bct.reachdist(x)
    d[np.where(np.isinf(d))] = 0
    print(np.sum(r), np.sum(d))
    assert np.sum(r) == 5804
    assert np.sum(d) == 30762

    bx = bct.binarize(x, copy=False)
    br, bd = bct.reachdist(bx)
    bd[np.where(np.isinf(bd))] = 0
    print(np.sum(br), np.sum(bd))
    assert np.sum(br) == 5804
    assert np.sum(bd) == 30762


def test_distance_bin():
    x = bct.binarize(load_sample(thres=.02), copy=False)
    d = bct.distance_bin(x)
    d[np.where(np.isinf(d))] = 0
    print(np.sum(d))
    assert np.sum(d) == 30506  # deals with diagonals differently


def test_distance_wei():
    x = load_sample(thres=.02)
    d, e = bct.distance_wei(x)
    d[np.where(np.isinf(d))] = 0
    print(np.sum(d), np.sum(e))

    assert np.allclose(np.sum(d), 155650.1, atol=.01)
    assert np.sum(e) == 30570


def test_charpath():
    x = load_sample(thres=.02)
    d, e = bct.distance_wei(x)
    l, eff,ecc,radius,diameter = bct.charpath(d)

    assert np.any(np.isinf(d))
    assert not np.isnan(radius)
    assert not np.isnan(diameter)

def test_distance_floyd():
    x = load_sample(thres=.31)
    spli, hopsi, pmati = bct.distance_wei_floyd(x, transform='inv')
    print(np.sum(spli))
    assert np.allclose(np.sum(spli), 11536.1, atol=.01)
    
def test_navigation_wu():
    x = load_sample(thres=.24)
    x_len = bct.invert(x)

    #randomly generate distances for testing purposes
    n = len(x)
    while True:
        centroids = np.random.randint(512, size=(n, 3))
        #make sure every centroid is unique
        if len(np.unique(centroids, axis=0)) == n:
            break

    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = np.linalg.norm(centroids[i, :] - centroids[j, :])
            
        
    sr, plbin, plwei, pldis, paths = bct.navigation_wu(x_len, d, max_hops=14)

    sr2, plbin2, plwei2, pldis2, paths2 = bct.navigation_wu(x_len, d, max_hops=None)

    #TODO create the centroids for an actual bit of sample data and converge the matlab algorithm
    #this procedure of random centroid generation generates a random reachability which is usually around 45-60%
    #but not guaranteed

