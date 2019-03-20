from __future__ import division, print_function
import random
import numpy as np


class BCTParamError(RuntimeError):
    pass


def teachers_round(x):
    '''
    Do rounding such that .5 always rounds to 1, and not bankers rounding.
    This is for compatibility with matlab functions, and ease of testing.
    '''
    if ((x > 0) and (x % 1 >= 0.5)) or ((x < 0) and (x % 1 > 0.5)):
        return int(np.ceil(x))
    else:
        return int(np.floor(x))


def pick_four_unique_nodes_quickly(n, seed=None):
    '''
    This is equivalent to np.random.choice(n, 4, replace=False)

    Another fellow suggested np.random.random_sample(n).argpartition(4) which is
    clever but still substantially slower.
    '''
    rng = get_rng(seed)
    k = rng.randint(n**4)
    a = k % n
    b = k // n % n
    c = k // n ** 2 % n
    d = k // n ** 3 % n
    if (a != b and a != c and a != d and b != c and b != d and c != d):
        return (a, b, c, d)
    else:
        # the probability of finding a wrong configuration is extremely low
        # unless for extremely small n. if n is extremely small the
        # computational demand is not a problem.

        # In my profiling it only took 0.4 seconds to include the uniqueness
        # check in 1 million runs of this function so I think it is OK.
        return pick_four_unique_nodes_quickly(n, rng)


def cuberoot(x):
    '''
    Correctly handle the cube root for negative weights, instead of uselessly
    crashing as in python or returning the wrong root as in matlab
    '''
    return np.sign(x) * np.abs(x)**(1 / 3)


def dummyvar(cis, return_sparse=False):
    '''
    This is an efficient implementation of matlab's "dummyvar" command
    using sparse matrices.

    input: partitions, NxM array-like containing M partitions of N nodes
        into <=N distinct communities

    output: dummyvar, an NxR matrix containing R column variables (indicator
        variables) with N entries, where R is the total number of communities
        summed across each of the M partitions.

        i.e.
        r = sum((max(len(unique(partitions[i]))) for i in range(m)))
    '''
    # num_rows is not affected by partition indexes
    n = np.size(cis, axis=0)
    m = np.size(cis, axis=1)
    r = np.sum((np.max(len(np.unique(cis[:, i])))) for i in range(m))
    nnz = np.prod(cis.shape)

    ix = np.argsort(cis, axis=0)
    # s_cis=np.sort(cis,axis=0)
    # FIXME use the sorted indices to sort by row efficiently
    s_cis = cis[ix][:, range(m), range(m)]

    mask = np.hstack((((True,),) * m, (s_cis[:-1, :] != s_cis[1:, :]).T))
    indptr, = np.where(mask.flat)
    indptr = np.append(indptr, nnz)

    import scipy.sparse as sp
    dv = sp.csc_matrix((np.repeat((1,), nnz), ix.T.flat, indptr), shape=(n, r))
    return dv.toarray()


def get_rng(seed=None):
    """
    By default, or if `seed` is np.random, return the global RandomState
    instance used by np.random.
    If `seed` is a RandomState instance, return it unchanged.
    Otherwise, use the passed (hashable) argument to seed a new instance
    of RandomState and return it.

    Parameters
    ----------
    seed : hashable or np.random.RandomState or np.random, optional

    Returns
    -------
    np.random.RandomState
    """
    if seed is None or seed == np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, np.random.RandomState):
        return seed
    try:
        rstate =  np.random.RandomState(seed)
    except ValueError:
        rstate = np.random.RandomState(random.Random(seed).randint(0, 2**32-1))
    return rstate

