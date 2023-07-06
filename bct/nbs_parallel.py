from __future__ import division, print_function
import numpy as np
import multiprocessing

from .utils import BCTParamError, get_rng
from .algorithms import get_components
from .due import due, BibTeX
from .citations import ZALESKY2010

@due.dcite(BibTeX(ZALESKY2010), description="Network-based statistic")

def ttest2_stat_only(x, y, tail):
    t = np.mean(x) - np.mean(y)
    n1, n2 = len(x), len(y)
    s = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1)
                    * np.var(y, ddof=1)) / (n1 + n2 - 2))
    denom = s * np.sqrt(1 / n1 + 1 / n2)
    if denom == 0:
        return 0
    if tail == 'both':
        return np.abs(t / denom)
    if tail == 'left':
        return -t / denom
    else:
        return t / denom

def ttest_paired_stat_only(A, B, tail):
    n = len(A - B)
    df = n - 1
    sample_ss = np.sum((A - B)**2) - np.sum(A - B)**2 / n
    unbiased_std = np.sqrt(sample_ss / (n - 1))
    z = np.mean(A - B) / unbiased_std
    t = z * np.sqrt(n)
    if tail == 'both':
        return np.abs(t)
    if tail == 'left':
        return -t
    else:
        return t
    
def _permutation(args):
    seed, u, xmat, ymat, thresh, tail, paired, m, n, ixes, nx, ny, verbose, null, max_sz, hit, k = args

    if seed is None:
        seed = u
    rng = get_rng(seed)
    if paired:
        indperm = np.sign(0.5 - rng.rand(1, nx))
        d = np.hstack((xmat, ymat)) * np.hstack((indperm, indperm))
    else:
        d = np.hstack((xmat, ymat))[:, rng.permutation(nx + ny)]

    t_stat_perm = np.zeros((m,))
    for i in range(m):
        if paired:
            t_stat_perm[i] = ttest_paired_stat_only(
                d[i, :nx], d[i, -nx:], tail)
        else:
            t_stat_perm[i] = ttest2_stat_only(d[i, :nx], d[i, -ny:], tail)

    ind_t, = np.where(t_stat_perm > thresh)

    adj_perm = np.zeros((n, n))
    adj_perm[(ixes[0][ind_t], ixes[1][ind_t])] = 1
    adj_perm = adj_perm + adj_perm.T

    a, sz = get_components(adj_perm)

    ind_sz, = np.where(sz > 1)
    ind_sz += 1
    nr_components_perm = np.size(ind_sz)
    sz_links_perm = np.zeros((nr_components_perm))
    for i in range(nr_components_perm):
        nodes, = np.where(ind_sz[i] == a)
        sz_links_perm[i] = np.sum(adj_perm[np.ix_(nodes, nodes)]) / 2

    if np.size(sz_links_perm):
        null[u] = np.max(sz_links_perm)
    else:
        null[u] = 0

    # compare to the true dataset
    if null[u] >= max_sz:
        hit += 1

    if verbose:
        print(('permutation %i of %i.  Permutation max is %s.  Observed max is %s.') % 
               (u + 1, k, null[u], max_sz))
    elif (u % (k / 10) == 0 or u == k - 1):
        print('permutation %i of %i.' % (u + 1, k))
    return null

def nbs_bct(x, y, thresh, k=1000, tail='both', paired=False, verbose=False, seed=None, workers=-1):

    if tail not in ('both', 'left', 'right'):
        raise BCTParamError('Tail must be both, left, right')

    ix, jx, nx = x.shape
    iy, jy, ny = y.shape

    if not ix == jx == iy == jy:
        raise BCTParamError('Population matrices are of inconsistent size')
    else:
        n = ix

    if paired and nx != ny:
        raise BCTParamError('Population matrices must be an equal size')

    # only consider upper triangular edges
    ixes = np.where(np.triu(np.ones((n, n)), 1))

    # number of edges
    m = np.size(ixes, axis=1)

    # vectorize connectivity matrices for speed
    xmat, ymat = np.zeros((m, nx)), np.zeros((m, ny))
    
    for i in range(nx):
        xmat[:, i] = x[:, :, i][ixes].squeeze()
    for i in range(ny):
        ymat[:, i] = y[:, :, i][ixes].squeeze()
    del x, y

    # perform t-test at each edge
    t_stat = np.zeros((m,))
    for i in range(m):
        if paired:
            t_stat[i] = ttest_paired_stat_only(xmat[i, :], ymat[i, :], tail)
        else:
            t_stat[i] = ttest2_stat_only(xmat[i, :], ymat[i, :], tail)

    # threshold
    ind_t, = np.where(t_stat > thresh)

    if len(ind_t) == 0:
        raise BCTParamError("Unsuitable threshold")

    # suprathreshold adjacency matrix
    adj = np.zeros((n, n))
    adj[(ixes[0][ind_t], ixes[1][ind_t])] = 1
    # adj[ixes][ind_t]=1
    adj = adj + adj.T

    a, sz = get_components(adj)

    # convert size from nodes to number of edges
    # only consider components comprising more than one node (e.g. a/l 1 edge)
    ind_sz, = np.where(sz > 1)
    ind_sz += 1
    nr_components = np.size(ind_sz)
    sz_links = np.zeros((nr_components,))
    for i in range(nr_components):
        nodes, = np.where(ind_sz[i] == a)
        sz_links[i] = np.sum(adj[np.ix_(nodes, nodes)]) / 2
        adj[np.ix_(nodes, nodes)] *= (i + 2)

    # subtract 1 to delete any edges not comprising a component
    adj[np.where(adj)] -= 1

    if np.size(sz_links):
        max_sz = np.max(sz_links)
    else:
        # max_sz=0
        raise BCTParamError('True matrix is degenerate')
    print('max component size is %i' % max_sz)

    print('Estimating null distribution with %i permutations. P-values will be returned at the end of the test.' % k)

    null = np.zeros((k,))
    hit = 0
    if workers == -1:
        workers = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(workers)
    perm_args = [(seed, u, xmat, ymat, thresh, tail, paired, m, n, ixes, nx, ny, verbose, null, max_sz, hit, k) for u in range(k)]

    # Parallelize permutation
    null_dist = pool.map(_permutation, perm_args)

    pool.close()
    pool.join()

    null_dist = np.array(null_dist)
    null_dist = np.array([max(i) for i in null_dist.T])

    pvals = np.zeros((nr_components,))
    # calculate p-vals
    for i in range(nr_components):
        pvals[i] = np.size(np.where(null >= sz_links[i])) / k

    return pvals, adj, null_dist