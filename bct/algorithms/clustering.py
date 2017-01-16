from __future__ import division, print_function
import numpy as np
from .modularity import modularity_louvain_und_sign
from bct.utils import cuberoot, BCTParamError, dummyvar, binarize
from .distance import breadthdist


def agreement(ci, buffsz=1000):
    '''
    Takes as input a set of vertex partitions CI of
    dimensions [vertex x partition]. Each column in CI contains the
    assignments of each vertex to a class/community/module. This function
    aggregates the partitions in CI into a square [vertex x vertex]
    agreement matrix D, whose elements indicate the number of times any two
    vertices were assigned to the same class.

    In the case that the number of nodes and partitions in CI is large
    (greater than ~1000 nodes or greater than ~1000 partitions), the script
    can be made faster by computing D in pieces. The optional input BUFFSZ
    determines the size of each piece. Trial and error has found that
    BUFFSZ ~ 150 works well.

    Parameters
    ----------
    ci : NxM np.ndarray
        set of M (possibly degenerate) partitions of N nodes
    buffsz : int | None
        sets buffer size. If not specified, defaults to 1000

    Returns
    -------
    D : NxN np.ndarray
        agreement matrix
    '''
    ci = np.array(ci)
    n_nodes, n_partitions = ci.shape

    if n_partitions <= buffsz: # Case 1: Use all partitions at once
        ind = dummyvar(ci)
        D = np.dot(ind, ind.T)
    else: # Case 2: Add together results from subsets of partitions
        a = np.arange(0, n_partitions, buffsz)
        b = np.arange(buffsz, n_partitions, buffsz)
        if len(a) != len(b):
            b = np.append(b, n_partitions)
        D = np.zeros((n_nodes, n_nodes))
        for i, j in zip(a, b):
            y = ci[:, i:j]
            ind = dummyvar(y)
            D += np.dot(ind, ind.T)

    np.fill_diagonal(D, 0)
    return D


def agreement_weighted(ci, wts):
    '''
    D = AGREEMENT_WEIGHTED(CI,WTS) is identical to AGREEMENT, with the
    exception that each partitions contribution is weighted according to
    the corresponding scalar value stored in the vector WTS. As an example,
    suppose CI contained partitions obtained using some heuristic for
    maximizing modularity. A possible choice for WTS might be the Q metric
    (Newman's modularity score). Such a choice would add more weight to
    higher modularity partitions.

    NOTE: Unlike AGREEMENT, this script does not have the input argument
    BUFFSZ.

    Parameters
    ----------
    ci : MxN np.ndarray
        set of M (possibly degenerate) partitions of N nodes
    wts : Mx1 np.ndarray
        relative weight of each partition

    Returns
    -------
    D : NxN np.ndarray
        weighted agreement matrix
    '''
    ci = np.array(ci)
    m, n = ci.shape
    wts = np.array(wts) / np.sum(wts)

    D = np.zeros((n, n))
    for i in range(m):
        d = dummyvar(ci[i, :].reshape(1, n))
        D += np.dot(d, d.T) * wts[i]
    return D


def clustering_coef_bd(A):
    '''
    The clustering coefficient is the fraction of triangles around a node
    (equiv. the fraction of nodes neighbors that are neighbors of each other).

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector

    Notes
    -----
    Methodological note: In directed graphs, 3 nodes generate up to 8
    triangles (2*2*2 edges). The number of existing triangles is the main
    diagonal of S^3/2. The number of all (in or out) neighbour pairs is
    K(K-1)/2. Each neighbour pair may generate two triangles. "False pairs"
    are i<->j edge pairs (these do not generate triangles). The number of
    false pairs is the main diagonal of A^2.
    Thus the maximum possible number of triangles =
           = (2 edges)*([ALL PAIRS] - [FALSE PAIRS])
           = 2 * (K(K-1)/2 - diag(A^2))
           = K(K-1) - 2(diag(A^2))
    '''
    S = A + A.T  # symmetrized input graph
    K = np.sum(S, axis=1)  # total degree (in+out)
    cyc3 = np.diag(np.dot(S, np.dot(S, S))) / 2  # number of 3-cycles
    K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, make C=0
    # number of all possible 3 cycles
    CYC3 = K * (K - 1) - 2 * np.diag(np.dot(A, A))
    C = cyc3 / CYC3
    return C


def clustering_coef_bu(G):
    '''
    The clustering coefficient is the fraction of triangles around a node
    (equiv. the fraction of nodes neighbors that are neighbors of each other).

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected connection matrix

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector
    '''
    n = len(G)
    C = np.zeros((n,))

    for u in range(n):
        V, = np.where(G[u, :])
        k = len(V)
        if k >= 2:  # degree must be at least 2
            S = G[np.ix_(V, V)]
            C[u] = np.sum(S) / (k * k - k)

    return C


def clustering_coef_wd(W):
    '''
    The weighted clustering coefficient is the average "intensity" of
    triangles around a node.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted directed connection matrix

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector

    Notes
    -----
    Methodological note (also see clustering_coef_bd)
    The weighted modification is as follows:
    - The numerator: adjacency matrix is replaced with weights matrix ^ 1/3
    - The denominator: no changes from the binary version

    The above reduces to symmetric and/or binary versions of the clustering
    coefficient for respective graphs.
    '''
    A = np.logical_not(W == 0).astype(float)  # adjacency matrix
    S = cuberoot(W) + cuberoot(W.T)  # symmetrized weights matrix ^1/3
    K = np.sum(A + A.T, axis=1)  # total degree (in+out)
    cyc3 = np.diag(np.dot(S, np.dot(S, S))) / 2  # number of 3-cycles
    K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, make C=0
    # number of all possible 3 cycles
    CYC3 = K * (K - 1) - 2 * np.diag(np.dot(A, A))
    C = cyc3 / CYC3  # clustering coefficient
    return C


def clustering_coef_wu(W):
    '''
    The weighted clustering coefficient is the average "intensity" of
    triangles around a node.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted undirected connection matrix

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector
    '''
    K = np.array(np.sum(np.logical_not(W == 0), axis=1), dtype=float)
    ws = cuberoot(W)
    cyc3 = np.diag(np.dot(ws, np.dot(ws, ws)))
    K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, set C=0
    C = cyc3 / (K * (K - 1))
    return C


def clustering_coef_wu_sign(W, coef_type='default'):
    '''
    Returns the weighted clustering coefficient generalized or separated
    for positive and negative weights.
  
    Three Algorithms are supported; herefore referred to as default, zhang,
    and constantini.

    1. Default (Onnela et al.), as in the traditional clustering coefficient
       computation. Computed separately for positive and negative weights.
    2. Zhang & Horvath. Similar to Onnela formula except weight information
       incorporated in denominator. Reduces sensitivity of the measure to
       weights directly connected to the node of interest. Computed
       separately for positive and negative weights.
    3. Constantini & Perugini generalization of Zhang & Horvath formula.
       Takes both positive and negative weights into account simultaneously.
       Particularly sensitive to non-redundancy in path information based on
       sign. Returns only one value.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted undirected connection matrix
    corr_type : enum
        Allowed values are 'default', 'zhang', 'constantini'

    Returns
    -------
    Cpos : Nx1 np.ndarray
        Clustering coefficient vector for positive weights
    Cneg : Nx1 np.ndarray
        Clustering coefficient vector for negative weights, unless
        coef_type == 'constantini'.

    References:
        Onnela et al. (2005) Phys Rev E 71:065103
        Zhang & Horvath (2005) Stat Appl Genet Mol Biol 41:1544-6115
        Costantini & Perugini (2014) PLOS ONE 9:e88669
    '''
    n = len(W)
    np.fill_diagonal(W, 0)

    if coef_type == 'default':
        W_pos = W * (W > 0)
        K_pos = np.array(np.sum(np.logical_not(W_pos == 0), axis=1),
                         dtype=float)
        ws_pos = cuberoot(W_pos)
        cyc3_pos = np.diag(np.dot(ws_pos, np.dot(ws_pos, ws_pos)))
        K_pos[np.where(cyc3_pos == 0)] = np.inf
        C_pos = cyc3_pos / (K_pos * (K_pos - 1))

        W_neg = -W * (W < 0)
        K_neg = np.array(np.sum(np.logical_not(W_neg == 0), axis=1),
                         dtype=float)
        ws_neg = cuberoot(W_neg)
        cyc3_neg = np.diag(np.dot(ws_neg, np.dot(ws_neg, ws_neg)))
        K_neg[np.where(cyc3_neg == 0)] = np.inf
        C_neg = cyc3_neg / (K_neg * (K_neg - 1))

        return C_pos, C_neg

    elif coef_type in ('zhang', 'Zhang'):
        W_pos = W * (W > 0)
        cyc3_pos = np.zeros((n,))
        cyc2_pos = np.zeros((n,))

        W_neg = -W * (W < 0)
        cyc3_neg = np.zeros((n,))
        cyc2_neg = np.zeros((n,))

        for i in range(n):
            for j in range(n):
                for q in range(n):
                    cyc3_pos[i] += W_pos[j, i] * W_pos[i, q] * W_pos[j, q]
                    cyc3_neg[i] += W_neg[j, i] * W_neg[i, q] * W_neg[j, q]
                    if j != q:
                        cyc2_pos[i] += W_pos[j, i] * W_pos[i, q]
                        cyc2_neg[i] += W_neg[j, i] * W_neg[i, q]

        cyc2_pos[np.where(cyc3_pos == 0)] = np.inf
        C_pos = cyc3_pos / cyc2_pos

        cyc2_neg[np.where(cyc3_neg == 0)] = np.inf
        C_neg = cyc3_neg / cyc2_neg

        return C_pos, C_neg

    elif coef_type in ('constantini', 'Constantini'):
        cyc3 = np.zeros((n,))
        cyc2 = np.zeros((n,))

        for i in range(n):
            for j in range(n):
                for q in range(n):
                    cyc3[i] += W[j, i] * W[i, q] * W[j, q]
                    if j != q:
                        cyc2[i] += W[j, i] * W[i, q]

        cyc2[np.where(cyc3 == 0)] = np.inf
        C = cyc3 / cyc2
        return C

def consensus_und(D, tau, reps=1000):
    '''
    This algorithm seeks a consensus partition of the
    agreement matrix D. The algorithm used here is almost identical to the
    one introduced in Lancichinetti & Fortunato (2012): The agreement
    matrix D is thresholded at a level TAU to remove an weak elements. The
    resulting matrix is then partitions REPS number of times using the
    Louvain algorithm (in principle, any clustering algorithm that can
    handle weighted matrixes is a suitable alternative to the Louvain
    algorithm and can be substituted in its place). This clustering
    produces a set of partitions from which a new agreement is built. If
    the partitions have not converged to a single representative partition,
    the above process repeats itself, starting with the newly built
    agreement matrix.

    NOTE: In this implementation, the elements of the agreement matrix must
    be converted into probabilities.

    NOTE: This implementation is slightly different from the original
    algorithm proposed by Lanchichinetti & Fortunato. In its original
    version, if the thresholding produces singleton communities, those
    nodes are reconnected to the network. Here, we leave any singleton
    communities disconnected.

    Parameters
    ----------
    D : NxN np.ndarray
        agreement matrix with entries between 0 and 1 denoting the probability
        of finding node i in the same cluster as node j
    tau : float
        threshold which controls the resolution of the reclustering
    reps : int
        number of times the clustering algorithm is reapplied. default value
        is 1000.

    Returns
    -------
    ciu : Nx1 np.ndarray
        consensus partition
    '''
    def unique_partitions(cis):
        # relabels the partitions to recognize different numbers on same
        # topology

        n, r = np.shape(cis)  # ci represents one vector for each rep
        ci_tmp = np.zeros(n)

        for i in range(r):
            for j, u in enumerate(sorted(
                    np.unique(cis[:, i], return_index=True)[1])):
                ci_tmp[np.where(cis[:, i] == cis[u, i])] = j
            cis[:, i] = ci_tmp
            # so far no partitions have been deleted from ci

        # now squash any of the partitions that are completely identical
        # do not delete them from ci which needs to stay same size, so make
        # copy
        ciu = []
        cis = cis.copy()
        c = np.arange(r)
        # count=0
        while (c != 0).sum() > 0:
            ciu.append(cis[:, 0])
            dup = np.where(np.sum(np.abs(cis.T - cis[:, 0]), axis=1) == 0)
            cis = np.delete(cis, dup, axis=1)
            c = np.delete(c, dup)
            # count+=1
            # print count,c,dup
            # if count>10:
            #	class QualitativeError(): pass
            #	raise QualitativeError()
        return np.transpose(ciu)

    n = len(D)
    flag = True
    while flag:
        flag = False
        dt = D * (D >= tau)
        np.fill_diagonal(dt, 0)

        if np.size(np.where(dt == 0)) == 0:
            ciu = np.arange(1, n + 1)
        else:
            cis = np.zeros((n, reps))
            for i in np.arange(reps):
                cis[:, i], _ = modularity_louvain_und_sign(dt)
            ciu = unique_partitions(cis)
            nu = np.size(ciu, axis=1)
            if nu > 1:
                flag = True
                D = agreement(cis) / reps

    return np.squeeze(ciu + 1)


def get_components(A, no_depend=False):
    '''
    Returns the components of an undirected graph specified by the binary and
    undirected adjacency matrix adj. Components and their constitutent nodes
    are assigned the same index and stored in the vector, comps. The vector,
    comp_sizes, contains the number of nodes beloning to each component.

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected adjacency matrix
    no_depend : Any
        Does nothing, included for backwards compatibility

    Returns
    -------
    comps : Nx1 np.ndarray
        vector of component assignments for each node
    comp_sizes : Mx1 np.ndarray
        vector of component sizes

    Notes
    -----
    Note: disconnected nodes will appear as components with a component
    size of 1

    Note: The identity of each component (i.e. its numerical value in the
    result) is not guaranteed to be identical the value returned in BCT,
    matlab code, although the component topology is.

    Many thanks to Nick Cullen for providing this implementation
    '''

    if not np.all(A == A.T):  # ensure matrix is undirected
        raise BCTParamError('get_components can only be computed for undirected'
                            ' matrices.  If your matrix is noisy, correct it with np.around')
    
    A = binarize(A, copy=True)
    n = len(A)
    np.fill_diagonal(A, 1)

    edge_map = [{u,v} for u in range(n) for v in range(n) if A[u,v] == 1]
    union_sets = []
    for item in edge_map:
        temp = []
        for s in union_sets:

            if not s.isdisjoint(item):
                item = s.union(item)
            else:
                temp.append(s)
        temp.append(item)
        union_sets = temp

    comps = np.array([i+1 for v in range(n) for i in 
        range(len(union_sets)) if v in union_sets[i]])
    comp_sizes = np.array([len(s) for s in union_sets])

    return comps, comp_sizes


def get_components_old(A, no_depend=False):
    '''
    Returns the components of an undirected graph specified by the binary and
    undirected adjacency matrix adj. Components and their constitutent nodes
    are assigned the same index and stored in the vector, comps. The vector,
    comp_sizes, contains the number of nodes beloning to each component.

    Parameters
    ----------
    adj : NxN np.ndarray
        binary undirected adjacency matrix
    no_depend : bool
        If true, doesn't import networkx to do the calculation. Default value
        is false.

    Returns
    -------
    comps : Nx1 np.ndarray
        vector of component assignments for each node
    comp_sizes : Mx1 np.ndarray
        vector of component sizes

    Notes
    -----
    Note: disconnected nodes will appear as components with a component
    size of 1

    Note: The identity of each component (i.e. its numerical value in the
    result) is not guaranteed to be identical the value returned in BCT,
    although the component topology is.

    Note: networkx is used to do the computation efficiently. If networkx is
    not available a breadth-first search that does not depend on networkx is
    used instead, but this is less efficient. The corresponding BCT function
    does the computation by computing the Dulmage-Mendelsohn decomposition. I
    don't know what a Dulmage-Mendelsohn decomposition is and there doesn't
    appear to be a python equivalent. If you think of a way to implement this
    better, let me know.
        '''
    # nonsquare matrices cannot be symmetric; no need to check

    if not np.all(A == A.T):  # ensure matrix is undirected
        raise BCTParamError('get_components can only be computed for undirected'
                            ' matrices.  If your matrix is noisy, correct it with np.around')

    A = binarize(A, copy=True)
    n = len(A)
    np.fill_diagonal(A, 1)

    try:
        if no_depend:
            raise ImportError()
        else:
            import networkx as nx
        net = nx.from_numpy_matrix(A)
        cpts = list(nx.connected_components(net))

        cptvec = np.zeros((n,))
        cptsizes = np.zeros(len(cpts))
        for i, cpt in enumerate(cpts):
            cptsizes[i] = len(cpt)
            for node in cpt:
                cptvec[node] = i + 1

    except ImportError:
        # if networkx is not available use less efficient breadth first search
        cptvec = np.zeros((n,))
        r, _ = breadthdist(A)
        for node, reach in enumerate(r):
            if cptvec[node] > 0:
                continue
            else:
                cptvec[np.where(reach)] = np.max(cptvec) + 1

        cptsizes = np.zeros(np.max(cptvec))
        for i in np.arange(np.max(cptvec)):
            cptsizes[i] = np.size(np.where(cptvec == i + 1))

    return cptvec, cptsizes


def number_of_components(A):
    _, csizes = get_components(A)
    return len(csizes)


def transitivity_bd(A):
    '''
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    T : float
        transitivity scalar

    Notes
    -----
    Methodological note: In directed graphs, 3 nodes generate up to 8
    triangles (2*2*2 edges). The number of existing triangles is the main

    diagonal of S^3/2. The number of all (in or out) neighbour pairs is
    K(K-1)/2. Each neighbour pair may generate two triangles. "False pairs"
    are i<->j edge pairs (these do not generate triangles). The number of
    false pairs is the main diagonal of A^2. Thus the maximum possible
    number of triangles = (2 edges)*([ALL PAIRS] - [FALSE PAIRS])
                        = 2 * (K(K-1)/2 - diag(A^2))
                        = K(K-1) - 2(diag(A^2))
    '''
    S = A + A.T  # symmetrized input graph
    K = np.sum(S, axis=1)  # total degree (in+out)
    cyc3 = np.diag(np.dot(S, np.dot(S, S))) / 2  # number of 3-cycles
    CYC3 = K * (K - 1) - 2 * np.diag(np.dot(A, A))  # number of all possible 3-cycles
    return np.sum(cyc3) / np.sum(CYC3)


def transitivity_bu(A):
    '''
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected connection matrix

    Returns
    -------
    T : float
        transitivity scalar
    '''
    tri3 = np.trace(np.dot(A, np.dot(A, A)))
    tri2 = np.sum(np.dot(A, A)) - np.trace(np.dot(A, A))
    return tri3 / tri2


def transitivity_wd(W):
    '''
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).

    Parameters
    ----------
    W : NxN np.ndarray
        weighted directed connection matrix

    Returns
    -------
    T : int
        transitivity scalar

    Methodological note (also see note for clustering_coef_bd)
    The weighted modification is as follows:
    - The numerator: adjacency matrix is replaced with weights matrix ^ 1/3
    - The denominator: no changes from the binary version

    The above reduces to symmetric and/or binary versions of the clustering
    coefficient for respective graphs.
    '''
    A = np.logical_not(W == 0).astype(float)  # adjacency matrix
    S = cuberoot(W) + cuberoot(W.T)  # symmetrized weights matrix ^1/3
    K = np.sum(A + A.T, axis=1)  # total degree (in+out)
    cyc3 = np.diag(np.dot(S, np.dot(S, S))) / 2  # number of 3-cycles
    K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, make T=0
    # number of all possible 3-cycles
    CYC3 = K * (K - 1) - 2 * np.diag(np.dot(A, A))
    return np.sum(cyc3) / np.sum(CYC3)  # transitivity


def transitivity_wu(W):
    '''
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).

    Parameters
    ----------
    W : NxN np.ndarray
        weighted undirected connection matrix

    Returns
    -------
    T : int
        transitivity scalar
    '''
    K = np.sum(np.logical_not(W == 0), axis=1)
    ws = cuberoot(W)
    cyc3 = np.diag(np.dot(ws, np.dot(ws, ws)))
    return np.sum(cyc3, axis=0) / np.sum(K * (K - 1), axis=0)
