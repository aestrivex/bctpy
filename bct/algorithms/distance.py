from __future__ import division, print_function
import numpy as np
from bct.utils import cuberoot, binarize, invert


def breadthdist(CIJ):
    '''
    The binary reachability matrix describes reachability between all pairs
    of nodes. An entry (u,v)=1 means that there exists a path from node u
    to node v; alternatively (u,v)=0.

    The distance matrix contains lengths of shortest paths between all
    pairs of nodes. An entry (u,v) represents the length of shortest path
    from node u to  node v. The average shortest path length is the
    characteristic path length of the network.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed/undirected connection matrix

    Returns
    -------
    R : NxN np.ndarray
        binary reachability matrix
    D : NxN np.ndarray
        distance matrix

    Notes
    -----
    slower but less memory intensive than "reachdist.m".
    '''
    n = len(CIJ)

    D = np.zeros((n, n))
    for i in range(n):
        D[i, :], _ = breadth(CIJ, i)

    D[D == 0] = np.inf
    R = (D != np.inf)
    return R, D


def breadth(CIJ, source):
    '''
    Implementation of breadth-first search.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed/undirected connection matrix
    source : int
        source vertex

    Returns
    -------
    distance : Nx1 np.ndarray
        vector of distances between source and ith vertex (0 for source)
    branch : Nx1 np.ndarray
        vertex that precedes i in the breadth-first search (-1 for source)

    Notes
    -----
    Breadth-first search tree does not contain all paths (or all
    shortest paths), but allows the determination of at least one path with
    minimum distance. The entire graph is explored, starting from source
    vertex 'source'.
    '''
    n = len(CIJ)

    # colors: white,gray,black
    white = 0
    gray = 1
    black = 2

    color = np.zeros((n,))
    distance = np.inf * np.ones((n,))
    branch = np.zeros((n,))

    # start on vertex source
    color[source] = gray
    distance[source] = 0
    branch[source] = -1
    Q = [source]

    # keep going until the entire graph is explored
    while Q:
        u = Q[0]
        ns, = np.where(CIJ[u, :])
        for v in ns:
            # this allows the source distance itself to be recorded
            if distance[v] == 0:
                distance[v] = distance[u] + 1
            if color[v] == white:
                color[v] = gray
                distance[v] = distance[u] + 1
                branch[v] = u
                Q.append(v)
        Q = Q[1:]
        color[u] = black

    return distance, branch


def charpath(D, include_diagonal=False, include_infinite=True):
    '''
    The characteristic path length is the average shortest path length in
    the network. The global efficiency is the average inverse shortest path
    length in the network.

    Parameters
    ----------
    D : NxN np.ndarray
        distance matrix
    include_diagonal : bool
        If True, include the weights on the diagonal. Default value is False.
    include_infinite : bool
        If True, include infinite distances in calculation

    Returns
    -------
    lambda : float
        characteristic path length
    efficiency : float
        global efficiency
    ecc : Nx1 np.ndarray
        eccentricity at each vertex
    radius : float
        radius of graph
    diameter : float
        diameter of graph

    Notes
    -----
    The input distance matrix may be obtained with any of the distance
    functions, e.g. distance_bin, distance_wei.
    Characteristic path length is calculated as the global mean of
    the distance matrix D, excludings any 'Infs' but including distances on
    the main diagonal.
    '''
    D = D.copy()

    if not include_diagonal:
        np.fill_diagonal(D, np.nan)

    if not include_infinite:
        D[np.isinf(D)] = np.nan

    Dv = D[np.logical_not(np.isnan(D))].ravel()

    # mean of finite entries of D[G]
    lambda_ = np.mean(Dv)

    # efficiency: mean of inverse entries of D[G]
    efficiency = np.mean(1 / Dv)

    # eccentricity for each vertex (ignore inf)
    ecc = np.array(np.ma.masked_where(np.isnan(D), D).max(axis=1))

    # radius of graph
    radius = np.min(ecc)  # but what about zeros?

    # diameter of graph
    diameter = np.max(ecc)

    return lambda_, efficiency, ecc, radius, diameter


def cycprob(Pq):
    '''
    Cycles are paths which begin and end at the same node. Cycle
    probability for path length d, is the fraction of all paths of length
    d-1 that may be extended to form cycles of length d.

    Parameters
    ----------
    Pq : NxNxQ np.ndarray
        Path matrix with Pq[i,j,q] = number of paths from i to j of length q.
        Produced by findpaths()

    Returns
    -------
    fcyc : Qx1 np.ndarray
        fraction of all paths that are cycles for each path length q
    pcyc : Qx1 np.ndarray
        probability that a non-cyclic path of length q-1 can be extended to
        form a cycle of length q for each path length q
    '''

    # note: fcyc[1] must be zero, as there cannot be cycles of length 1
    fcyc = np.zeros(np.size(Pq, axis=2))
    for q in range(np.size(Pq, axis=2)):
        if np.sum(Pq[:, :, q]) > 0:
            fcyc[q] = np.sum(np.diag(Pq[:, :, q])) / np.sum(Pq[:, :, q])
        else:
            fcyc[q] = 0

    # note: pcyc[1] is not defined (set to zero)
    # note: pcyc[2] is equal to the fraction of reciprocal connections
    # note: there are no non-cyclic paths of length N and no cycles of len N+1
    pcyc = np.zeros(np.size(Pq, axis=2))
    for q in range(np.size(Pq, axis=2)):
        if np.sum(Pq[:, :, q - 1]) - np.sum(np.diag(Pq[:, :, q - 1])) > 0:
            pcyc[q] = (np.sum(np.diag(Pq[:, :, q - 1])) /
                       np.sum(Pq[:, :, q - 1]) - np.sum(np.diag(Pq[:, :, q - 1])))
        else:
            pcyc[q] = 0

    return fcyc, pcyc


def distance_bin(G):
    '''
    The distance matrix contains lengths of shortest paths between all
    pairs of nodes. An entry (u,v) represents the length of shortest path
    from node u to node v. The average shortest path length is the
    characteristic path length of the network.

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed/undirected connection matrix

    Returns
    -------
    D : NxN
        distance matrix

    Notes
    -----
    Lengths between disconnected nodes are set to Inf.
    Lengths on the main diagonal are set to 0.
    Algorithm: Algebraic shortest paths.
    '''
    G = binarize(G, copy=True)
    D = np.eye(len(G))
    n = 1
    nPATH = G.copy()  # n path matrix
    L = (nPATH != 0)  # shortest n-path matrix

    while np.any(L):
        D += n * L
        n += 1
        nPATH = np.dot(nPATH, G)
        L = (nPATH != 0) * (D == 0)

    D[D == 0] = np.inf  # disconnected nodes are assigned d=inf
    np.fill_diagonal(D, 0)
    return D


def distance_wei(G):
    '''
    The distance matrix contains lengths of shortest paths between all
    pairs of nodes. An entry (u,v) represents the length of shortest path
    from node u to node v. The average shortest path length is the
    characteristic path length of the network.

    Parameters
    ----------
    L : NxN np.ndarray
        Directed/undirected connection-length matrix.
        NB L is not the adjacency matrix. See below.

    Returns
    -------
    D : NxN np.ndarray
        distance (shortest weighted path) matrix
    B : NxN np.ndarray
        matrix of number of edges in shortest weighted path

    Notes
    -----
       The input matrix must be a connection-length matrix, typically
    obtained via a mapping from weight to length. For instance, in a
    weighted correlation network higher correlations are more naturally
    interpreted as shorter distances and the input matrix should
    consequently be some inverse of the connectivity matrix.
       The number of edges in shortest weighted paths may in general
    exceed the number of edges in shortest binary paths (i.e. shortest
    paths computed on the binarized connectivity matrix), because shortest
    weighted paths have the minimal weighted distance, but not necessarily
    the minimal number of edges.
       Lengths between disconnected nodes are set to Inf.
       Lengths on the main diagonal are set to 0.

    Algorithm: Dijkstra's algorithm.
    '''
    n = len(G)
    D = np.zeros((n, n))  # distance matrix
    D[np.logical_not(np.eye(n))] = np.inf
    B = np.zeros((n, n))  # number of edges matrix

    for u in range(n):
        # distance permanence (true is temporary)
        S = np.ones((n,), dtype=bool)
        G1 = G.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                W, = np.where(G1[v, :])  # neighbors of shortest nodes

                td = np.array(
                    [D[u, W].flatten(), (D[u, v] + G1[v, W]).flatten()])
                d = np.min(td, axis=0)
                wi = np.argmin(td, axis=0)

                D[u, W] = d  # smallest of old/new path lengths
                ind = W[np.where(wi == 1)]  # indices of lengthened paths
                # increment nr_edges for lengthened paths
                B[u, ind] = B[u, v] + 1

            if D[u, S].size == 0:  # all nodes reached
                break
            minD = np.min(D[u, S])
            if np.isinf(minD):  # some nodes cannot be reached
                break

            V, = np.where(D[u, :] == minD)

    return D, B


def efficiency_bin(G, local=False):
    '''
    The global efficiency is the average of inverse shortest path length,
    and is inversely related to the characteristic path length.

    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected connection matrix
    local : bool
        If True, computes local efficiency instead of global efficiency.
        Default value = False.

    Returns
    -------
    Eglob : float
        global efficiency, only if local=False
    Eloc : Nx1 np.ndarray
        local efficiency, only if local=True
    '''
    def distance_inv(g):
        D = np.eye(len(g))
        n = 1
        nPATH = g.copy()
        L = (nPATH != 0)

        while np.any(L):
            D += n * L
            n += 1
            nPATH = np.dot(nPATH, g)
            L = (nPATH != 0) * (D == 0)
        D[np.logical_not(D)] = np.inf
        D = 1 / D
        np.fill_diagonal(D, 0)
        return D

    G = binarize(G)
    n = len(G)  # number of nodes
    if local:
        E = np.zeros((n,))  # local efficiency

        for u in range(n):
            # V,=np.where(G[u,:])			#neighbors
            # k=len(V)					#degree
            # if k>=2:					#degree must be at least 2
            #	e=distance_inv(G[V].T[V])
            #	E[u]=np.sum(e)/(k*k-k)	#local efficiency computation

            # find pairs of neighbors
            V, = np.where(np.logical_or(G[u, :], G[u, :].T))
            # inverse distance matrix
            e = distance_inv(G[np.ix_(V, V)])
            # symmetrized inverse distance matrix
            se = e + e.T

            # symmetrized adjacency vector
            sa = G[u, V] + G[V, u].T
            numer = np.sum(np.outer(sa.T, sa) * se) / 2
            if numer != 0:
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                E[u] = numer / denom  # local efficiency

    else:
        e = distance_inv(G)
        E = np.sum(e) / (n * n - n)  # global efficiency
    return E


def efficiency_wei(Gw, local=False):
    '''
    The global efficiency is the average of inverse shortest path length,
    and is inversely related to the characteristic path length.

    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted connection matrix
        (all weights in W must be between 0 and 1)
    local : bool
        If True, computes local efficiency instead of global efficiency.
        Default value = False.

    Returns
    -------
    Eglob : float
        global efficiency, only if local=False
    Eloc : Nx1 np.ndarray
        local efficiency, only if local=True

    Notes
    -----
       The  efficiency is computed using an auxiliary connection-length
    matrix L, defined as L_ij = 1/W_ij for all nonzero L_ij; This has an
    intuitive interpretation, as higher connection weights intuitively
    correspond to shorter lengths.
       The weighted local efficiency broadly parallels the weighted
    clustering coefficient of Onnela et al. (2005) and distinguishes the
    influence of different paths based on connection weights of the
    corresponding neighbors to the node in question. In other words, a path
    between two neighbors with strong connections to the node in question
    contributes more to the local efficiency than a path between two weakly
    connected neighbors. Note that this weighted variant of the local
    efficiency is hence not a strict generalization of the binary variant.

    Algorithm:  Dijkstra's algorithm
    '''
    def distance_inv_wei(G):
        n = len(G)
        D = np.zeros((n, n))  # distance matrix
        D[np.logical_not(np.eye(n))] = np.inf

        for u in range(n):
            # distance permanence (true is temporary)
            S = np.ones((n,), dtype=bool)
            G1 = G.copy()
            V = [u]
            while True:
                S[V] = 0  # distance u->V is now permanent
                G1[:, V] = 0  # no in-edges as already shortest
                for v in V:
                    W, = np.where(G1[v, :])  # neighbors of smallest nodes
                    td = np.array(
                        [D[u, W].flatten(), (D[u, v] + G1[v, W]).flatten()])
                    D[u, W] = np.min(td, axis=0)

                if D[u, S].size == 0:  # all nodes reached
                    break
                minD = np.min(D[u, S])
                if np.isinf(minD):  # some nodes cannot be reached
                    break
                V, = np.where(D[u, :] == minD)

        np.fill_diagonal(D, 1)
        D = 1 / D
        np.fill_diagonal(D, 0)
        return D

    n = len(Gw)
    Gl = invert(Gw, copy=True)  # connection length matrix
    A = np.array((Gw != 0), dtype=int)
    if local:
        E = np.zeros((n,))  # local efficiency
        for u in range(n):
            # V,=np.where(Gw[u,:])		#neighbors
            # k=len(V)					#degree
            # if k>=2:					#degree must be at least 2
            #	e=(distance_inv_wei(Gl[V].T[V])*np.outer(Gw[V,u],Gw[u,V]))**1/3
            #	E[u]=np.sum(e)/(k*k-k)

            # find pairs of neighbors
            V, = np.where(np.logical_or(Gw[u, :], Gw[:, u].T))
            # symmetrized vector of weights
            sw = cuberoot(Gw[u, V]) + cuberoot(Gw[V, u].T)
            # inverse distance matrix
            e = distance_inv_wei(Gl[np.ix_(V, V)])
            # symmetrized inverse distance matrix
            se = cuberoot(e) + cuberoot(e.T)

            numer = np.sum(np.outer(sw.T, sw) * se) / 2
            if numer != 0:
                # symmetrized adjacency vector
                sa = A[u, V] + A[V, u].T
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                # print numer,denom
                E[u] = numer / denom  # local efficiency

    else:
        e = distance_inv_wei(Gl)
        E = np.sum(e) / (n * n - n)
    return E


def findpaths(CIJ, qmax, sources, savepths=False):
    '''
    Paths are sequences of linked nodes, that never visit a single node
    more than once. This function finds all paths that start at a set of
    source nodes, up to a specified length. Warning: very memory-intensive.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed/undirected connection matrix
    qmax : int
        maximal path length
    sources : Nx1 np.ndarray
        source units from which paths are grown
    savepths : bool
        True if all paths are to be collected and returned. This functionality
        is currently not enabled.

    Returns
    -------
    Pq : NxNxQ np.ndarray
        Path matrix with P[i,j,jq] = number of paths from i to j with length q
    tpath : int
        total number of paths found
    plq : Qx1 np.ndarray
        path length distribution as a function of q
    qstop : int
        path length at which findpaths is stopped
    allpths : None
        a matrix containing all paths up to qmax. This function is extremely
        complicated and reimplementing it in bctpy is not straightforward.
    util : NxQ np.ndarray
        node use index

    Notes
    -----
    Note that Pq(:,:,N) can only carry entries on the diagonal, as all
    "legal" paths of length N-1 must terminate.  Cycles of length N are
    possible, with all vertices visited exactly once (except for source and
    target). 'qmax = N' can wreak havoc (due to memory problems).

    Note: Weights are discarded.
    Note: I am certain that this algorithm is rather inefficient -
    suggestions for improvements are welcome.

    '''
    CIJ = binarize(CIJ, copy=True)  # ensure CIJ is binary
    n = len(CIJ)
    k = np.sum(CIJ)
    pths = []
    Pq = np.zeros((n, n, qmax))
    util = np.zeros((n, qmax))

    # this code is for pathlength=1
    # paths are seeded from sources
    q = 1
    for j in range(n):
        for i in range(len(sources)):
            i_s = sources[i]
            if CIJ[i_s, j] == 1:
                pths.append([i_s, j])
    pths = np.array(pths)

    # calculate the use index per vertex (for paths of length 1)
    util[:, q], _ = np.histogram(pths, bins=n)
    # now enter the found paths of length 1 into the pathmatrix Pq
    for nrp in range(np.size(pths, axis=0)):
        Pq[pths[nrp, 0], pths[nrp, q], q - 1] += 1

    # begin saving allpths
    if savepths:
        allpths = pths.copy()
    else:
        allpths = []

    npthscnt = k

    # big loop for all other pathlengths q
    for q in range(2, qmax + 1):
        # to keep track of time...
        print((
            'current pathlength (q=i, number of paths so far (up to q-1)=i' % (q, np.sum(Pq))))

        # old paths are now in 'pths'
        # new paths are about to be collected in 'npths'
        # estimate needed allocation for new paths
        len_npths = np.min((np.ceil(1.1 * npthscnt * k / n), 100000000))
        npths = np.zeros((q + 1, len_npths))

        # find the unique set of endpoints of 'pths'
        endp = np.unique(pths[:, q - 1])
        npthscnt = 0

        for i in endp:  # set of endpoints of previous paths
            # in 'pb' collect all previous paths with 'i' as their endpoint
            pb, = np.where(pths[:, q - 1] == i)
            # find the outgoing connections from i (breadth-first)
            nendp, = np.where(CIJ[i, :] == 1)
            # if i is not a dead end
            if nendp.size:
                for j in nendp:  # endpoints of next edge
                    # find new paths -- only legal ones, no vertex twice
                    # visited
                    pb_temp = pb[np.sum(j == pths[pb, 1:q], axis=1) == 0]

                    # add new paths to 'npths'
                    pbx = pths[pb_temp - 1, :]
                    npx = np.ones((len(pb_temp), 1)) * j
                    npths[:, npthscnt:npthscnt + len(pb_temp)] = np.append(
                        pbx, npx, axis=1).T
                    npthscnt += len(pb_temp)
                    # count new paths and add the number to P
                    Pq[:n, j, q -
                        1] += np.histogram(pths[pb_temp - 1, 0], bins=n)[0]

        # note: 'npths' now contains a list of all the paths of length q
        if len_npths > npthscnt:
            npths = npths[:, :npthscnt]

        # append the matrix of all paths
        # FIXME
        if savepths:
            raise NotImplementedError("Sorry allpaths is not yet implemented")

        # calculate the use index per vertex (correct for cycles, count
        # source/target only once)
        util[:, q - 1] += (np.histogram(npths[:, :npthscnt], bins=n)[0] -
                           np.diag(Pq[:, :, q - 1]))

        # elininate cycles from "making it" to the next level, so that "pths"
        # contains all the paths that have a chance of being continued
        if npths.size:
            pths = np.squeeze(npths[:, np.where(npths[0, :] != npths[q, :])]).T
        else:
            pths = []

        # if there are no 'pths' paths left, end the search
        if not pths.size:
            qstop = q
            tpath = np.sum(Pq)
            plq = np.sum(np.sum(Pq, axis=0), axis=0)
            return

    qstop = q
    tpath = np.sum(Pq)  # total number of paths
    plq = np.sum(np.sum(Pq, axis=0), axis=0)  # path length distribution

    return Pq, tpath, plq, qstop, allpths, util


def findwalks(CIJ):
    '''
    Walks are sequences of linked nodes, that may visit a single node more
    than once. This function finds the number of walks of a given length,
    between any two nodes.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed/undirected connection matrix

    Returns
    -------
    Wq : NxNxQ np.ndarray
        Wq[i,j,q] is the number of walks from i to j of length q
    twalk : int
        total number of walks found
    wlq : Qx1 np.ndarray
        walk length distribution as a function of q

    Notes
    -----
    Wq grows very quickly for larger N,K,q. Weights are discarded.
    '''
    CIJ = binarize(CIJ, copy=True)
    n = len(CIJ)
    Wq = np.zeros((n, n, n))
    CIJpwr = CIJ.copy()
    Wq[:, :, 1] = CIJ
    for q in range(n):
        CIJpwr = np.dot(CIJpwr, CIJ)
        Wq[:, :, q] = CIJpwr

    twalk = np.sum(Wq)  # total number of walks
    wlq = np.sum(np.sum(Wq, axis=0), axis=0)
    return Wq, twalk, wlq


def reachdist(CIJ, ensure_binary=True):
    '''
    The binary reachability matrix describes reachability between all pairs
    of nodes. An entry (u,v)=1 means that there exists a path from node u
    to node v; alternatively (u,v)=0.

    The distance matrix contains lengths of shortest paths between all
    pairs of nodes. An entry (u,v) represents the length of shortest path
    from node u to  node v. The average shortest path length is the
    characteristic path length of the network.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed/undirected connection matrix
    ensure_binary : bool
        Binarizes input. Defaults to true. No user who is not testing 
        something will ever want to not use this, use distance_wei instead for 
        unweighted matrices.

    Returns
    -------
    R : NxN np.ndarray
        binary reachability matrix
    D : NxN np.ndarray
        distance matrix

    Notes
    -----
    faster but more memory intensive than "breadthdist.m".
    '''
    def reachdist2(CIJ, CIJpwr, R, D, n, powr, col, row):
        CIJpwr = np.dot(CIJpwr, CIJ)
        R = np.logical_or(R, CIJpwr != 0)
        D += R

        if powr <= n and np.any(R[np.ix_(row, col)] == 0):
            powr += 1
            R, D, powr = reachdist2(CIJ, CIJpwr, R, D, n, powr, col, row)
        return R, D, powr

    if ensure_binary:
        CIJ = binarize(CIJ)

    R = CIJ.copy()
    D = CIJ.copy()
    powr = 2
    n = len(CIJ)
    CIJpwr = CIJ.copy()

    # check for vertices that have no incoming or outgoing connections
    # these are ignored by reachdist
    id = np.sum(CIJ, axis=0)
    od = np.sum(CIJ, axis=1)
    id0, = np.where(id == 0)  # nothing goes in, so column(R) will be 0
    od0, = np.where(od == 0)  # nothing comes out, so row(R) will be 0
    # use these colums and rows to check for reachability
    col = list(range(n))
    col = np.delete(col, id0)
    row = list(range(n))
    row = np.delete(row, od0)

    R, D, powr = reachdist2(CIJ, CIJpwr, R, D, n, powr, col, row)

    #'invert' CIJdist to get distances
    D = powr - D + 1

    # put inf if no path found
    D[D == n + 2] = np.inf
    D[:, id0] = np.inf
    D[od0, :] = np.inf

    return R, D
