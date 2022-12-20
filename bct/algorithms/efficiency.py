from __future__ import division, print_function
import numpy as np
from bct.utils import cuberoot, BCTParamError, binarize, invert
from .distance import distance_wei_floyd, mean_first_passage_time
from ..due import due, BibTeX
from ..citations import LATORA2001, ONNELA2005, FAGIOLO2007, RUBINOV2010

@due.dcite(BibTeX(LATORA2001), description="Unweighted global efficiency")
@due.dcite(BibTeX(ONNELA2005), description="Unweighted global efficiency")
@due.dcite(BibTeX(FAGIOLO2007), description="Unweighted global efficiency")
@due.dcite(BibTeX(RUBINOV2010), description="Unweighted global efficiency")
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


@due.dcite(BibTeX(LATORA2001), description="Weighted global efficiency")
@due.dcite(BibTeX(ONNELA2005), description="Weighted global efficiency")
@due.dcite(BibTeX(FAGIOLO2007), description="Weighted global efficiency")
@due.dcite(BibTeX(RUBINOV2010), description="Weighted global efficiency")
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
    local = bool or enum
        If True or 'local', computes local efficiency instead of global efficiency.
        If False or 'global', uses the global efficiency
        If 'original', will use the original algorithm provided by (Rubinov
        & Sporns 2010). This version is not recommended. The local efficiency
        calculation was improved in (Wang et al. 2016) as a true generalization
        of the binary variant.
        
    Returns
    -------
    Eglob : float
        global efficiency, only if local in (False, 'global')
    Eloc : Nx1 np.ndarray
        local efficiency, only if local in (True, 'local', 'original')

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
    if local not in (True, False, 'local', 'global', 'original'):
        raise BCTParamError("local param must be any of True, False, "
            "'local', 'global', or 'original'")

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
    #local efficiency algorithm described by Rubinov and Sporns 2010, not recommended
    if local == 'original':
        E = np.zeros((n,))
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

    #local efficiency algorithm described by Wang et al 2016, recommended
    elif local in (True, 'local'):
        E = np.zeros((n,))
        for u in range(n):
            V, = np.where(np.logical_or(Gw[u, :], Gw[:, u].T))
            sw = cuberoot(Gw[u, V]) + cuberoot(Gw[V, u].T)
            e = distance_inv_wei(cuberoot(Gl)[np.ix_(V, V)])
            se = e+e.T
         
            numer = np.sum(np.outer(sw.T, sw) * se) / 2
            if numer != 0:
                # symmetrized adjacency vector
                sa = A[u, V] + A[V, u].T
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                # print numer,denom
                E[u] = numer / denom  # local efficiency

    elif local in (False, 'global'):
        e = distance_inv_wei(Gl)
        E = np.sum(e) / (n * n - n)
    return E


def diffusion_efficiency(adj):
    '''
    The diffusion efficiency between nodes i and j is the inverse of the
    mean first passage time from i to j, that is the expected number of
    steps it takes a random walker starting at node i to arrive for the
    first time at node j. Note that the mean first passage time is not a
    symmetric measure -- mfpt(i,j) may be different from mfpt(j,i) -- and
    the pair-wise diffusion efficiency matrix is hence also not symmetric.

    Parameters
    ----------
    adj : np.ndarray
        weighted/unweighted, directed/undirected adjacency matrix

    Returns
    -------
    gediff : float
        mean global diffusion efficiency
    ediff : np.ndarray
        pairwise NxN diffusion efficiency matrix
    '''
    n = len(adj)
    adj = adj.copy()
    mfpt = mean_first_passage_time(adj)
    with np.errstate(divide='ignore'):
        ediff = 1 / mfpt
    np.fill_diagonal(ediff, 0)
    gediff = np.sum(ediff) / (n ** 2 - n)
    return gediff, ediff


def resource_efficiency_bin(adj, lamb, spl=None, m=None):
    '''
    The resource efficiency between nodes i and j is inversly proportional
    to the amount of resources (i.e. number of particles or messages)
    required to ensure with probability 0 < lambda < 1 that at least one of
    them will arrive at node j in exactly SPL steps, where SPL is the
    length of the shortest-path between i and j.
 
    The shortest-path probability between nodes i and j is the probability
    that a single random walker starting at node i will arrive at node j by
    following (one of) the shortest path(s).

    Parameters 
    ----------
    adj : NxN np.ndarray
        Unweighted, undirected adjacency matrix
    lamb : float
        Probability parameter of finding a path, set to np.nan if computation
        of resource efficiency matrix is not desired.
        Thanks Guido for making calling a variable lambda a syntax error in 
        python while still not providing inline syntax for a fully first class
        function.
    spl : NxN np.ndarray
        Shortest path length matrix (optional)
    m : NxN np.ndarray
        Transition probability matrix (optional)

    Returns
    -------
    Eres : NxN np.ndarray
        resource efficiency matrix. If lambda was provided as NaN, then Eres
        will be np.nan
    prob_spl : NxN np.ndarray
        probabilistic shortest path matrix

    Note that global measures for the resource efficiency and probabilistic
    shortest paths exist and are well defined, they are the mean values of the
    off diagonal elements
    GEres = mean(Eres(~eye(N) > 0))
    Gprob_SPL = mean(prob_SPL(~eye(N) > 0))
    '''
    n = len(adj)

    if not np.isnan(lamb):
        if lamb <= 0 or lamb >= 1:
            raise BCTParamError("Lambda must be a nonzero probability")

        z = np.zeros((n, n))

    def prob_first_particle_arrival(m, l, hvec, lamb):
        prob = np.zeros((n, n))
        
        if not np.isnan(lamb):
            resources = np.zeros((n, n))    
        else:
            resources = None

        # for each destination node h
        for h in hvec:
            B_h = m.copy()
            B_h[h, :] = 0
            B_h[h, h] = 1  
            # h becomes absorbant state

            B_h_L = B_h ** l

            term = 1 - B_h_L[:, h]

            prob[:, h] = 1-term

            if not np.isnan(lamb):
                with np.errstate(divide='ignore'):
                    resources[:, h] = (np.repeat( np.log(1 - lamb), n, 0) / 
                                       np.log(term))

        return prob, resources

    if spl is None:
        spl, _, _ = distance_wei_floyd(adj)

    if m is None:
        m = np.linalg.solve( np.diag( np.sum(adj, axis=1) ), adj)

    lvalues = np.unique(spl)
    lvalues = lvalues[np.where(lvalues != 0)]

    #a priori zero probability of going through SPL among nodes
    prob_spl = np.zeros((n, n))
    
    for splvalue in lvalues:
        hcols = np.where(spl == splvalue)
        hvec = np.unique(hcols)
        entries = spl == splvalue

        prob_aux, z_aux = prob_first_particle_arrival(m, splvalue, hvec, lamb)

        prob_aux[entries == 0] = 0
        prob_spl += prob_aux

        if not np.isnan(lamb):
            z_aux[entries == 0] = 0
            z += z_aux

    np.fill_diagonal(prob_spl, 0)

    if not np.isnan(lamb):
        z[prob_spl == 1] = 1
        with np.errstate(divide='ignore'):
            Eres = 1 / z
        np.fill_diagonal(Eres, 0)
    else:
        Eres = np.nan

    return Eres, prob_spl

def rout_efficiency(D, transform=None):
    '''
    The routing efficiency is the average of inverse shortest path length.
 
    The local routing efficiency of a node u is the routing efficiency
    computed on the subgraph formed by the neighborhood of node u
    (excluding node u).

    Parameters
    ----------
    D : NxN np.ndarray
        Weighted/unweighted directed/undirected connection weight or length
        matrix
    transform : str or None, optional
        If `adjacency` is a connection weight array, specify a transform to map
        input connection weights to connection lengths. Options include ['log',
        'inv'], where 'log' is `-np.log(adjacency)` and 'inv' is `1/adjacency`.
        Default: None

    Returns
    -------
    GErout : float
        Mean global routing efficiency
    Erout : NxN np.ndarray
        Pairwise routing efficiency matrix
    Eloc : Nx1 np.ndarray
        Local efficiency vector
    '''
    n = len(D)
    Erout, _, _ = distance_wei_floyd(D, transform=transform)
    with np.errstate(divide='ignore'):
        Erout = 1 / Erout
    np.fill_diagonal(Erout, 0)
    GErout = (np.sum(Erout[np.where(np.logical_not(np.isnan(Erout)))]) / 
              (n ** 2 - n))

    Eloc = np.zeros((n,))
    for u in range(n):
        Gu, = np.where(np.logical_or(D[u, :], D[:, u].T))
        nGu = len(Gu)
        e, _, _ = distance_wei_floyd(D[Gu, :][:, Gu], transform=transform)
        with np.errstate(divide='ignore'):
            e = 1 / e
        np.fill_diagonal(e, 0)
        Eloc[u] = np.sum(e) / nGu

    return GErout, Erout, Eloc

    
