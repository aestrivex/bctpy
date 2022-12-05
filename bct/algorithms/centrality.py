from __future__ import division, print_function
import numpy as np
from .core import kcore_bd, kcore_bu
from .distance import reachdist
from bct.utils import invert, BCTParamError
from ..due import due, BibTeX
from ..citations import (
    BRANDES2001, KINTALI2008, SHANNON1948, RUBINOV2011, NEWMAN2016, HONEY2007,
    HAGMANN2008, GUIMERA2005, MORRISON2005, BOLDI2009, ESTRADA2005, ESTRADA2010
)


@due.dcite(BibTeX(BRANDES2001), description="Unweighted betweenness centrality")
@due.dcite(BibTeX(KINTALI2008), description="Unweighted betweenness centrality")
def betweenness_bin(G):
    '''
    Node betweenness centrality is the fraction of all shortest paths in
    the network that contain a given node. Nodes with high values of
    betweenness centrality participate in a large number of shortest paths.

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed/undirected connection matrix

    BC : Nx1 np.ndarray
        node betweenness centrality vector

    Notes
    -----
    Betweenness centrality may be normalised to the range [0,1] as
    BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    '''
    G = np.array(G, dtype=float)  # force G to have float type so it can be
    # compared to float np.inf

    n = len(G)  # number of nodes
    I = np.eye(n)  # identity matrix
    d = 1  # path length
    NPd = G.copy()  # number of paths of length |d|
    NSPd = G.copy()  # number of shortest paths of length |d|
    NSP = G.copy()  # number of shortest paths of any length
    L = G.copy()  # length of shortest paths

    NSP[np.where(I)] = 1
    L[np.where(I)] = 1

    # calculate NSP and L
    while np.any(NSPd):
        d += 1
        NPd = np.dot(NPd, G)
        NSPd = NPd * (L == 0)
        NSP += NSPd
        L = L + d * (NSPd != 0)

    L[L == 0] = np.inf  # L for disconnected vertices is inf
    L[np.where(I)] = 0
    NSP[NSP == 0] = 1  # NSP for disconnected vertices is 1

    DP = np.zeros((n, n))  # vertex on vertex dependency
    diam = d - 1

    # calculate DP
    for d in range(diam, 1, -1):
        DPd1 = np.dot(((L == d) * (1 + DP) / NSP), G.T) * \
            ((L == (d - 1)) * NSP)
        DP += DPd1

    return np.sum(DP, axis=0)


@due.dcite(BibTeX(BRANDES2001), description="Weighted betweenness centrality")
@due.dcite(BibTeX(KINTALI2008), description="Weighted betweenness centrality")
def betweenness_wei(G):
    '''
    Node betweenness centrality is the fraction of all shortest paths in
    the network that contain a given node. Nodes with high values of
    betweenness centrality participate in a large number of shortest paths.

    Parameters
    ----------
    L : NxN np.ndarray
        directed/undirected weighted connection matrix

    Returns
    -------
    BC : Nx1 np.ndarray
        node betweenness centrality vector

    Notes
    -----
       The input matrix must be a connection-length matrix, typically
        obtained via a mapping from weight to length. For instance, in a
        weighted correlation network higher correlations are more naturally
        interpreted as shorter distances and the input matrix should
        consequently be some inverse of the connectivity matrix.
       Betweenness centrality may be normalised to the range [0,1] as
        BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    '''
    n = len(G)
    BC = np.zeros((n,))  # vertex betweenness

    for u in range(n):
        D = np.tile(np.inf, (n,))
        D[u] = 0  # distance from u
        NP = np.zeros((n,))
        NP[u] = 1  # number of paths from u
        S = np.ones((n,), dtype=bool)  # distance permanence
        P = np.zeros((n, n))  # predecessors
        Q = np.zeros((n,), dtype=int)  # indices
        q = n - 1  # order of non-increasing distance

        G1 = G.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                Q[q] = v
                q -= 1
                W, = np.where(G1[v, :])  # neighbors of v
                for w in W:
                    Duw = D[v] + G1[v, w]  # path length to be tested
                    if Duw < D[w]:  # if new u->w shorter than old
                        D[w] = Duw
                        NP[w] = NP[v]  # NP(u->w) = NP of new path
                        P[w, :] = 0
                        P[w, v] = 1  # v is the only predecessor
                    elif Duw == D[w]:  # if new u->w equal to old
                        NP[w] += NP[v]  # NP(u->w) sum of old and new
                        P[w, v] = 1  # v is also predecessor

            if D[S].size == 0:
                break  # all nodes were reached
            if np.isinf(np.min(D[S])):  # some nodes cannot be reached
                Q[:q + 1], = np.where(np.isinf(D))  # these are first in line
                break
            V, = np.where(D == np.min(D[S]))

        DP = np.zeros((n,))
        for w in Q[:n - 1]:
            BC[w] += DP[w]
            for v in np.where(P[w, :])[0]:
                DP[v] += (1 + DP[w]) * NP[v] / NP[w]

    return BC


@due.dcite(BibTeX(RUBINOV2011), description="Entropy-based diversity coefficient")
@due.dcite(BibTeX(SHANNON1948), description="Entropy-based diversity coefficient")
def diversity_coef_sign(W, ci):
    '''
    The Shannon-entropy based diversity coefficient measures the diversity
    of intermodular connections of individual nodes and ranges from 0 to 1.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected connection matrix with positive and negative weights
    ci : Nx1 np.ndarray
        community affiliation vector

    Returns
    -------
    Hpos : Nx1 np.ndarray
        diversity coefficient based on positive connections
    Hneg : Nx1 np.ndarray
        diversity coefficient based on negative connections
    '''
    n = len(W)  # number of nodes

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    m = np.max(ci)  # number of modules

    def entropy(w_):
        S = np.sum(w_, axis=1)  # strength
        Snm = np.zeros((n, m))  # node-to-module degree
        for i in range(m):
            Snm[:, i] = np.sum(w_[:, ci == i + 1], axis=1)
        pnm = Snm / (np.tile(S, (m, 1)).T)
        pnm[np.isnan(pnm)] = 0
        pnm[np.logical_not(pnm)] = 1
        return -np.sum(pnm * np.log(pnm), axis=1) / np.log(m)

    #explicitly ignore compiler warning for division by zero
    with np.errstate(invalid='ignore'):
        Hpos = entropy(W * (W > 0))
        Hneg = entropy(-W * (W < 0))

    return Hpos, Hneg


@due.dcite(BibTeX(BRANDES2001), description="Unweighted edge betweenness centrality")
def edge_betweenness_bin(G):
    '''
    Edge betweenness centrality is the fraction of all shortest paths in
    the network that contain a given edge. Edges with high values of
    betweenness centrality participate in a large number of shortest paths.

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed/undirected connection matrix

    Returns
    -------
    EBC : NxN np.ndarray
        edge betweenness centrality matrix
    BC : Nx1 np.ndarray
        node betweenness centrality vector

    Notes
    -----
    Betweenness centrality may be normalised to the range [0,1] as
    BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    '''
    n = len(G)
    BC = np.zeros((n,))  # vertex betweenness
    EBC = np.zeros((n, n))  # edge betweenness

    for u in range(n):
        D = np.zeros((n,))
        D[u] = 1  # distance from u
        NP = np.zeros((n,))
        NP[u] = 1  # number of paths from u
        P = np.zeros((n, n))  # predecessors
        Q = np.zeros((n,), dtype=int)  # indices
        q = n - 1  # order of non-increasing distance

        Gu = G.copy()
        V = np.array([u])
        while V.size:
            Gu[:, V] = 0  # remove remaining in-edges
            for v in V:
                Q[q] = v
                q -= 1
                W, = np.where(Gu[v, :])  # neighbors of V
                for w in W:
                    if D[w]:
                        NP[w] += NP[v]  # NP(u->w) sum of old and new
                        P[w, v] = 1  # v is a predecessor
                    else:
                        D[w] = 1
                        NP[w] = NP[v]  # NP(u->v) = NP of new path
                        P[w, v] = 1  # v is a predecessor
            V, = np.where(np.any(Gu[V, :], axis=0))

        if np.any(np.logical_not(D)):  # if some vertices unreachable
            Q[:q], = np.where(np.logical_not(D))  # ...these are first in line

        DP = np.zeros((n,))             # dependency
        for w in Q[:n - 1]:
            BC[w] += DP[w]
            for v in np.where(P[w, :])[0]:
                DPvw = (1 + DP[w]) * NP[v] / NP[w]
                DP[v] += DPvw
                EBC[v, w] += DPvw

    return EBC, BC


@due.dcite(BibTeX(BRANDES2001), description="Weighted edge betweenness centrality")
def edge_betweenness_wei(G):
    '''
    Edge betweenness centrality is the fraction of all shortest paths in
    the network that contain a given edge. Edges with high values of
    betweenness centrality participate in a large number of shortest paths.

    Parameters
    ----------
    L : NxN np.ndarray
        directed/undirected weighted connection matrix

    Returns
    -------
    EBC : NxN np.ndarray
        edge betweenness centrality matrix
    BC : Nx1 np.ndarray
        nodal betweenness centrality vector

    Notes
    -----
    The input matrix must be a connection-length matrix, typically
        obtained via a mapping from weight to length. For instance, in a
        weighted correlation network higher correlations are more naturally
        interpreted as shorter distances and the input matrix should
        consequently be some inverse of the connectivity matrix.
    Betweenness centrality may be normalised to the range [0,1] as
        BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    '''
    n = len(G)
    BC = np.zeros((n,))  # vertex betweenness
    EBC = np.zeros((n, n))  # edge betweenness

    for u in range(n):
        D = np.tile(np.inf, n)
        D[u] = 0  # distance from u
        NP = np.zeros((n,))
        NP[u] = 1  # number of paths from u
        S = np.ones((n,), dtype=bool)  # distance permanence
        P = np.zeros((n, n))  # predecessors
        Q = np.zeros((n,), dtype=int)  # indices
        q = n - 1  # order of non-increasing distance

        G1 = G.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                Q[q] = v
                q -= 1
                W, = np.where(G1[v, :])  # neighbors of v
                for w in W:
                    Duw = D[v] + G1[v, w]  # path length to be tested
                    if Duw < D[w]:  # if new u->w shorter than old
                        D[w] = Duw
                        NP[w] = NP[v]  # NP(u->w) = NP of new path
                        P[w, :] = 0
                        P[w, v] = 1  # v is the only predecessor
                    elif Duw == D[w]:  # if new u->w equal to old
                        NP[w] += NP[v]  # NP(u->w) sum of old and new
                        P[w, v] = 1  # v is also a predecessor

            if D[S].size == 0:
                break  # all nodes reached, or
            if np.isinf(np.min(D[S])):  # some cannot be reached
                Q[:q + 1], = np.where(np.isinf(D)) # these are first in line.
                break
            V, = np.where(D == np.min(D[S]))

        DP = np.zeros((n,))  # dependency
        for w in Q[:n - 1]:
            BC[w] += DP[w]
            for v in np.where(P[w, :])[0]:
                DPvw = (1 + DP[w]) * NP[v] / NP[w]
                DP[v] += DPvw
                EBC[v, w] += DPvw

    return EBC, BC


@due.dcite(BibTeX(NEWMAN2016), description="Eigenvector centrality")
def eigenvector_centrality_und(CIJ):
    '''
    Eigenector centrality is a self-referential measure of centrality:
    nodes have high eigenvector centrality if they connect to other nodes
    that have high eigenvector centrality. The eigenvector centrality of
    node i is equivalent to the ith element in the eigenvector
    corresponding to the largest eigenvalue of the adjacency matrix.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary/weighted undirected adjacency matrix

    v : Nx1 np.ndarray
        eigenvector associated with the largest eigenvalue of the matrix
    '''
    from scipy import linalg

    n = len(CIJ)
    vals, vecs = linalg.eig(CIJ)
    i = np.argmax(vals)
    return np.abs(vecs[:, i])


def erange(CIJ):
    '''
    Shortcuts are central edges which significantly reduce the
    characteristic path length in the network.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    Erange : NxN np.ndarray
        range for each edge, i.e. the length of the shortest path from i to j
        for edge c(i,j) after the edge has been removed from the graph
    eta : float
        average range for the entire graph
    Eshort : NxN np.ndarray
        entries are ones for shortcut edges
    fs : float
        fractions of shortcuts in the graph

    Follows the treatment of 'shortcuts' by Duncan Watts
    '''
    N = len(CIJ)
    K = np.size(np.where(CIJ)[1])
    Erange = np.zeros((N, N))
    i, j = np.where(CIJ)

    for c in range(len(i)):
        CIJcut = CIJ.copy()
        CIJcut[i[c], j[c]] = 0
        R, D = reachdist(CIJcut)
        Erange[i[c], j[c]] = D[i[c], j[c]]

    # average range (ignore Inf)
    eta = (np.sum(Erange[np.logical_and(Erange > 0, Erange < np.inf)]) /
           len(Erange[np.logical_and(Erange > 0, Erange < np.inf)]))

    # Original entries of D are ones, thus entries of Erange
    # must be two or greater.
    # If Erange(i,j) > 2, then the edge is a shortcut.
    # 'fshort' is the fraction of shortcuts over the entire graph.

    Eshort = Erange > 2
    fs = len(np.where(Eshort)) / K

    return Erange, eta, Eshort, fs


@due.dcite(BibTeX(HONEY2007), description="Flow coefficient")
def flow_coef_bd(CIJ):
    '''
    Computes the flow coefficient for each node and averaged over the
    network, as described in Honey et al. (2007) PNAS. The flow coefficient
    is similar to betweenness centrality, but works on a local
    neighborhood. It is mathematically related to the clustering
    coefficient  (cc) at each node as, fc+cc <= 1.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    fc : Nx1 np.ndarray
        flow coefficient for each node
    FC : float
        average flow coefficient over the network
    total_flo : int
        number of paths that "flow" across the central node
    '''
    N = len(CIJ)

    fc = np.zeros((N,))
    total_flo = np.zeros((N,))
    max_flo = np.zeros((N,))

    # loop over nodes
    for v in range(N):
        # find neighbors - note: both incoming and outgoing connections
        nb, = np.where(CIJ[v, :] + CIJ[:, v].T)
        fc[v] = 0
        if np.where(nb)[0].size:
            CIJflo = -CIJ[np.ix_(nb, nb)]
            for i in range(len(nb)):
                for j in range(len(nb)):
                    if CIJ[nb[i], v] and CIJ[v, nb[j]]:
                        CIJflo[i, j] += 1
            total_flo[v] = np.sum(
                (CIJflo == 1) * np.logical_not(np.eye(len(nb))))
            max_flo[v] = len(nb) * len(nb) - len(nb)
            fc[v] = total_flo[v] / max_flo[v]

    fc[np.isnan(fc)] = 0
    FC = np.mean(fc)

    return fc, FC, total_flo


def gateway_coef_sign(W, ci, centrality_type='degree'):
    '''
    The gateway coefficient is a variant of participation coefficient.
    It is weighted by how critical the connections are to intermodular
    connectivity (e.g. if a node is the only connection between its
    module and another module, it will have a higher gateway coefficient,
    unlike participation coefficient).

    Parameters
    ----------
    W : NxN np.ndarray
        undirected signed connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector
    centrality_type : enum
        'degree' - uses the weighted degree (i.e, node strength)
        'betweenness' - uses the betweenness centrality

    Returns
    -------
    Gpos : N x nr_mod np.ndarray
        gateway coefficient for positive weights
    Gneg : N x nr_mod np.ndarray
        gateway coefficient for negative weights

    Reference:
        Vargas ER, Wahl LM, Eur Phys J B (2014) 87:1-10
    '''
    if centrality_type not in ('degree', 'betweenness'):
        raise BCTParamError("centrality_type must be either 'degree' "
            "or 'betweenness' ")
    
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    n = len(W)
    np.fill_diagonal(W, 0)

    def gcoef(W):
        nr_modules = int(np.max(ci))

        #strength
        s = np.sum(W, axis=1)
        #neighbor community affiliation
        Gc = np.inner((W != 0), np.diag(ci))
        #weighting of connections of node to each module
        ks = np.zeros((n, nr_modules))
        #module to module connectivities
        kjs = np.zeros((n, nr_modules))
        #sum of centralities of neighbors within a module
        cs = np.zeros((n, nr_modules))

        if centrality_type == 'degree':
            cent = s.copy()
        elif centrality_type == 'betweenness':
            cent = betweenness_wei(invert(W))


        max_centrality = 0
        for i in range(nr_modules):
            #compute total weight of connections from each node to module
            ks[:,i] = np.sum(W * (Gc == i+1), axis=1)

            #find module with max centrality value
            centrality = np.sum(cent[ci == i+1])
            if centrality > max_centrality:
                max_centrality = centrality

        for i in range(nr_modules):
            #if more than 1 node in the module
            if np.sum(ci == i+1) > 1:         
                kj = np.ones((np.sum(ci == i+1), 1)) * np.sum(ks[ci == i+1, :])
                kj[i] /= 2
                
                kjs[ci == i+1, :] = kj
                        
        for i in range(n):
            #if node is connected
            if s[i] > 0:
                for j in range(nr_modules):
                    #sum of centralities of neighbors of node within a module
                    in_mod_nodes, = np.where(ci == j+1)
                    neighbs, = np.where(W[in_mod_nodes, i] > 0)
                    cs[i, j] = np.sum(cent[neighbs])

        #normalize by total connections
        ksm = ks / kjs

        #account for division by 0
        ksm[np.where(kjs == 0)] = 0

        #normalize by max centrality
        centm = cs / max_centrality
    
        gs = (1 - ksm * centm) ** 2
        sm = np.transpose(np.tile(s, (nr_modules, 1)))
        Gw = 1 - np.sum((ks ** 2) / (sm ** 2) * gs, axis=1)

        #account for divison by 0 and set nodes with no neighbors to 0
        Gw[np.where(np.isnan(Gw))] = 0
        Gw[np.where(np.logical_not(Gw))] = 0

        return Gw

    G_pos = gcoef(W * (W > 0))
    G_neg = gcoef(-W * (W < 0))
    return G_pos, G_neg


@due.dcite(BibTeX(HAGMANN2008), description="Unweighted directed k-coreness centrality")
def kcoreness_centrality_bd(CIJ):
    '''
    The k-core is the largest subgraph comprising nodes of degree at least
    k. The coreness of a node is k if the node belongs to the k-core but
    not to the (k+1)-core. This function computes k-coreness of all nodes
    for a given binary directed connection matrix.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    coreness : Nx1 np.ndarray
        node coreness
    kn : Nx1 np.ndarray
        size of k-core for all values of K between 1 and N
    '''
    N = len(CIJ)

    coreness = np.zeros((N,))
    kn = np.zeros((N,))

    for k in range(N):
        CIJkcore, kn[k] = kcore_bd(CIJ, k)
        ss = np.sum(CIJkcore, axis=0) > 0
        coreness[ss] = k

    return coreness, kn


@due.dcite(BibTeX(HAGMANN2008), description="Unweighted undirected k-coreness centrality")
def kcoreness_centrality_bu(CIJ):
    '''
    The k-core is the largest subgraph comprising nodes of degree at least
    k. The coreness of a node is k if the node belongs to the k-core but
    not to the (k+1)-core. This function computes the coreness of all nodes
    for a given binary undirected connection matrix.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary undirected connection matrix

    Returns
    -------
    coreness : Nx1 np.ndarray
        node coreness
    kn : Nx1 np.ndarray
        size of k-core for all values of K between 1 and N
    '''
    N = len(CIJ)

    # determine if the network is undirected -- if not, compute coreness
    # on the corresponding undirected network
    CIJund = CIJ + CIJ.T
    if np.any(CIJund > 1):
        CIJ = np.array(CIJund > 0, dtype=float)

    coreness = np.zeros((N,))
    kn = np.zeros((N,))
    for k in range(N):
        CIJkcore, kn[k] = kcore_bu(CIJ, k)
        ss = np.sum(CIJkcore, axis=0) > 0
        coreness[ss] = k

    return coreness, kn


@due.dcite(BibTeX(GUIMERA2005), description="Within-module degree z-score")
def module_degree_zscore(W, ci, flag=0):
    '''
    The within-module degree z-score is a within-module version of degree
    centrality.

    Parameters
    ----------
    W : NxN np.narray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.array_like
        community affiliation vector
    flag : int
        Graph type. 0: undirected graph (default)
                    1: directed graph in degree
                    2: directed graph out degree
                    3: directed graph in and out degree

    Returns
    -------
    Z : Nx1 np.ndarray
        within-module degree Z-score
    '''
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    if flag == 2:
        W = W.copy()
        W = W.T
    elif flag == 3:
        W = W.copy()
        W = W + W.T

    n = len(W)
    Z = np.zeros((n,))  # number of vertices
    for i in range(1, int(np.max(ci) + 1)):
        Koi = np.sum(W[np.ix_(ci == i, ci == i)], axis=1)
        Z[np.where(ci == i)] = (Koi - np.mean(Koi)) / np.std(Koi)

    Z[np.where(np.isnan(Z))] = 0
    return Z


@due.dcite(BibTeX(MORRISON2005), description="PageRank centrality")
@due.dcite(BibTeX(BOLDI2009), description="PageRank centrality")
def pagerank_centrality(A, d, falff=None):
    '''
    The PageRank centrality is a variant of eigenvector centrality. This
    function computes the PageRank centrality of each vertex in a graph.

    Formally, PageRank is defined as the stationary distribution achieved
    by instantiating a Markov chain on a graph. The PageRank centrality of
    a given vertex, then, is proportional to the number of steps (or amount
    of time) spent at that vertex as a result of such a process.

    The PageRank index gets modified by the addition of a damping factor,
    d. In terms of a Markov chain, the damping factor specifies the
    fraction of the time that a random walker will transition to one of its
    current state's neighbors. The remaining fraction of the time the
    walker is restarted at a random vertex. A common value for the damping
    factor is d = 0.85.

    Parameters
    ----------
    A : NxN np.narray
        adjacency matrix
    d : float
        damping factor (see description)
    falff : Nx1 np.ndarray | None
        Initial page rank probability, non-negative values. Default value is
        None. If not specified, a naive bayesian prior is used.

    Returns
    -------
    r : Nx1 np.ndarray
        vectors of page rankings

    Notes
    -----
    Note: The algorithm will work well for smaller matrices (number of
    nodes around 1000 or less)
    '''
    from scipy import linalg

    N = len(A)
    if falff is None:
        norm_falff = np.ones((N,)) / N
    else:
        norm_falff = falff / np.sum(falff)

    deg = np.sum(A, axis=0)
    deg[deg == 0] = 1
    D1 = np.diag(1 / deg)
    B = np.eye(N) - d * np.dot(A, D1)
    b = (1 - d) * norm_falff
    r = linalg.solve(B, b)
    r /= np.sum(r)
    return r


@due.dcite(BibTeX(GUIMERA2005), description="Participation coefficient")
def participation_coef(W, ci, degree='undirected'):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.

    Parameters
    ----------
    W : NxN np.ndarray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector
    degree : str
        Flag to describe nature of graph 'undirected': For undirected graphs
                                         'in': Uses the in-degree
                                         'out': Uses the out-degree

    Returns
    -------
    P : Nx1 np.ndarray
        participation coefficient
    '''
    if degree == 'in':
        W = W.T

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
    Kc2 = np.zeros((n,))  # community-specific neighbors

    for i in range(1, int(np.max(ci)) + 1):
        Kc2 = Kc2 + np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return P

def participation_coef_sparse(W, ci, degree='undirected'):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.
    Parameters
    ----------
    W : NxN np.ndarray
        binary/weighted directed/undirected connection
        must be as scipy.sparse.csr matrix
    ci : Nx1 np.ndarray
        community affiliation vector
    degree : str
        Flag to describe nature of graph 'undirected': For undirected graphs
                                         'in': Uses the in-degree
                                         'out': Uses the out-degree
    Returns
    -------
    P : Nx1 np.ndarray
        participation coefficient
    '''
    if degree == 'in':
        W = W.T

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = W.shape[0]  # number of vertices
    Ko = np.array(W.sum(axis=1)).flatten().astype(float)  # (out) degree
    Gc = W.copy().astype('int16')
    Gc[Gc!=0] = 1 
    Gc = Gc * np.diag(ci)# neighbor community affiliation
    
    P = np.zeros((n))
    for i in range(1, int(np.max(ci)) + 1):
        P = P + (np.array((W.multiply(Gc == i).astype(int)).sum(axis=1)).flatten() / Ko)**2
    P = 1 - P
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return P

def participation_coef_sign(W, ci):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected connection matrix with positive and negative weights
    ci : Nx1 np.ndarray
        community affiliation vector

    Returns
    -------
    Ppos : Nx1 np.ndarray
        participation coefficient from positive weights
    Pneg : Nx1 np.ndarray
        participation coefficient from negative weights
    '''
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices

    def pcoef(W_):
        S = np.sum(W_, axis=1)  # strength
        # neighbor community affil.
        Gc = np.dot(np.logical_not(W_ == 0), np.diag(ci))
        Sc2 = np.zeros((n,))

        for i in range(1, int(np.max(ci) + 1)):
            Sc2 += np.square(np.sum(W_ * (Gc == i), axis=1))

        P = np.ones((n,)) - Sc2 / np.square(S)
        P[np.where(np.isnan(P))] = 0
        P[np.where(np.logical_not(P))] = 0  # p_ind=0 if no (out)neighbors
        return P

    #explicitly ignore compiler warning for division by zero
    with np.errstate(invalid='ignore'):
        Ppos = pcoef(W * (W > 0))
        Pneg = pcoef(-W * (W < 0))

    return Ppos, Pneg


@due.dcite(BibTeX(ESTRADA2005), description="Subgraph centrality")
@due.dcite(BibTeX(ESTRADA2010), description="Subgraph centrality")
def subgraph_centrality(CIJ):
    '''
    The subgraph centrality of a node is a weighted sum of closed walks of
    different lengths in the network starting and ending at the node. This
    function returns a vector of subgraph centralities for each node of the
    network.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary adjacency matrix

    Cs : Nx1 np.ndarray
        subgraph centrality
    '''
    from scipy import linalg

    vals, vecs = linalg.eig(CIJ)  # compute eigendecomposition
    # lambdas=np.diag(vals)
    # compute eigenvector centr.
    Cs = np.real(np.dot(vecs * vecs, np.exp(vals)))
    return Cs  # imaginary part from precision error
