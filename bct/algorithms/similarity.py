from __future__ import division, print_function
import numpy as np
from bct.utils import BCTParamError, binarize
from .degree import degrees_dir, degrees_und


def edge_nei_overlap_bd(CIJ):
    '''
    This function determines the neighbors of two nodes that are linked by
    an edge, and then computes their overlap.  Connection matrix must be
    binary and directed.  Entries of 'EC' that are 'inf' indicate that no
    edge is present.  Entries of 'EC' that are 0 denote "local bridges",
    i.e. edges that link completely non-overlapping neighborhoods.  Low
    values of EC indicate edges that are "weak ties".

    If CIJ is weighted, the weights are ignored. Neighbors of a node can be
    linked by incoming, outgoing, or reciprocal connections.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        directed binary/weighted connection matrix

    Returns
    -------
    EC : NxN np.ndarray
        edge neighborhood overlap matrix
    ec : Kx1 np.ndarray
        edge neighborhood overlap per edge vector
    degij : NxN np.ndarray
        degrees of node pairs connected by each edge
    '''

    ik, jk = np.where(CIJ)
    lel = len(CIJ[ik, jk])
    n = len(CIJ)

    _, _, deg = degrees_dir(CIJ)

    ec = np.zeros((lel,))
    degij = np.zeros((2, lel))
    for e in range(lel):
        neiik = np.setdiff1d(np.union1d(
            np.where(CIJ[ik[e], :]), np.where(CIJ[:, ik[e]])), (ik[e], jk[e]))
        neijk = np.setdiff1d(np.union1d(
            np.where(CIJ[jk[e], :]), np.where(CIJ[:, jk[e]])), (ik[e], jk[e]))
        ec[e] = len(np.intersect1d(neiik, neijk)) / \
            len(np.union1d(neiik, neijk))
        degij[:, e] = (deg[ik[e]], deg[jk[e]])

    EC = np.tile(np.inf, (n, n))
    EC[ik, jk] = ec
    return EC, ec, degij


def edge_nei_overlap_bu(CIJ):
    '''
    This function determines the neighbors of two nodes that are linked by
    an edge, and then computes their overlap.  Connection matrix must be
    binary and directed.  Entries of 'EC' that are 'inf' indicate that no
    edge is present.  Entries of 'EC' that are 0 denote "local bridges", i.e.
    edges that link completely non-overlapping neighborhoods.  Low values
    of EC indicate edges that are "weak ties".

    If CIJ is weighted, the weights are ignored.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        undirected binary/weighted connection matrix

    Returns
    -------
    EC : NxN np.ndarray
        edge neighborhood overlap matrix
    ec : Kx1 np.ndarray
        edge neighborhood overlap per edge vector
    degij : NxN np.ndarray
        degrees of node pairs connected by each edge
    '''
    ik, jk = np.where(CIJ)
    lel = len(CIJ[ik, jk])
    n = len(CIJ)

    deg = degrees_und(CIJ)

    ec = np.zeros((lel,))
    degij = np.zeros((2, lel))
    for e in range(lel):
        neiik = np.setdiff1d(np.union1d(
            np.where(CIJ[ik[e], :]), np.where(CIJ[:, ik[e]])), (ik[e], jk[e]))
        neijk = np.setdiff1d(np.union1d(
            np.where(CIJ[jk[e], :]), np.where(CIJ[:, jk[e]])), (ik[e], jk[e]))
        ec[e] = len(np.intersect1d(neiik, neijk)) / \
            len(np.union1d(neiik, neijk))
        degij[:, e] = (deg[ik[e]], deg[jk[e]])

    EC = np.tile(np.inf, (n, n))
    EC[ik, jk] = ec
    return EC, ec, degij


def gtom(adj, nr_steps):
    '''
    The m-th step generalized topological overlap measure (GTOM) quantifies
    the extent to which a pair of nodes have similar m-th step neighbors.
    Mth-step neighbors are nodes that are reachable by a path of at most
    length m.

    This function computes the the M x M generalized topological overlap
    measure (GTOM) matrix for number of steps, numSteps.

    Parameters
    ----------
    adj : NxN np.ndarray
        connection matrix
    nr_steps : int
        number of steps

    Returns
    -------
    gt : NxN np.ndarray
        GTOM matrix

    Notes
    -----
    When numSteps is equal to 1, GTOM is identical to the topological
    overlap measure (TOM) from reference [2]. In that case the 'gt' matrix
    records, for each pair of nodes, the fraction of neighbors the two
    nodes share in common, where "neighbors" are one step removed. As
    'numSteps' is increased, neighbors that are furter out are considered.
    Elements of 'gt' are bounded between 0 and 1.  The 'gt' matrix can be
    converted from a similarity to a distance matrix by taking 1-gt.
    '''
    bm = binarize(adj, copy=True)
    bm_aux = bm.copy()
    nr_nodes = len(adj)

    if nr_steps > nr_nodes:
        print("Warning: nr_steps exceeded nr_nodes. Setting nr_steps=nr_nodes")
    if nr_steps == 0:
        return bm
    else:
        for steps in range(2, nr_steps):
            for i in range(nr_nodes):
                # neighbors of node i
                ng_col, = np.where(bm_aux[i, :] == 1)
                # neighbors of neighbors of node i
                nng_row, nng_col = np.where(bm_aux[ng_col, :] == 1)
                new_ng = np.setdiff1d(nng_col, (i,))

                # neighbors of neighbors of i become considered neighbors of i
                bm_aux[i, new_ng] = 1
                bm_aux[new_ng, i] = 1

        # numerator of GTOM formula
        numerator_mat = np.dot(bm_aux, bm_aux) + bm + np.eye(nr_nodes)

        # vector of node degrees
        bms = np.sum(bm_aux, axis=0)
        bms_r = np.tile(bms, (nr_nodes, 1))

        denominator_mat = -bm + np.where(bms_r > bms_r.T, bms_r, bms_r.T) + 1
        return numerator_mat / denominator_mat


def matching_ind(CIJ):
    '''
    For any two nodes u and v, the matching index computes the amount of
    overlap in the connection patterns of u and v. Self-connections and
    u-v connections are ignored. The matching index is a symmetric
    quantity, similar to a correlation or a dot product.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        adjacency matrix

    Returns
    -------
    Min : NxN np.ndarray
        matching index for incoming connections
    Mout : NxN np.ndarray
        matching index for outgoing connections
    Mall : NxN np.ndarray
        matching index for all connections

    Notes
    -----
    Does not use self- or cross connections for comparison.
    Does not use connections that are not present in BOTH u and v.
    All output matrices are calculated for upper triangular only.
    '''
    n = len(CIJ)

    Min = np.zeros((n, n))
    Mout = np.zeros((n, n))
    Mall = np.zeros((n, n))

    # compare incoming connections
    for i in range(n - 1):
        for j in range(i + 1, n):
            c1i = CIJ[:, i]
            c2i = CIJ[:, j]
            usei = np.logical_or(c1i, c2i)
            usei[i] = 0
            usei[j] = 0
            nconi = np.sum(c1i[usei]) + np.sum(c2i[usei])
            if not nconi:
                Min[i, j] = 0
            else:
                Min[i, j] = 2 * \
                    np.sum(np.logical_and(c1i[usei], c2i[usei])) / nconi

            c1o = CIJ[i, :]
            c2o = CIJ[j, :]
            useo = np.logical_or(c1o, c2o)
            useo[i] = 0
            useo[j] = 0
            ncono = np.sum(c1o[useo]) + np.sum(c2o[useo])
            if not ncono:
                Mout[i, j] = 0
            else:
                Mout[i, j] = 2 * \
                    np.sum(np.logical_and(c1o[useo], c2o[useo])) / ncono

            c1a = np.ravel((c1i, c1o))
            c2a = np.ravel((c2i, c2o))
            usea = np.logical_or(c1a, c2a)
            usea[i] = 0
            usea[i + n] = 0
            usea[j] = 0
            usea[j + n] = 0
            ncona = np.sum(c1a[usea]) + np.sum(c2a[usea])
            if not ncona:
                Mall[i, j] = 0
            else:
                Mall[i, j] = 2 * \
                    np.sum(np.logical_and(c1a[usea], c2a[usea])) / ncona

    Min = Min + Min.T
    Mout = Mout + Mout.T
    Mall = Mall + Mall.T

    return Min, Mout, Mall


def matching_ind_und(CIJ0):
    '''
    M0 = MATCHING_IND_UND(CIJ) computes matching index for undirected
    graph specified by adjacency matrix CIJ. Matching index is a measure of
    similarity between two nodes' connectivity profiles (excluding their
    mutual connection, should it exist).

    Parameters
    ----------
    CIJ : NxN np.ndarray
        undirected adjacency matrix

    Returns
    -------
    M0 : NxN np.ndarray
        matching index matrix
    '''
    K = np.sum(CIJ0, axis=0)
    n = len(CIJ0)
    R = (K != 0)
    N = np.sum(R)
    xR, = np.where(R == 0)
    CIJ = np.delete(np.delete(CIJ0, xR, axis=0), xR, axis=1)
    I = np.logical_not(np.eye(N))
    M = np.zeros((N, N))

    for i in range(N):
        c1 = CIJ[i, :]
        use = np.logical_or(c1, CIJ)
        use[:, i] = 0
        use *= I

        ncon1 = c1 * use
        ncon2 = c1 * CIJ
        ncon = np.sum(ncon1 + ncon2, axis=1)
        print(ncon)

        M[:, i] = 2 * np.sum(np.logical_and(ncon1, ncon2), axis=1) / ncon

    M *= I
    M[np.isnan(M)] = 0
    M0 = np.zeros((n, n))
    yR, = np.where(R)
    M0[np.ix_(yR, yR)] = M
    return M0


def dice_pairwise_und(a1, a2):
    '''
    Calculates pairwise dice similarity for each vertex between two
    matrices. Treats the matrices as binary and undirected.

    Paramaters
    ----------
    A1 : NxN np.ndarray
        Matrix 1
    A2 : NxN np.ndarray
        Matrix 2

    Returns
    -------
    D : Nx1 np.ndarray
        dice similarity vector
    '''
    a1 = binarize(a1, copy=True)
    a2 = binarize(a2, copy=True)  # ensure matrices are binary

    n = len(a1)
    np.fill_diagonal(a1, 0)
    np.fill_diagonal(a2, 0)  # set diagonals to 0

    d = np.zeros((n,))  # dice similarity

    # calculate the common neighbors for each vertex
    for i in range(n):
        d[i] = 2 * (np.sum(np.logical_and(a1[:, i], a2[:, i])) /
                    (np.sum(a1[:, i]) + np.sum(a2[:, i])))

    return d


def corr_flat_und(a1, a2):
    '''
    Returns the correlation coefficient between two flattened adjacency
    matrices.  Only the upper triangular part is used to avoid double counting
    undirected matrices.  Similarity metric for weighted matrices.

    Parameters
    ----------
    A1 : NxN np.ndarray
        undirected matrix 1
    A2 : NxN np.ndarray
        undirected matrix 2

    Returns
    -------
    r : float
        Correlation coefficient describing edgewise similarity of a1 and a2
    '''
    n = len(a1)
    if len(a2) != n:
        raise BCTParamError("Cannot calculate flattened correlation on "
                            "matrices of different size")
    triu_ix = np.where(np.triu(np.ones((n, n)), 1))
    return np.corrcoef(a1[triu_ix].flat, a2[triu_ix].flat)[0][1]


def corr_flat_dir(a1, a2):
    '''
    Returns the correlation coefficient between two flattened adjacency
    matrices.  Similarity metric for weighted matrices.

    Parameters
    ----------
    A1 : NxN np.ndarray
        directed matrix 1
    A2 : NxN np.ndarray
        directed matrix 2

    Returns
    -------
    r : float
        Correlation coefficient describing edgewise similarity of a1 and a2
    '''
    n = len(a1)
    if len(a2) != n:
        raise BCTParamError("Cannot calculate flattened correlation on "
                            "matrices of different size")
    ix = np.logical_not(np.eye(n))
    return np.corrcoef(a1[ix].flat, a2[ix].flat)[0][1]
