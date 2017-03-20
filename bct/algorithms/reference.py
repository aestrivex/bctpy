from __future__ import division, print_function
import numpy as np
from bct.utils import BCTParamError, binarize
from bct.utils import pick_four_unique_nodes_quickly
from .clustering import number_of_components


def latmio_dir_connected(R, itr, D=None):
    '''
    This function "latticizes" a directed network, while preserving the in-
    and out-degree distributions. In weighted networks, the function
    preserves the out-strength but not the in-strength distributions. The
    function also ensures that the randomized network maintains
    connectedness, the ability for every node to reach every other node in
    the network. The input network for this function must be connected.

    Parameters
    ----------
    R : NxN np.ndarray
        directed binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.
    D : np.ndarray | None
        distance-to-diagonal matrix. Defaults to the actual distance matrix
        if not specified.

    Returns
    -------
    Rlatt : NxN np.ndarray
        latticized network in original node ordering
    Rrp : NxN np.ndarray
        latticized network in node ordering used for latticization
    ind_rp : Nx1 np.ndarray
        node ordering used for latticization
    eff : int
        number of actual rewirings carried out
    '''
    n = len(R)

    ind_rp = np.random.permutation(n)  # random permutation of nodes
    R = R.copy()
    R = R[np.ix_(ind_rp, ind_rp)]

    # create distance to diagonal matrix if not specified by user
    if D is None:
        D = np.zeros((n, n))
        un = np.mod(range(1, n), n)
        um = np.mod(range(n - 1, 0, -1), n)
        u = np.append((0,), np.where(un < um, un, um))

        for v in range(int(np.ceil(n / 2))):
            D[n - v - 1, :] = np.append(u[v + 1:], u[:v + 1])
            D[v, :] = D[n - v - 1, :][::-1]

    i, j = np.where(R)
    k = len(i)
    itr *= k

    # maximal number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1)))

    # actual number of successful rewirings
    eff = 0

    for it in range(itr):
        att = 0
        while att <= max_attempts:  # while not rewired
            rewire = True
            while True:
                e1 = np.random.randint(k)
                e2 = np.random.randint(k)
                while e1 == e2:
                    e2 = np.random.randint(k)
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if a != c and a != d and b != c and b != d:
                    break

            # rewiring condition
            if not (R[a, d] or R[c, b]):
                # lattice condition
                if (D[a, b] * R[a, b] + D[c, d] * R[c, d] >= D[a, d] * R[a, b] + D[c, b] * R[c, d]):
                    # connectedness condition
                    if not (np.any((R[a, c], R[d, b], R[d, c])) and
                            np.any((R[c, a], R[b, d], R[b, a]))):
                        P = R[(a, c), :].copy()
                        P[0, b] = 0
                        P[0, d] = 1
                        P[1, d] = 0
                        P[1, b] = 1
                        PN = P.copy()
                        PN[0, a] = 1
                        PN[1, c] = 1
                        while True:
                            P[0, :] = np.any(R[P[0, :] != 0, :], axis=0)
                            P[1, :] = np.any(R[P[1, :] != 0, :], axis=0)
                            P *= np.logical_not(PN)
                            PN += P
                            if not np.all(np.any(P, axis=1)):
                                rewire = False
                                break
                            elif np.any(PN[0, (b, c)]) and np.any(PN[1, (d, a)]):
                                break
                    # end connectedness testing

                    if rewire:  # reassign edges
                        R[a, d] = R[a, b]
                        R[a, b] = 0
                        R[c, b] = R[c, d]
                        R[c, d] = 0

                        j.setflags(write=True)
                        j[e1] = d
                        j[e2] = b  # reassign edge indices
                        eff += 1
                        break
            att += 1

    Rlatt = R[np.ix_(ind_rp[::-1], ind_rp[::-1])]  # reverse random permutation

    return Rlatt, R, ind_rp, eff


def latmio_dir(R, itr, D=None):
    '''
    This function "latticizes" a directed network, while preserving the in-
    and out-degree distributions. In weighted networks, the function
    preserves the out-strength but not the in-strength distributions.

    Parameters
    ----------
    R : NxN np.ndarray
        directed binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.
    D : np.ndarray | None
        distance-to-diagonal matrix. Defaults to the actual distance matrix
        if not specified.

    Returns
    -------
    Rlatt : NxN np.ndarray
        latticized network in original node ordering
    Rrp : NxN np.ndarray
        latticized network in node ordering used for latticization
    ind_rp : Nx1 np.ndarray
        node ordering used for latticization
    eff : int
        number of actual rewirings carried out
    '''
    n = len(R)

    ind_rp = np.random.permutation(n)  # randomly reorder matrix
    R = R.copy()
    R = R[np.ix_(ind_rp, ind_rp)]

    # create distance to diagonal matrix if not specified by user
    if D is None:
        D = np.zeros((n, n))
        un = np.mod(range(1, n), n)
        um = np.mod(range(n - 1, 0, -1), n)
        u = np.append((0,), np.where(un < um, un, um))

        for v in range(int(np.ceil(n / 2))):
            D[n - v - 1, :] = np.append(u[v + 1:], u[:v + 1])
            D[v, :] = D[n - v - 1, :][::-1]

    i, j = np.where(R)
    k = len(i)
    itr *= k

    # maximal number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1)))

    # actual number of successful rewirings
    eff = 0

    for it in range(itr):
        att = 0
        while att <= max_attempts:  # while not rewired
            while True:
                e1 = np.random.randint(k)
                e2 = np.random.randint(k)
                while e1 == e2:
                    e2 = np.random.randint(k)
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if a != c and a != d and b != c and b != d:
                    break

            # rewiring condition
            if not (R[a, d] or R[c, b]):
                # lattice condition
                if (D[a, b] * R[a, b] + D[c, d] * R[c, d] >= D[a, d] * R[a, b] + D[c, b] * R[c, d]):
                    R[a, d] = R[a, b]
                    R[a, b] = 0
                    R[c, b] = R[c, d]
                    R[c, d] = 0

                    j.setflags(write=True)
                    j[e1] = d
                    j[e2] = b  # reassign edge indices
                    eff += 1
                    break
            att += 1

    Rlatt = R[np.ix_(ind_rp[::-1], ind_rp[::-1])]  # reverse random permutation

    return Rlatt, R, ind_rp, eff


def latmio_und_connected(R, itr, D=None):
    '''
    This function "latticizes" an undirected network, while preserving the
    degree distribution. The function does not preserve the strength
    distribution in weighted networks. The function also ensures that the
    randomized network maintains connectedness, the ability for every node
    to reach every other node in the network. The input network for this
    function must be connected.

    Parameters
    ----------
    R : NxN np.ndarray
        undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.
    D : np.ndarray | None
        distance-to-diagonal matrix. Defaults to the actual distance matrix
        if not specified.

    Returns
    -------
    Rlatt : NxN np.ndarray
        latticized network in original node ordering
    Rrp : NxN np.ndarray
        latticized network in node ordering used for latticization
    ind_rp : Nx1 np.ndarray
        node ordering used for latticization
    eff : int
        number of actual rewirings carried out
    '''
    if not np.all(R == R.T):
        raise BCTParamError("Input must be undirected")

    if number_of_components(R) > 1:
        raise BCTParamError("Input is not connected")

    n = len(R)

    ind_rp = np.random.permutation(n)  # randomly reorder matrix
    R = R.copy()
    R = R[np.ix_(ind_rp, ind_rp)]

    if D is None:
        D = np.zeros((n, n))
        un = np.mod(range(1, n), n)
        um = np.mod(range(n - 1, 0, -1), n)
        u = np.append((0,), np.where(un < um, un, um))

        for v in range(int(np.ceil(n / 2))):
            D[n - v - 1, :] = np.append(u[v + 1:], u[:v + 1])
            D[v, :] = D[n - v - 1, :][::-1]

    i, j = np.where(np.tril(R))
    k = len(i)
    itr *= k

    # maximal number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1) / 2))

    # actual number of successful rewirings
    eff = 0

    for it in range(itr):
        att = 0
        while att <= max_attempts:
            rewire = True
            while True:
                e1 = np.random.randint(k)
                e2 = np.random.randint(k)
                while e1 == e2:
                    e2 = np.random.randint(k)
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if a != c and a != d and b != c and b != d:
                    break

            if np.random.random() > .5:
                i.setflags(write=True)
                j.setflags(write=True)
                i[e2] = d
                j[e2] = c  # flip edge c-d with 50% probability
                c = i[e2]
                d = j[e2]  # to explore all potential rewirings

            # rewiring condition
            if not (R[a, d] or R[c, b]):
                # lattice condition
                if (D[a, b] * R[a, b] + D[c, d] * R[c, d] >= D[a, d] * R[a, b] + D[c, b] * R[c, d]):
                    # connectedness condition
                    if not (R[a, c] or R[b, d]):
                        P = R[(a, d), :].copy()
                        P[0, b] = 0
                        P[1, c] = 0
                        PN = P.copy()
                        PN[:, d] = 1
                        PN[:, a] = 1
                        while True:
                            P[0, :] = np.any(R[P[0, :] != 0, :], axis=0)
                            P[1, :] = np.any(R[P[1, :] != 0, :], axis=0)
                            P *= np.logical_not(PN)
                            if not np.all(np.any(P, axis=1)):
                                rewire = False
                                break
                            elif np.any(P[:, (b, c)]):
                                break
                            PN += P
                    # end connectedness testing

                    if rewire:  # reassign edges
                        R[a, d] = R[a, b]
                        R[a, b] = 0
                        R[d, a] = R[b, a]
                        R[b, a] = 0
                        R[c, b] = R[c, d]
                        R[c, d] = 0
                        R[b, c] = R[d, c]
                        R[d, c] = 0

                        j.setflags(write=True)
                        j[e1] = d
                        j[e2] = b
                        eff += 1
                        break
            att += 1

    Rlatt = R[np.ix_(ind_rp[::-1], ind_rp[::-1])]
    return Rlatt, R, ind_rp, eff


def latmio_und(R, itr, D=None):
    '''
    This function "latticizes" an undirected network, while preserving the
    degree distribution. The function does not preserve the strength
    distribution in weighted networks.

    Parameters
    ----------
    R : NxN np.ndarray
        undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.
    D : np.ndarray | None
        distance-to-diagonal matrix. Defaults to the actual distance matrix
        if not specified.

    Returns
    -------
    Rlatt : NxN np.ndarray
        latticized network in original node ordering
    Rrp : NxN np.ndarray
        latticized network in node ordering used for latticization
    ind_rp : Nx1 np.ndarray
        node ordering used for latticization
    eff : int
        number of actual rewirings carried out
    '''
    n = len(R)

    ind_rp = np.random.permutation(n)  # randomly reorder matrix
    R = R.copy()
    R = R[np.ix_(ind_rp, ind_rp)]

    if D is None:
        D = np.zeros((n, n))
        un = np.mod(range(1, n), n)
        um = np.mod(range(n - 1, 0, -1), n)
        u = np.append((0,), np.where(un < um, un, um))

        for v in range(int(np.ceil(n / 2))):
            D[n - v - 1, :] = np.append(u[v + 1:], u[:v + 1])
            D[v, :] = D[n - v - 1, :][::-1]

    i, j = np.where(np.tril(R))
    k = len(i)
    itr *= k

    # maximal number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1) / 2))

    # actual number of successful rewirings
    eff = 0

    for it in range(itr):
        att = 0
        while att <= max_attempts:
            while True:
                e1 = np.random.randint(k)
                e2 = np.random.randint(k)
                while e1 == e2:
                    e2 = np.random.randint(k)
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if a != c and a != d and b != c and b != d:
                    break

            if np.random.random() > .5:
                i.setflags(write=True)
                j.setflags(write=True)
                i[e2] = d
                j[e2] = c  # flip edge c-d with 50% probability
                c = i[e2]
                d = j[e2]  # to explore all potential rewirings

            # rewiring condition
            if not (R[a, d] or R[c, b]):
                # lattice condition
                if (D[a, b] * R[a, b] + D[c, d] * R[c, d] >= D[a, d] * R[a, b] + D[c, b] * R[c, d]):
                    R[a, d] = R[a, b]
                    R[a, b] = 0
                    R[d, a] = R[b, a]
                    R[b, a] = 0
                    R[c, b] = R[c, d]
                    R[c, d] = 0
                    R[b, c] = R[d, c]
                    R[d, c] = 0

                    j.setflags(write=True)
                    j[e1] = d
                    j[e2] = b
                    eff += 1
                    break
            att += 1

    Rlatt = R[np.ix_(ind_rp[::-1], ind_rp[::-1])]
    return Rlatt, R, ind_rp, eff


def makeevenCIJ(n, k, sz_cl):
    '''
    This function generates a random, directed network with a specified
    number of fully connected modules linked together by evenly distributed
    remaining random connections.

    Parameters
    ----------
    N : int
        number of vertices (must be power of 2)
    K : int
        number of edges
    sz_cl : int
        size of clusters (must be power of 2)

    Returns
    -------
    CIJ : NxN np.ndarray
        connection matrix

    Notes
    -----
    N must be a power of 2.
            A warning is generated if all modules contain more edges than K.
            Cluster size is 2^sz_cl;
    '''
    # compute number of hierarchical levels and adjust cluster size
    mx_lvl = int(np.floor(np.log2(n)))
    sz_cl -= 1

    # make a stupid little template
    t = np.ones((2, 2)) * 2

    # check n against the number of levels
    Nlvl = 2**mx_lvl
    if Nlvl != n:
        print("Warning: n must be a power of 2")
    n = Nlvl

    # create hierarchical template
    for lvl in range(1, mx_lvl):
        s = 2**(lvl + 1)
        CIJ = np.ones((s, s))
        grp1 = range(int(s / 2))
        grp2 = range(int(s / 2), s)
        ix1 = np.add.outer(np.array(grp1) * s, grp1).flatten()
        ix2 = np.add.outer(np.array(grp2) * s, grp2).flatten()
        CIJ.flat[ix1] = t  # numpy indexing is teh sucks :(
        CIJ.flat[ix2] = t
        CIJ += 1
        t = CIJ.copy()

    CIJ -= (np.ones((s, s)) + mx_lvl * np.eye(s))

    # assign connection probabilities
    CIJp = (CIJ >= (mx_lvl - sz_cl))

    # determine nr of non-cluster connections left and their possible positions
    rem_k = k - np.size(np.where(CIJp.flatten()))
    if rem_k < 0:
        print("Warning: K is too small, output matrix contains clusters only")
        return CIJp
    a, b = np.where(np.logical_not(CIJp + np.eye(n)))

    # assign remK randomly dstributed connections
    rp = np.random.permutation(len(a))
    a = a[rp[:rem_k]]
    b = b[rp[:rem_k]]
    for ai, bi in zip(a, b):
        CIJp[ai, bi] = 1

    return np.array(CIJp, dtype=int)


def makefractalCIJ(mx_lvl, E, sz_cl):
    '''
    This function generates a directed network with a hierarchical modular
    organization. All modules are fully connected and connection density
    decays as 1/(E^n), with n = index of hierarchical level.

    Parameters
    ----------
    mx_lvl : int
        number of hierarchical levels, N = 2^mx_lvl
    E : int
        connection density fall off per level
    sz_cl : int
        size of clusters (must be power of 2)

    Returns
    -------
    CIJ : NxN np.ndarray
        connection matrix
    K : int
        number of connections present in output CIJ
    '''
    # make a stupid little template
    t = np.ones((2, 2)) * 2

    # compute N and cluster size
    n = 2**mx_lvl
    sz_cl -= 1

    for lvl in range(1, mx_lvl):
        s = 2**(lvl + 1)
        CIJ = np.ones((s, s))
        grp1 = range(int(s / 2))
        grp2 = range(int(s / 2), s)
        ix1 = np.add.outer(np.array(grp1) * s, grp1).flatten()
        ix2 = np.add.outer(np.array(grp2) * s, grp2).flatten()
        CIJ.flat[ix1] = t  # numpy indexing is teh sucks :(
        CIJ.flat[ix2] = t
        CIJ += 1
        t = CIJ.copy()

    CIJ -= (np.ones((s, s)) + mx_lvl * np.eye(s))

    # assign connection probabilities
    ee = mx_lvl - CIJ - sz_cl
    ee = (ee > 0) * ee
    prob = (1 / E**ee) * (np.ones((s, s)) - np.eye(s))
    CIJ = (prob > np.random.random((n, n)))

    # count connections
    k = np.sum(CIJ)

    return np.array(CIJ, dtype=int), k


def makerandCIJdegreesfixed(inv, outv):
    '''
    This function generates a directed random network with a specified
    in-degree and out-degree sequence.

    Parameters
    ----------
    inv : Nx1 np.ndarray
        in-degree vector
    outv : Nx1 np.ndarray
        out-degree vector

    Returns
    -------
    CIJ : NxN np.ndarray

    Notes
    -----
    Necessary conditions include:
            length(in) = length(out) = n
            sum(in) = sum(out) = k
            in(i), out(i) < n-1
            in(i) + out(j) < n+2
            in(i) + out(i) < n

        No connections are placed on the main diagonal

        The algorithm used in this function is not, technically, guaranteed to
        terminate. If a valid distribution of in and out degrees is provided,
        this function will find it in bounded time with probability
        1-(1/(2*(k^2))). This turns out to be a serious problem when
        computing infinite degree matrices, but offers good performance
        otherwise.
    '''
    n = len(inv)
    k = np.sum(inv)
    in_inv = np.zeros((k,))
    out_inv = np.zeros((k,))
    i_in = 0
    i_out = 0

    for i in range(n):
        in_inv[i_in:i_in + inv[i]] = i
        out_inv[i_out:i_out + outv[i]] = i
        i_in += inv[i]
        i_out += outv[i]

    CIJ = np.eye(n)
    edges = np.array((out_inv, in_inv[np.random.permutation(k)]))

    # create CIJ and check for double edges and self connections
    for i in range(k):
        if CIJ[edges[0, i], edges[1, i]]:
            tried = set()
            while True:
                if len(tried) == k:
                    raise BCTParamError('Could not resolve the given '
                                        'in and out vectors')
                switch = np.random.randint(k)
                while switch in tried:
                    switch = np.random.randint(k)
                if not (CIJ[edges[0, i], edges[1, switch]] or
                        CIJ[edges[0, switch], edges[1, i]]):
                    CIJ[edges[0, switch], edges[1, switch]] = 0
                    CIJ[edges[0, switch], edges[1, i]] = 1
                    if switch < i:
                        CIJ[edges[0, switch], edges[1, switch]] = 0
                        CIJ[edges[0, switch], edges[1, i]] = 1
                    t = edges[1, i]
                    edges[1, i] = edges[1, switch]
                    edges[1, switch] = t
                    break
                tried.add(switch)
        else:
            CIJ[edges[0, i], edges[1, i]] = 1

    CIJ -= np.eye(n)
    return CIJ


def makerandCIJ_dir(n, k):
    '''
    This function generates a directed random network

    Parameters
    ----------
    N : int
        number of vertices
    K : int
        number of edges

    Returns
    -------
    CIJ : NxN np.ndarray
        directed random connection matrix

    Notes
    -----
    no connections are placed on the main diagonal.
    '''
    ix, = np.where(np.logical_not(np.eye(n)).flat)
    rp = np.random.permutation(np.size(ix))

    CIJ = np.zeros((n, n))
    CIJ.flat[ix[rp][:k]] = 1
    return CIJ


def makerandCIJ_und(n, k):
    '''
    This function generates an undirected random network

    Parameters
    ----------
    N : int
        number of vertices
    K : int
        number of edges

    Returns
    -------
    CIJ : NxN np.ndarray
        undirected random connection matrix

    Notes
    -----
    no connections are placed on the main diagonal.
    '''
    ix, = np.where(np.triu(np.logical_not(np.eye(n))).flat)
    rp = np.random.permutation(np.size(ix))

    CIJ = np.zeros((n, n))
    CIJ.flat[ix[rp][:k]] = 1
    return CIJ


def makeringlatticeCIJ(n, k):
    '''
    This function generates a directed lattice network with toroidal
    boundary counditions (i.e. with ring-like "wrapping around").

    Parameters
    ----------
    N : int
        number of vertices
    K : int
        number of edges

    Returns
    -------
    CIJ : NxN np.ndarray
        connection matrix

    Notes
    -----
    The lattice is made by placing connections as close as possible
    to the main diagonal, with wrapping around. No connections are made
    on the main diagonal. In/Outdegree is kept approx. constant at K/N.
    '''
    # initialize
    CIJ = np.zeros((n, n))
    CIJ1 = np.ones((n, n))
    kk = 0
    count = 0
    seq = range(1, n)
    seq2 = range(n - 1, 0, -1)

    # fill in
    while kk < k:
        count += 1
        dCIJ = np.triu(CIJ1, seq[count]) - np.triu(CIJ1, seq[count] + 1)
        dCIJ2 = np.triu(CIJ1, seq2[count]) - np.triu(CIJ1, seq2[count] + 1)
        dCIJ = dCIJ + dCIJ.T + dCIJ2 + dCIJ2.T
        CIJ += dCIJ
        kk = int(np.sum(CIJ))

    # remove excess connections
    overby = kk - k
    if overby:
        i, j = np.where(dCIJ)
        rp = np.random.permutation(np.size(i))
        for ii in range(overby):
            CIJ[i[rp[ii]], j[rp[ii]]] = 0

    return CIJ


def maketoeplitzCIJ(n, k, s):
    '''
    This function generates a directed network with a Gaussian drop-off in
    edge density with increasing distance from the main diagonal. There are
    toroidal boundary counditions (i.e. no ring-like "wrapping around").

    Parameters
    ----------
    N : int
        number of vertices
    K : int
        number of edges
    s : float
        standard deviation of toeplitz

    Returns
    -------
    CIJ : NxN np.ndarray
        connection matrix

    Notes
    -----
    no connections are placed on the main diagonal.
    '''
    from scipy import linalg, stats
    pf = stats.norm.pdf(range(1, n), .5, s)
    template = linalg.toeplitz(np.append((0,), pf), r=np.append((0,), pf))
    template *= (k / np.sum(template))

    CIJ = np.zeros((n, n))
    itr = 0
    while np.sum(CIJ) != k:
        CIJ = (np.random.random((n, n)) < template)
        itr += 1
        if itr > 10000:
            raise BCTParamError('Infinite loop was caught generating toeplitz '
                                'matrix.  This means the matrix could not be resolved with the '
                                'specified parameters.')

    return CIJ


def null_model_dir_sign(W, bin_swaps=5, wei_freq=.1):
    '''
    This function randomizes an directed network with positive and
    negative weights, while preserving the degree and strength
    distributions. This function calls randmio_dir.m

    Parameters
    ----------
    W : NxN np.ndarray
        directed weighted connection matrix
    bin_swaps : int
        average number of swaps in each edge binary randomization. Default
        value is 5. 0 swaps implies no binary randomization.
    wei_freq : float
        frequency of weight sorting in weighted randomization. 0<=wei_freq<1.
        wei_freq == 1 implies that weights are sorted at each step.
        wei_freq == 0.1 implies that weights sorted each 10th step (faster,
            default value)
        wei_freq == 0 implies no sorting of weights (not recommended)

    Returns
    -------
    W0 : NxN np.ndarray
        randomized weighted connection matrix
    R : 4-tuple of floats
        Correlation coefficients between strength sequences of input and
        output connection matrices, rpos_in, rpos_out, rneg_in, rneg_out

    Notes
    -----
    The value of bin_swaps is ignored when binary topology is fully
       connected (e.g. when the network has no negative weights).
    Randomization may be better (and execution time will be slower) for
       higher values of bin_swaps and wei_freq. Higher values of bin_swaps may
       enable a more random binary organization, and higher values of wei_freq
       may enable a more accurate conservation of strength sequences.
    R are the correlation coefficients between positive and negative
       in-strength and out-strength sequences of input and output connection
       matrices and are used to evaluate the accuracy with which strengths
       were preserved. Note that correlation coefficients may be a rough
       measure of strength-sequence accuracy and one could implement more
       formal tests (such as the Kolmogorov-Smirnov test) if desired.
    '''
    W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)  # clear diagonal
    Ap = (W > 0)  # positive adjmat

    if np.size(np.where(Ap.flat)) < (n * (n - 1)):
        W_r = randmio_und_signed(W, bin_swaps)
        Ap_r = W_r > 0
        An_r = W_r < 0
    else:
        Ap_r = Ap
        An_r = An

    W0 = np.zeros((n, n))
    for s in (1, -1):
        if s == 1:
            Acur = Ap
            A_rcur = Ap_r
        else:
            Acur = An
            A_rcur = An_r

        Si = np.sum(W * Acur, axis=0)  # positive in-strength
        So = np.sum(W * Acur, axis=1)  # positive out-strength
        Wv = np.sort(W[Acur].flat)  # sorted weights vector
        i, j = np.where(A_rcur)
        Lij, = np.where(A_rcur.flat)  # weights indices

        P = np.outer(So, Si)

        if wei_freq == 0:  # get indices of Lij that sort P
            Oind = np.argsort(P.flat[Lij])  # assign corresponding sorted
            W0.flat[Lij[Oind]] = s * Wv  # weight at this index
        else:
            wsize = np.size(Wv)
            wei_period = np.round(1 / wei_freq)  # convert frequency to period
            lq = np.arange(wsize, 0, -wei_period, dtype=int)
            for m in lq:  # iteratively explore at this period
                # get indices of Lij that sort P
                Oind = np.argsort(P.flat[Lij])
                R = np.random.permutation(m)[:np.min((m, wei_period))]
                for q, r in enumerate(R):
                    # choose random index of sorted expected weight
                    o = Oind[r]
                    W0.flat[Lij[o]] = s * Wv[r]  # assign corresponding weight

                    # readjust expected weighted probability for i[o],j[o]
                    f = 1 - Wv[r] / So[i[o]]
                    P[i[o], :] *= f
                    f = 1 - Wv[r] / So[j[o]]
                    P[j[o], :] *= f

                    # readjust in-strength of i[o]
                    So[i[o]] -= Wv[r]
                    # readjust out-strength of j[o]
                    Si[j[o]] -= Wv[r]

                O = Oind[R]
                # remove current indices from further consideration
                Lij = np.delete(Lij, O)
                i = np.delete(i, O)
                j = np.delete(j, O)
                Wv = np.delete(Wv, O)

    rpos_in = np.corrcoef(np.sum(W * (W > 0), axis=0),
                          np.sum(W0 * (W0 > 0), axis=0))
    rpos_ou = np.corrcoef(np.sum(W * (W > 0), axis=1),
                          np.sum(W0 * (W0 > 0), axis=1))
    rneg_in = np.corrcoef(np.sum(-W * (W < 0), axis=0),
                          np.sum(-W0 * (W0 < 0), axis=0))
    rneg_ou = np.corrcoef(np.sum(-W * (W < 0), axis=1),
                          np.sum(-W0 * (W0 < 0), axis=1))
    return W0, (rpos_in[0, 1], rpos_ou[0, 1], rneg_in[0, 1], rneg_ou[0, 1])


def null_model_und_sign(W, bin_swaps=5, wei_freq=.1):
    '''
    This function randomizes an undirected network with positive and
    negative weights, while preserving the degree and strength
    distributions. This function calls randmio_und.m

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted connection matrix
    bin_swaps : int
        average number of swaps in each edge binary randomization. Default
        value is 5. 0 swaps implies no binary randomization.
    wei_freq : float
        frequency of weight sorting in weighted randomization. 0<=wei_freq<1.
        wei_freq == 1 implies that weights are sorted at each step.
        wei_freq == 0.1 implies that weights sorted each 10th step (faster,
            default value)
        wei_freq == 0 implies no sorting of weights (not recommended)

    Returns
    -------
    W0 : NxN np.ndarray
        randomized weighted connection matrix
    R : 4-tuple of floats
        Correlation coefficients between strength sequences of input and
        output connection matrices, rpos_in, rpos_out, rneg_in, rneg_out

    Notes
    -----
    The value of bin_swaps is ignored when binary topology is fully
        connected (e.g. when the network has no negative weights).
    Randomization may be better (and execution time will be slower) for
        higher values of bin_swaps and wei_freq. Higher values of bin_swaps
        may enable a more random binary organization, and higher values of
        wei_freq may enable a more accurate conservation of strength
        sequences.
    R are the correlation coefficients between positive and negative
        strength sequences of input and output connection matrices and are
        used to evaluate the accuracy with which strengths were preserved.
        Note that correlation coefficients may be a rough measure of
        strength-sequence accuracy and one could implement more formal tests
        (such as the Kolmogorov-Smirnov test) if desired.
    '''
    if not np.all(W == W.T):
        raise BCTParamError("Input must be undirected")
    W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)  # clear diagonal
    Ap = (W > 0)  # positive adjmat
    An = (W < 0)  # negative adjmat

    if np.size(np.where(Ap.flat)) < (n * (n - 1)):
        W_r, eff = randmio_und_signed(W, bin_swaps)
        Ap_r = W_r > 0
        An_r = W_r < 0
    else:
        Ap_r = Ap
        An_r = An

    W0 = np.zeros((n, n))
    for s in (1, -1):
        if s == 1:
            Acur = Ap
            A_rcur = Ap_r
        else:
            Acur = An
            A_rcur = An_r

        S = np.sum(W * Acur, axis=0)  # strengths
        Wv = np.sort(W[np.where(np.triu(Acur))])  # sorted weights vector
        i, j = np.where(np.triu(A_rcur))
        Lij, = np.where(np.triu(A_rcur).flat)  # weights indices

        P = np.outer(S, S)

        if wei_freq == 0:  # get indices of Lij that sort P
            Oind = np.argsort(P.flat[Lij])  # assign corresponding sorted
            W0.flat[Lij[Oind]] = s * Wv  # weight at this index
        else:
            wsize = np.size(Wv)
            wei_period = np.round(1 / wei_freq)  # convert frequency to period
            lq = np.arange(wsize, 0, -wei_period, dtype=int)
            for m in lq:  # iteratively explore at this period
                # get indices of Lij that sort P
                Oind = np.argsort(P.flat[Lij])
                R = np.random.permutation(m)[:np.min((m, wei_period))]
                for q, r in enumerate(R):
                    # choose random index of sorted expected weight
                    o = Oind[r]
                    W0.flat[Lij[o]] = s * Wv[r]  # assign corresponding weight

                    # readjust expected weighted probability for i[o],j[o]
                    f = 1 - Wv[r] / S[i[o]]
                    P[i[o], :] *= f
                    P[:, i[o]] *= f
                    f = 1 - Wv[r] / S[j[o]]
                    P[j[o], :] *= f
                    P[:, j[o]] *= f

                    # readjust strength of i[o]
                    S[i[o]] -= Wv[r]
                    # readjust strength of j[o]
                    S[j[o]] -= Wv[r]

                O = Oind[R]
                # remove current indices from further consideration
                Lij = np.delete(Lij, O)
                i = np.delete(i, O)
                j = np.delete(j, O)
                Wv = np.delete(Wv, R)

    W0 = W0 + W0.T

    rpos_in = np.corrcoef(np.sum(W * (W > 0), axis=0),
                          np.sum(W0 * (W0 > 0), axis=0))
    rpos_ou = np.corrcoef(np.sum(W * (W > 0), axis=1),
                          np.sum(W0 * (W0 > 0), axis=1))
    rneg_in = np.corrcoef(np.sum(-W * (W < 0), axis=0),
                          np.sum(-W0 * (W0 < 0), axis=0))
    rneg_ou = np.corrcoef(np.sum(-W * (W < 0), axis=1),
                          np.sum(-W0 * (W0 < 0), axis=1))
    return W0, (rpos_in[0, 1], rpos_ou[0, 1], rneg_in[0, 1], rneg_ou[0, 1])


def randmio_dir_connected(R, itr):
    '''
    This function randomizes a directed network, while preserving the in-
    and out-degree distributions. In weighted networks, the function
    preserves the out-strength but not the in-strength distributions. The
    function also ensures that the randomized network maintains
    connectedness, the ability for every node to reach every other node in
    the network. The input network for this function must be connected.

    Parameters
    ----------
    W : NxN np.ndarray
        directed binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.

    Returns
    -------
    R : NxN np.ndarray
        randomized network
    eff : int
        number of actual rewirings carried out
    '''
    R = R.copy()
    n = len(R)
    i, j = np.where(R)
    k = len(i)
    itr *= k

    max_attempts = np.round(n * k / (n * (n - 1)))
    eff = 0

    for it in range(int(itr)):
        att = 0
        while att <= max_attempts:  # while not rewired
            rewire = True
            while True:
                e1 = np.random.randint(k)
                e2 = np.random.randint(k)
                while e1 == e2:
                    e2 = np.random.randint(k)
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if a != c and a != d and b != c and b != d:
                    break  # all 4 vertices must be different

            # rewiring condition
            if not (R[a, d] or R[c, b]):
                # connectedness condition
                if not (np.any((R[a, c], R[d, b], R[d, c])) and
                        np.any((R[c, a], R[b, d], R[b, a]))):
                    P = R[(a, c), :].copy()
                    P[0, b] = 0
                    P[0, d] = 1
                    P[1, d] = 0
                    P[1, b] = 1
                    PN = P.copy()
                    PN[0, a] = 1
                    PN[1, c] = 1
                    while True:
                        P[0, :] = np.any(R[P[0, :] != 0, :], axis=0)
                        P[1, :] = np.any(R[P[1, :] != 0, :], axis=0)
                        P *= np.logical_not(PN)
                        PN += P
                        if not np.all(np.any(P, axis=1)):
                            rewire = False
                            break
                        elif np.any(PN[0, (b, c)]) and np.any(PN[1, (d, a)]):
                            break
                # end connectedness testing

                if rewire:  # reassign edges
                    R[a, d] = R[a, b]
                    R[a, b] = 0
                    R[c, b] = R[c, d]
                    R[c, d] = 0

                    j.setflags(write=True)
                    j[e1] = d  # reassign edge indices
                    j[e2] = b
                    eff += 1
                    break
            att += 1

    return R, eff


def randmio_dir(R, itr):
    '''
    This function randomizes a directed network, while preserving the in-
    and out-degree distributions. In weighted networks, the function
    preserves the out-strength but not the in-strength distributions.

    Parameters
    ----------
    W : NxN np.ndarray
        directed binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.

    Returns
    -------
    R : NxN np.ndarray
        randomized network
    eff : int
        number of actual rewirings carried out
    '''
    R = R.copy()
    n = len(R)
    i, j = np.where(R)
    k = len(i)
    itr *= k

    max_attempts = np.round(n * k / (n * (n - 1)))
    eff = 0

    for it in range(int(itr)):
        att = 0
        while att <= max_attempts:  # while not rewired
            while True:
                e1 = np.random.randint(k)
                e2 = np.random.randint(k)
                while e1 == e2:
                    e2 = np.random.randint(k)
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if a != c and a != d and b != c and b != d:
                    break  # all 4 vertices must be different

            # rewiring condition
            if not (R[a, d] or R[c, b]):
                R[a, d] = R[a, b]
                R[a, b] = 0
                R[c, b] = R[c, d]
                R[c, d] = 0

                i.setflags(write=True)
                j.setflags(write=True)
                i[e1] = d
                j[e2] = b  # reassign edge indices
                eff += 1
                break
            att += 1

    return R, eff


def randmio_und_connected(R, itr):
    '''
    This function randomizes an undirected network, while preserving the
    degree distribution. The function does not preserve the strength
    distribution in weighted networks. The function also ensures that the
    randomized network maintains connectedness, the ability for every node
    to reach every other node in the network. The input network for this
    function must be connected.

    NOTE the changes to the BCT matlab function of the same name 
    made in the Jan 2016 release 
    have not been propagated to this function because of substantially
    decreased time efficiency in the implementation. Expect these changes
    to be merged eventually.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.

    Returns
    -------
    R : NxN np.ndarray
        randomized network
    eff : int
        number of actual rewirings carried out
    '''
    if not np.all(R == R.T):
        raise BCTParamError("Input must be undirected")

    if number_of_components(R) > 1:
        raise BCTParamError("Input is not connected")

    R = R.copy()
    n = len(R)
    i, j = np.where(np.tril(R))
    k = len(i)
    itr *= k

    # maximum number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1)))
    # actual number of successful rewirings
    eff = 0

    for it in range(int(itr)):
        att = 0
        while att <= max_attempts:  # while not rewired
            rewire = True
            while True:
                e1 = np.random.randint(k)
                e2 = np.random.randint(k)
                while e1 == e2:
                    e2 = np.random.randint(k)
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if a != c and a != d and b != c and b != d:
                    break  # all 4 vertices must be different

            if np.random.random() > .5:

                i.setflags(write=True)
                j.setflags(write=True)
                i[e2] = d
                j[e2] = c  # flip edge c-d with 50% probability
                c = i[e2]
                d = j[e2]  # to explore all potential rewirings

            # rewiring condition
            if not (R[a, d] or R[c, b]):
                # connectedness condition
                if not (R[a, c] or R[b, d]):
                    P = R[(a, d), :].copy()
                    P[0, b] = 0
                    P[1, c] = 0
                    PN = P.copy()
                    PN[:, d] = 1
                    PN[:, a] = 1
                    while True:
                        P[0, :] = np.any(R[P[0, :] != 0, :], axis=0)
                        P[1, :] = np.any(R[P[1, :] != 0, :], axis=0)
                        P *= np.logical_not(PN)
                        if not np.all(np.any(P, axis=1)):
                            rewire = False
                            break
                        elif np.any(P[:, (b, c)]):
                            break
                        PN += P
                # end connectedness testing

                if rewire:
                    R[a, d] = R[a, b]
                    R[a, b] = 0
                    R[d, a] = R[b, a]
                    R[b, a] = 0
                    R[c, b] = R[c, d]
                    R[c, d] = 0
                    R[b, c] = R[d, c]
                    R[d, c] = 0

                    j.setflags(write=True)
                    j[e1] = d
                    j[e2] = b  # reassign edge indices
                    eff += 1
                    break
            att += 1

    return R, eff


def randmio_dir_signed(R, itr):
    '''
    This function randomizes a directed weighted network with positively
    and negatively signed connections, while preserving the positive and
    negative degree distributions. In weighted networks by default the
    function preserves the out-degree strength but not the in-strength
    distributions

    Parameters
    ---------
    W : NxN np.ndarray
        directed binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.

    Returns
    -------
    R : NxN np.ndarray
        randomized network
    eff : int
        number of actual rewirings carried out
    '''
    R = R.copy()
    n = len(R)

    itr *= n * (n - 1)

    #maximal number of rewiring attempts per iter
    max_attempts = n
    #actual number of successful rewirings
    eff = 0

    #print(itr)

    for it in range(int(itr)):
        #print(it)
        att = 0
        while att <= max_attempts:
            #select four distinct vertices
        
            a, b, c, d = pick_four_unique_nodes_quickly(n)

            #a, b, c, d = np.random.choice(n, 4)
            #a, b, c, d = np.random.permutation(4)

            r0_ab = R[a, b]
            r0_cd = R[c, d]
            r0_ad = R[a, d]
            r0_cb = R[c, b]

            #print(np.sign(r0_ab), np.sign(r0_ad))

            #rewiring condition
            if (    np.sign(r0_ab) == np.sign(r0_cd) and
                    np.sign(r0_ad) == np.sign(r0_cb) and
                    np.sign(r0_ab) != np.sign(r0_ad)):


                R[a, d] = r0_ab
                R[a, b] = r0_ad
                R[c, b] = r0_cd
                R[c, d] = r0_cb

                eff += 1
                break

            att += 1

    #print(eff)

    return R, eff

def randmio_und(R, itr):
    '''
    This function randomizes an undirected network, while preserving the
    degree distribution. The function does not preserve the strength
    distribution in weighted networks.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.

    Returns
    -------
    R : NxN np.ndarray
        randomized network
    eff : int
        number of actual rewirings carried out
    '''
    if not np.all(R == R.T):
        raise BCTParamError("Input must be undirected")
    R = R.copy()
    n = len(R)
    i, j = np.where(np.tril(R))
    k = len(i)
    itr *= k

    # maximum number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1)))
    # actual number of successful rewirings
    eff = 0

    for it in range(int(itr)):
        att = 0
        while att <= max_attempts:  # while not rewired
            while True:
                e1, e2 = np.random.randint(k, size=(2,))
                while e1 == e2:
                    e2 = np.random.randint(k)
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if a != c and a != d and b != c and b != d:
                    break  # all 4 vertices must be different

            if np.random.random() > .5:
                i.setflags(write=True)
                j.setflags(write=True)
                i[e2] = d
                j[e2] = c  # flip edge c-d with 50% probability
                c = i[e2]
                d = j[e2]  # to explore all potential rewirings

            # rewiring condition
            if not (R[a, d] or R[c, b]):
                R[a, d] = R[a, b]
                R[a, b] = 0
                R[d, a] = R[b, a]
                R[b, a] = 0
                R[c, b] = R[c, d]
                R[c, d] = 0
                R[b, c] = R[d, c]
                R[d, c] = 0

                j.setflags(write=True)
                j[e1] = d
                j[e2] = b  # reassign edge indices
                eff += 1
                break
            att += 1

    return R, eff


def randmio_und_signed(R, itr):
    '''
    This function randomizes an undirected weighted network with positive
    and negative weights, while simultaneously preserving the degree
    distribution of positive and negative weights. The function does not
    preserve the strength distribution in weighted networks.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.

    Returns
    -------
    R : NxN np.ndarray
        randomized network
    '''
    R = R.copy()
    n = len(R)

    itr *= int(n * (n -1) / 2)

    max_attempts = int(np.round(n / 2))
    eff = 0

    for it in range(int(itr)):
        att = 0
        while att <= max_attempts:

            a, b, c, d = pick_four_unique_nodes_quickly(n)

            r0_ab = R[a, b]
            r0_cd = R[c, d]
            r0_ad = R[a, d]
            r0_cb = R[c, b]

            #rewiring condition
            if (    np.sign(r0_ab) == np.sign(r0_cd) and
                    np.sign(r0_ad) == np.sign(r0_cb) and
                    np.sign(r0_ab) != np.sign(r0_ad)):
        
                R[a, d] = R[d, a] = r0_ab
                R[a, b] = R[b, a] = r0_ad

                R[c, b] = R[b, c] = r0_cd
                R[c, d] = R[d, c] = r0_cb

                eff += 1
                break

            att += 1

    return R, eff

def randomize_graph_partial_und(A, B, maxswap):
    '''
    A = RANDOMIZE_GRAPH_PARTIAL_UND(A,B,MAXSWAP) takes adjacency matrices A
    and B and attempts to randomize matrix A by performing MAXSWAP
    rewirings. The rewirings will avoid any spots where matrix B is
    nonzero.

    Parameters
    ----------
    A : NxN np.ndarray
        undirected adjacency matrix to randomize
    B : NxN np.ndarray
        mask; edges to avoid
    maxswap : int
        number of rewirings

    Returns
    -------
    A : NxN np.ndarray
        randomized matrix

    Notes
    -----
    1. Graph may become disconnected as a result of rewiring. Always
      important to check.
    2. A can be weighted, though the weighted degree sequence will not be
      preserved.
    3. A must be undirected.
    '''
    A = A.copy()
    i, j = np.where(np.triu(A, 1))
    i.setflags(write=True)
    j.setflags(write=True)
    m = len(i)

    nswap = 0
    while nswap < maxswap:
        while True:
            e1, e2 = np.random.randint(m, size=(2,))
            while e1 == e2:
                e2 = np.random.randint(m)
            a = i[e1]
            b = j[e1]
            c = i[e2]
            d = j[e2]

            if a != c and a != d and b != c and b != d:
                break  # all 4 vertices must be different

        if np.random.random() > .5:
            i[e2] = d
            j[e2] = c  # flip edge c-d with 50% probability
            c = i[e2]
            d = j[e2]  # to explore all potential rewirings

        # rewiring condition
        if not (A[a, d] or A[c, b] or B[a, d] or B[c, b]):  # avoid specified ixes
            A[a, d] = A[a, b]
            A[a, b] = 0
            A[d, a] = A[b, a]
            A[b, a] = 0
            A[c, b] = A[c, d]
            A[c, d] = 0
            A[b, c] = A[d, c]
            A[d, c] = 0

            j[e1] = d
            j[e2] = b  # reassign edge indices
            nswap += 1
    return A


def randomizer_bin_und(R, alpha):
    '''
    This function randomizes a binary undirected network, while preserving
    the degree distribution. The function directly searches for rewirable
    edge pairs (rather than trying to rewire edge pairs at random), and
    hence avoids long loops and works especially well in dense matrices.

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected connection matrix
    alpha : float
        fraction of edges to rewire

    Returns
    -------
    R : NxN np.ndarray
        randomized network
    '''
    R = binarize(R, copy=True)  # binarize
    if not np.all(R == R.T):
        raise BCTParamError(
            'randomizer_bin_und only takes undirected matrices')

    ax = len(R)
    nr_poss_edges = (np.dot(ax, ax) - ax) / 2  # find maximum possible edges

    savediag = np.diag(R)
    np.fill_diagonal(R, np.inf)  # replace diagonal with high value

    # if there are more edges than non-edges, invert the matrix to reduce
    # computation time.  "invert" means swap meaning of 0 and 1, not matrix
    # inversion

    i, j = np.where(np.triu(R, 1))
    k = len(i)
    if k > nr_poss_edges / 2:
        swap = True
        R = np.logical_not(R)
        np.fill_diagonal(R, np.inf)
        i, j = np.where(np.triu(R, 1))
        k = len(i)
    else:
        swap = False

    # exclude fully connected nodes
    fullnodes = np.where((np.sum(np.triu(R, 1), axis=0) +
                          np.sum(np.triu(R, 1), axis=1).T) == (ax - 1))
    if np.size(fullnodes):
        R[fullnodes, :] = 0
        R[:, fullnodes] = 0
        np.fill_diagonal(R, np.inf)
        i, j = np.where(np.triu(R, 1))
        k = len(i)

    if k == 0 or k >= (nr_poss_edges - 1):
        raise BCTParamError("No possible randomization")

    for it in range(k):
        if np.random.random() > alpha:
            continue  # rewire alpha% of edges

        a = i[it]
        b = j[it]  # it is the chosen edge from a<->b

        alliholes, = np.where(R[:, a] == 0)  # find where each end can connect
        alljholes, = np.where(R[:, b] == 0)

        # we can only use edges with connection to neither node
        i_intersect = np.intersect1d(alliholes, alljholes)
        # find which of these nodes are connected
        ii, jj = np.where(R[np.ix_(i_intersect, i_intersect)])

        # if there is an edge to switch
        if np.size(ii):
            # choose one randomly
            nummates = np.size(ii)
            mate = np.random.randint(nummates)

            # randomly orient the second edge
            if np.random.random() > .5:
                c = i_intersect[ii[mate]]
                d = i_intersect[jj[mate]]
            else:
                d = i_intersect[ii[mate]]
                c = i_intersect[jj[mate]]

            # swap the edges
            R[a, b] = 0
            R[c, d] = 0
            R[b, a] = 0
            R[d, c] = 0
            R[a, c] = 1
            R[b, d] = 1
            R[c, a] = 1
            R[d, b] = 1

            # update the edge index (this is inefficient)
            for m in range(k):
                if i[m] == d and j[m] == c:
                    i.setflags(write=True)
                    j.setflags(write=True)
                    i[it] = c
                    j[m] = b
                elif i[m] == c and j[m] == d:
                    i.setflags(write=True)
                    j.setflags(write=True)
                    j[it] = c
                    i[m] = b

    # restore fullnodes
    if np.size(fullnodes):
        R[fullnodes, :] = 1
        R[:, fullnodes] = 1

    # restore inversion
    if swap:
        R = np.logical_not(R)

    # restore diagonal
    np.fill_diagonal(R, 0)
    R += savediag

    return np.array(R, dtype=int)
