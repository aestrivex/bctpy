from __future__ import division, print_function
import numpy as np
from bct.utils import BCTParamError, binarize

motiflib = 'motif34lib.mat'

# FIXME there may be some subtle bugs here


def find_motif34(m, n=None):
    '''
    This function returns all motif isomorphs for a given motif id and
    class (3 or 4). The function also returns the motif id for a given
    motif matrix

    1. Input:       Motif_id,           e.g. 1 to 13, if class is 3
                 Motif_class,        number of nodes, 3 or 4.
    Output:      Motif_matrices,     all isomorphs for the given motif

    2. Input:       Motif_matrix        e.g. [0 1 0; 0 0 1; 1 0 0]
    Output       Motif_id            e.g. 1 to 13, if class is 3

    Parameters
    ----------
    m : int | matrix
        In use case 1, a motif_id which is an integer.
        In use case 2, the entire matrix of the motif
        (e.g. [0 1 0; 0 0 1; 1 0 0])
    n : int | None
        In use case 1, the motif class, which is the number of nodes. This is
        either 3 or 4.
        In use case 2, None.

    Returns
    -------
    M : np.ndarray | int
        In use case 1, returns all isomorphs for the given motif
        In use case 2, returns the motif_id for the specified motif matrix
    '''
    from scipy import io
    import os
    fname = os.path.join(os.path.dirname(__file__), motiflib)
    z = (0,)
    if n == 3:
        mot = io.loadmat(fname)
        m3 = mot['m3']
        id3 = mot['id3'].squeeze()
        ix, = np.where(id3 == m)
        M = np.zeros((3, 3, len(ix)))
        for i, ind in enumerate(ix):
            M[:, :, i] = np.reshape(np.concatenate(
                (z, m3[ind, 0:3], z, m3[ind, 3:6], z)), (3, 3))
    elif n == 4:
        mot = io.loadmat(fname)
        m4 = mot['m4']
        id4 = mot['id4'].squeeze()
        ix, = np.where(id4 == m)
        M = np.zeros((4, 4, len(ix)))
        for i, ind in enumerate(ix):
            M[:, :, i] = np.reshape(np.concatenate(
                (z, m4[ind, 0:4], z, m4[ind, 4:8], z, m4[ind, 8:12], z)), (4, 4))
    elif n is None:
        try:
            m = np.array(m)
        except TypeError:
            raise BCTParamError('motif matrix must be an array-like')
        if m.shape[0] == 3:
            M, = np.where(motif3struct_bin(m))
        elif m.shape[0] == 4:
            M, = np.where(motif4struct_bin(m))
        else:
            raise BCTParamError('motif matrix must be 3x3 or 4x4')
    else:
        raise BCTParamError('Invalid motif class, must be 3, 4, or None')

    return M


def make_motif34lib():
    '''
    This function generates the motif34lib.mat library required for all
    other motif computations. Not to be called externally.
    '''
    from scipy import io
    import os

    def motif3generate():
        n = 0
        M = np.zeros((54, 6), dtype=bool)  # isomorphs
        # canonical labels (predecssors of IDs)
        CL = np.zeros((54, 6), dtype=np.uint8)
        cl = np.zeros((6,), dtype=np.uint8)
        for i in range(2**6):  # loop through all subgraphs
            m = '{0:b}'.format(i)
            m = str().zfill(6 - len(m)) + m
            G = np.array(((0, m[2], m[4]), (m[0], 0, m[5]),
                          (m[1], m[3], 0)), dtype=int)
            ko = np.sum(G, axis=1)
            ki = np.sum(G, axis=0)
            if np.all(ko + ki):  # if subgraph weakly connected
                u = np.array((ko, ki)).T
                cl.flat = u[np.lexsort((ki, ko))]
                CL[n, :] = cl  # assign motif label to isomorph
                M[n, :] = np.array((G.T.flat[1:4], G.T.flat[5:8])).flat
                n += 1

        # convert CLs into motif IDs
        _, ID = np.unique(
            CL.view(CL.dtype.descr * CL.shape[1]), return_inverse=True)
        ID += 1

        # convert IDs into sporns & kotter classification
        id_mika = (1, 3, 4, 6, 7, 8, 11)
        id_olaf = (-3, -6, -1, -11, -4, -7, -8)
        for mika, olaf in zip(id_mika, id_olaf):
            ID[ID == mika] = olaf
        ID = np.abs(ID)

        ix = np.argsort(ID)
        ID = ID[ix]  # sort IDs
        M = M[ix, :]  # sort isomorphs
        N = np.squeeze(np.sum(M, axis=1))  # number of edges
        Mn = np.array(np.sum(np.tile(np.power(10, np.arange(5, -1, -1)),
                                     (M.shape[0], 1)) * M, axis=1), dtype=np.uint32)
        return M, Mn, ID, N

    def motif4generate():
        n = 0
        M = np.zeros((3834, 12), dtype=bool)  # isomorphs
        CL = np.zeros((3834, 16), dtype=np.uint8)  # canonical labels
        cl = np.zeros((16,), dtype=np.uint8)
        for i in range(2**12):  # loop through all subgraphs
            m = '{0:b}'.format(i)
            m = str().zfill(12 - len(m)) + m
            G = np.array(((0, m[3], m[6], m[9]), (m[0], 0, m[7], m[10]),
                          (m[1], m[4], 0, m[11]), (m[2], m[5], m[8], 0)), dtype=int)
            Gs = G + G.T
            v = Gs[0, :]
            for j in range(2):
                v = np.any(Gs[v != 0, :], axis=0) + v
            if np.all(v):  # if subgraph weakly connected
                G2 = np.dot(G, G) != 0
                ko = np.sum(G, axis=1)
                ki = np.sum(G, axis=0)
                ko2 = np.sum(G2, axis=1)
                ki2 = np.sum(G2, axis=0)

                u = np.array((ki, ko, ki2, ko2)).T
                cl.flat = u[np.lexsort((ko2, ki2, ko, ki))]
                CL[n, :] = cl  # assign motif label to isomorph
                M[n, :] = np.array((G.T.flat[1:5], G.T.flat[6:10],
                                    G.T.flat[11:15])).flat
                n += 1

        # convert CLs into motif IDs
        _, ID = np.unique(
            CL.view(CL.dtype.descr * CL.shape[1]), return_inverse=True)
        ID += 1

        ix = np.argsort(ID)
        ID = ID[ix]  # sort IDs
        M = M[ix, :]  # sort isomorphs
        N = np.sum(M, axis=1)  # number of edges
        Mn = np.array(np.sum(np.tile(np.power(10, np.arange(11, -1, -1)),
                                     (M.shape[0], 1)) * M, axis=1), dtype=np.uint64)
        return M, Mn, ID, N

    dir = os.path.dirname(__file__)
    fname = os.path.join(dir, motiflib)
    if os.path.exists(fname):
        print("motif34lib already exists")
        return

    m3, m3n, id3, n3 = motif3generate()
    m4, m4n, id4, n4 = motif4generate()

    io.savemat(fname, mdict={'m3': m3, 'm3n': m3n, 'id3': id3, 'n3': n3,
                             'm4': m4, 'm4n': m4n, 'id4': id4, 'n4': n4})


def motif3funct_bin(A):
    '''
    Functional motifs are subsets of connection patterns embedded within
    anatomical motifs. Motif frequency is the frequency of occurrence of
    motifs around a node.

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    F : 13xN np.ndarray
        motif frequency matrix
    f : 13x1 np.ndarray
        motif frequency vector (averaged over all nodes)
    '''
    from scipy import io
    import os
    fname = os.path.join(os.path.dirname(__file__), motiflib)
    mot = io.loadmat(fname)
    m3 = mot['m3']
    id3 = mot['id3'].squeeze()
    n3 = mot['n3'].squeeze()

    n = len(A)  # number of vertices in A
    f = np.zeros((13,))  # motif count for whole graph
    F = np.zeros((13, n))  # motif frequency

    A = binarize(A, copy=True)  # ensure A is binary
    As = np.logical_or(A, A.T)  # symmetrized adjmat

    for u in range(n - 2):
        # v1: neighbors of u (>u)
        V1 = np.append(np.zeros((u,), dtype=int), As[u, u + 1:n + 1])
        for v1 in np.where(V1)[0]:
            # v2: neighbors of v1 (>u)
            V2 = np.append(np.zeros((u,), dtype=int), As[v1, u + 1:n + 1])
            V2[V1] = 0  # not already in V1
            # and all neighbors of u (>v1)
            V2 = np.logical_or(
                np.append(np.zeros((v1,)), As[u, v1 + 1:n + 1]), V2)
            for v2 in np.where(V2)[0]:
                a = np.array((A[v1, u], A[v2, u], A[u, v1],
                              A[v2, v1], A[u, v2], A[v1, 2]))
                # find all contained isomorphs
                ix = (np.dot(m3, a) == n3)
                id = id3[ix] - 1

                # unique motif occurrences
                idu, jx = np.unique(id, return_index=True)
                jx = np.append((0,), jx + 1)

                mu = len(idu)  # number of unique motifs
                f2 = np.zeros((mu,))
                for h in range(mu):  # for each unique motif
                    f2[h] = jx[h + 1] - jx[h]  # and frequencies

                # then add to a cumulative count
                f[idu] += f2
                # numpy indexing is teh sucks :(
                F[idu, u] += f2
                F[idu, v1] += f2
                F[idu, v2] += f2

    return f, F


def motif3funct_wei(W):
    '''
    Functional motifs are subsets of connection patterns embedded within
    anatomical motifs. Motif frequency is the frequency of occurrence of
    motifs around a node. Motif intensity and coherence are weighted
    generalizations of motif frequency.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted directed connection matrix (all weights between 0 and 1)

    Returns
    -------
    I : 13xN np.ndarray
        motif intensity matrix
    Q : 13xN np.ndarray
        motif coherence matrix
    F : 13xN np.ndarray
        motif frequency matrix

    Notes
    -----
    Average intensity and coherence are given by I./F and Q./F.
    '''
    from scipy import io
    import os
    fname = os.path.join(os.path.dirname(__file__), motiflib)
    mot = io.loadmat(fname)
    m3 = mot['m3']
    id3 = mot['id3'].squeeze()
    n3 = mot['n3'].squeeze()

    n = len(W)
    I = np.zeros((13, n))  # intensity
    Q = np.zeros((13, n))  # coherence
    F = np.zeros((13, n))  # frequency

    A = binarize(W, copy=True)  # create binary adjmat
    As = np.logical_or(A, A.T)  # symmetrized adjmat

    for u in range(n - 2):
        # v1: neighbors of u (>u)
        V1 = np.append(np.zeros((u,), dtype=int), As[u, u + 1:n + 1])
        for v1 in np.where(V1)[0]:
            # v2: neighbors of v1 (>u)
            V2 = np.append(np.zeros((u,), dtype=int), As[v1, u + 1:n + 1])
            V2[V1] = 0  # not already in V1
            # and all neighbors of u (>v1)
            V2 = np.logical_or(
                np.append(np.zeros((v1,)), As[u, v1 + 1:n + 1]), V2)
            for v2 in np.where(V2)[0]:
                a = np.array((A[v1, u], A[v2, u], A[u, v1],
                              A[v2, v1], A[u, v2], A[v1, v2]))
                ix = (np.dot(m3, a) == n3)
                m = np.sum(ix)

                w = np.array((W[v1, u], W[v2, u], W[u, v1],
                              W[v2, v1], W[u, v2], W[v1, v2]))

                M = m3[ix, :] * np.tile(w, (m, 1))
                id = id3[ix] - 1
                l = n3[ix]
                x = np.sum(M, axis=1) / l  # arithmetic mean
                M[M == 0] = 1  # enable geometric mean
                i = np.prod(M, axis=1)**(1 / l)  # intensity
                q = i / x  # coherence

                # unique motif occurrences
                idu, jx = np.unique(id, return_index=True)
                jx = np.append((0,), jx + 1)

                mu = len(idu)  # number of unique motifs
                i2, q2, f2 = np.zeros((3, mu))

                for h in range(mu):
                    i2[h] = np.sum(i[jx[h] + 1:jx[h + 1] + 1])
                    q2[h] = np.sum(q[jx[h] + 1:jx[h + 1] + 1])
                    f2[h] = jx[h + 1] - jx[h]

                # then add to cumulative count
                I[idu, u] += i2
                I[idu, v1] += i2
                I[idu, v2] += i2
                Q[idu, u] += q2
                Q[idu, v1] += q2
                Q[idu, v2] += q2
                F[idu, u] += f2
                F[idu, v1] += f2
                F[idu, v2] += f2

    return I, Q, F


def motif3struct_bin(A):
    '''
    Structural motifs are patterns of local connectivity. Motif frequency
    is the frequency of occurrence of motifs around a node.

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    F : 13xN np.ndarray
        motif frequency matrix
    f : 13x1 np.ndarray
        motif frequency vector (averaged over all nodes)
    '''
    from scipy import io
    import os
    fname = os.path.join(os.path.dirname(__file__), motiflib)
    mot = io.loadmat(fname)
    m3n = mot['m3n']
    id3 = mot['id3'].squeeze()

    n = len(A)  # number of vertices in A
    f = np.zeros((13,))  # motif count for whole graph
    F = np.zeros((13, n))  # motif frequency

    A = binarize(A, copy=True)  # ensure A is binary
    As = np.logical_or(A, A.T)  # symmetrized adjmat

    for u in range(n - 2):
        # v1: neighbors of u (>u)
        V1 = np.append(np.zeros((u,), dtype=int), As[u, u + 1:n + 1])
        for v1 in np.where(V1)[0]:
            # v2: neighbors of v1 (>u)
            V2 = np.append(np.zeros((u,), dtype=int), As[v1, u + 1:n + 1])
            V2[V1] = 0  # not already in V1
            # and all neighbors of u (>v1)
            V2 = np.logical_or(
                np.append(np.zeros((v1,)), As[u, v1 + 1:n + 1]), V2)
            for v2 in np.where(V2)[0]:
                a = np.array((A[v1, u], A[v2, u], A[u, v1],
                              A[v2, v1], A[u, v2], A[v1, v2]))
                s = np.uint32(np.sum(np.power(10, np.arange(5, -1, -1)) * a))
                ix = id3[np.squeeze(s == m3n)] - 1
                F[ix, u] += 1
                F[ix, v1] += 1
                F[ix, v2] += 1
                f[ix] += 1

    return f, F


def motif3struct_wei(W):
    '''
    Structural motifs are patterns of local connectivity. Motif frequency
    is the frequency of occurrence of motifs around a node. Motif intensity
    and coherence are weighted generalizations of motif frequency.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted directed connection matrix (all weights between 0 and 1)

    Returns
    -------
    I : 13xN np.ndarray
        motif intensity matrix
    Q : 13xN np.ndarray
        motif coherence matrix
    F : 13xN np.ndarray
        motif frequency matrix

    Notes
    -----
    Average intensity and coherence are given by I./F and Q./F.
    '''
    from scipy import io
    import os
    fname = os.path.join(os.path.dirname(__file__), motiflib)
    mot = io.loadmat(fname)
    m3 = mot['m3']
    m3n = mot['m3n']
    id3 = mot['id3'].squeeze()
    n3 = mot['n3'].squeeze()

    n = len(W)  # number of vertices in W
    I = np.zeros((13, n))  # intensity
    Q = np.zeros((13, n))  # coherence
    F = np.zeros((13, n))  # frequency

    A = binarize(W, copy=True)  # create binary adjmat
    As = np.logical_or(A, A.T)  # symmetrized adjmat

    for u in range(n - 2):
        # v1: neighbors of u (>u)
        V1 = np.append(np.zeros((u,), dtype=int), As[u, u + 1:n + 1])
        for v1 in np.where(V1)[0]:
            # v2: neighbors of v1 (>u)
            V2 = np.append(np.zeros((u,), dtype=int), As[v1, u + 1:n + 1])
            V2[V1] = 0  # not already in V1
            # and all neighbors of u (>v1)
            V2 = np.logical_or(
                np.append(np.zeros((v1,)), As[u, v1 + 1:n + 1]), V2)
            for v2 in np.where(V2)[0]:
                a = np.array((A[v1, u], A[v2, u], A[u, v1],
                              A[v2, v1], A[u, v2], A[v1, 2]))
                s = np.uint32(np.sum(np.power(10, np.arange(5, -1, -1)) * a))
                ix = np.squeeze(s == m3n)

                w = np.array((W[v1, u], W[v2, u], W[u, v1],
                              W[v2, v1], W[u, v2], W[v1, v2]))

                M = w * m3[ix, :]
                id = id3[ix] - 1
                l = n3[ix]
                x = np.sum(M, axis=1) / l  # arithmetic mean
                M[M == 0] = 1  # enable geometric mean
                i = np.prod(M, axis=1)**(1 / l)  # intensity
                q = i / x  # coherence

                # add to cumulative counts
                I[id, u] += i
                I[id, v1] += i
                I[id, v2] += i
                Q[id, u] += q
                Q[id, v1] += q
                Q[id, v2] += q
                F[id, u] += 1
                F[id, v1] += 1
                F[id, v1] += 1

    return I, Q, F


def motif4funct_bin(A):
    '''
    Functional motifs are subsets of connection patterns embedded within
    anatomical motifs. Motif frequency is the frequency of occurrence of
    motifs around a node.

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    F : 199xN np.ndarray
        motif frequency matrix
    f : 199x1 np.ndarray
        motif frequency vector (averaged over all nodes)
    '''
    from scipy import io
    import os
    fname = os.path.join(os.path.dirname(__file__), motiflib)
    mot = io.loadmat(fname)
    m4 = mot['m4']
    id4 = mot['id4'].squeeze()
    n4 = mot['n4'].squeeze()

    n = len(A)
    f = np.zeros((199,))
    F = np.zeros((199, n))  # frequency

    A = binarize(A, copy=True)  # ensure A is binary
    As = np.logical_or(A, A.T)  # symmetrized adjmat

    for u in range(n - 3):
        # v1: neighbors of u (>u)
        V1 = np.append(np.zeros((u,), dtype=int), As[u, u + 1:n + 1])
        for v1 in np.where(V1)[0]:
            V2 = np.append(np.zeros((u,), dtype=int), As[v1, u + 1:n + 1])
            V2[V1] = 0  # not already in V1
            # and all neighbors of u (>v1)
            V2 = np.logical_or(
                np.append(np.zeros((v1,)), As[u, v1 + 1:n + 1]), V2)
            for v2 in np.where(V2)[0]:
                vz = np.max((v1, v2))  # vz: largest rank node
                # v3: all neighbors of v2 (>u)
                V3 = np.append(np.zeros((u,), dtype=int), As[v2, u + 1:n + 1])
                V3[V2] = 0  # not already in V1 and V2
                # and all neighbors of v1 (>v2)
                V3 = np.logical_or(
                    np.append(np.zeros((v2,)), As[v1, v2 + 1:n + 1]), V3)
                V3[V1] = 0  # not already in V1
                # and all neighbors of u (>vz)
                V3 = np.logical_or(
                    np.append(np.zeros((vz,)), As[u, vz + 1:n + 1]), V3)
                for v3 in np.where(V3)[0]:
                    a = np.array((A[v1, u], A[v2, u], A[v3, u], A[u, v1], A[v2, v1],
                                  A[v3, v1], A[u, v2], A[v1, v2], A[
                                      v3, v2], A[u, v3], A[v1, v3],
                                  A[v2, v3]))

                    ix = (np.dot(m4, a) == n4)  # find all contained isomorphs
                    id = id4[ix] - 1

                    # unique motif occurrences
                    idu, jx = np.unique(id, return_index=True)
                    jx = np.append((0,), jx)
                    mu = len(idu)  # number of unique motifs
                    f2 = np.zeros((mu,))
                    for h in range(mu):
                        f2[h] = jx[h + 1] - jx[h]

                    # add to cumulative count
                    f[idu] += f2
                    F[idu, u] += f2
                    F[idu, v1] += f2
                    F[idu, v2] += f2
                    F[idu, v3] += f2

    return f, F


def motif4funct_wei(W):
    '''
    Functional motifs are subsets of connection patterns embedded within
    anatomical motifs. Motif frequency is the frequency of occurrence of
    motifs around a node. Motif intensity and coherence are weighted
    generalizations of motif frequency.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted directed connection matrix (all weights between 0 and 1)

    Returns
    -------
    I : 199xN np.ndarray
        motif intensity matrix
    Q : 199xN np.ndarray
        motif coherence matrix
    F : 199xN np.ndarray
        motif frequency matrix

    Notes
    -----
    Average intensity and coherence are given by I./F and Q./F.
    '''
    from scipy import io
    import os
    fname = os.path.join(os.path.dirname(__file__), motiflib)
    mot = io.loadmat(fname)
    m4 = mot['m4']
    id4 = mot['id4'].squeeze()
    n4 = mot['n4'].squeeze()

    n = len(W)
    I = np.zeros((199, n))  # intensity
    Q = np.zeros((199, n))  # coherence
    F = np.zeros((199, n))  # frequency

    A = binarize(W, copy=True)  # ensure A is binary
    As = np.logical_or(A, A.T)  # symmetrized adjmat

    for u in range(n - 3):
        # v1: neighbors of u (>u)
        V1 = np.append(np.zeros((u,), dtype=int), As[u, u + 1:n + 1])
        for v1 in np.where(V1)[0]:
            V2 = np.append(np.zeros((u,), dtype=int), As[v1, u + 1:n + 1])
            V2[V1] = 0  # not already in V1
            # and all neighbors of u (>v1)
            V2 = np.logical_or(
                np.append(np.zeros((v1,)), As[u, v1 + 1:n + 1]), V2)
            for v2 in np.where(V2)[0]:
                vz = np.max((v1, v2))  # vz: largest rank node
                # v3: all neighbors of v2 (>u)
                V3 = np.append(np.zeros((u,), dtype=int), As[v2, u + 1:n + 1])
                V3[V2] = 0  # not already in V1 and V2
                # and all neighbors of v1 (>v2)
                V3 = np.logical_or(
                    np.append(np.zeros((v2,)), As[v1, v2 + 1:n + 1]), V3)
                V3[V1] = 0  # not already in V1
                # and all neighbors of u (>vz)
                V3 = np.logical_or(
                    np.append(np.zeros((vz,)), As[u, vz + 1:n + 1]), V3)
                for v3 in np.where(V3)[0]:
                    a = np.array((A[v1, u], A[v2, u], A[v3, u], A[u, v1], A[v2, v1],
                                  A[v3, v1], A[u, v2], A[v1, v2], A[
                                      v3, v2], A[u, v3], A[v1, v3],
                                  A[v2, v3]))
                    ix = (np.dot(m4, a) == n4)  # find all contained isomorphs

                    w = np.array((W[v1, u], W[v2, u], W[v3, u], W[u, v1], W[v2, v1],
                                  W[v3, v1], W[u, v2], W[v1, v2], W[
                                      v3, v2], W[u, v3], W[v1, v3],
                                  W[v2, v3]))

                    m = np.sum(ix)
                    M = m4[ix, :] * np.tile(w, (m, 1))
                    id = id4[ix] - 1
                    l = n4[ix]
                    x = np.sum(M, axis=1) / l  # arithmetic mean
                    M[M == 0] = 1  # enable geometric mean
                    i = np.prod(M, axis=1)**(1 / l)  # intensity
                    q = i / x  # coherence

                    # unique motif occurrences
                    idu, jx = np.unique(id, return_index=True)
                    jx = np.append((0,), jx + 1)

                    mu = len(idu)  # number of unique motifs
                    i2, q2, f2 = np.zeros((3, mu))

                    for h in range(mu):
                        i2[h] = np.sum(i[jx[h] + 1:jx[h + 1] + 1])
                        q2[h] = np.sum(q[jx[h] + 1:jx[h + 1] + 1])
                        f2[h] = jx[h + 1] - jx[h]

                    # then add to cumulative count
                    I[idu, u] += i2
                    I[idu, v1] += i2
                    I[idu, v2] += i2
                    I[idu, v3] += i2
                    Q[idu, u] += q2
                    Q[idu, v1] += q2
                    Q[idu, v2] += q2
                    Q[idu, v3] += q2
                    F[idu, u] += f2
                    F[idu, v1] += f2
                    F[idu, v2] += f2
                    F[idu, v3] += f2

    return I, Q, F


def motif4struct_bin(A):
    '''
    Structural motifs are patterns of local connectivity. Motif frequency
    is the frequency of occurrence of motifs around a node.

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    F : 199xN np.ndarray
        motif frequency matrix
    f : 199x1 np.ndarray
        motif frequency vector (averaged over all nodes)
    '''
    from scipy import io
    import os
    fname = os.path.join(os.path.dirname(__file__), motiflib)
    mot = io.loadmat(fname)
    m4n = mot['m4n']
    id4 = mot['id4'].squeeze()

    n = len(A)
    f = np.zeros((199,))
    F = np.zeros((199, n))  # frequency

    A = binarize(A, copy=True)  # ensure A is binary
    As = np.logical_or(A, A.T)  # symmetrized adjmat

    for u in range(n - 3):
        # v1: neighbors of u (>u)
        V1 = np.append(np.zeros((u,), dtype=int), As[u, u + 1:n + 1])
        for v1 in np.where(V1)[0]:
            V2 = np.append(np.zeros((u,), dtype=int), As[v1, u + 1:n + 1])
            V2[V1] = 0  # not already in V1
            # and all neighbors of u (>v1)
            V2 = np.logical_or(
                np.append(np.zeros((v1,)), As[u, v1 + 1:n + 1]), V2)
            for v2 in np.where(V2)[0]:
                vz = np.max((v1, v2))  # vz: largest rank node
                # v3: all neighbors of v2 (>u)
                V3 = np.append(np.zeros((u,), dtype=int), As[v2, u + 1:n + 1])
                V3[V2] = 0  # not already in V1 and V2
                # and all neighbors of v1 (>v2)
                V3 = np.logical_or(
                    np.append(np.zeros((v2,)), As[v1, v2 + 1:n + 1]), V3)
                V3[V1] = 0  # not already in V1
                # and all neighbors of u (>vz)
                V3 = np.logical_or(
                    np.append(np.zeros((vz,)), As[u, vz + 1:n + 1]), V3)
                for v3 in np.where(V3)[0]:

                    a = np.array((A[v1, u], A[v2, u], A[v3, u], A[u, v1], A[v2, v1],
                                  A[v3, v1], A[u, v2], A[v1, v2], A[
                                      v3, v2], A[u, v3], A[v1, v3],
                                  A[v2, v3]))

                    s = np.uint64(
                        np.sum(np.power(10, np.arange(11, -1, -1)) * a))
                    ix = id4[np.squeeze(s == m4n)]
                    F[ix, u] += 1
                    F[ix, v1] += 1
                    F[ix, v2] += 1
                    F[ix, v3] += 1
                    f[ix] += 1

    return f, F


def motif4struct_wei(W):
    '''
    Structural motifs are patterns of local connectivity. Motif frequency
    is the frequency of occurrence of motifs around a node. Motif intensity
    and coherence are weighted generalizations of motif frequency.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted directed connection matrix (all weights between 0 and 1)

    Returns
    -------
    I : 199xN np.ndarray
        motif intensity matrix
    Q : 199xN np.ndarray
        motif coherence matrix
    F : 199xN np.ndarray
        motif frequency matrix

    Notes
    -----
    Average intensity and coherence are given by I./F and Q./F.
    '''
    from scipy import io
    import os
    fname = os.path.join(os.path.dirname(__file__), motiflib)
    mot = io.loadmat(fname)
    m4 = mot['m4']
    m4n = mot['m4n']
    id4 = mot['id4'].squeeze()
    n4 = mot['n4'].squeeze()

    n = len(W)
    I = np.zeros((199, n))  # intensity
    Q = np.zeros((199, n))  # coherence
    F = np.zeros((199, n))  # frequency

    A = binarize(W, copy=True)  # ensure A is binary
    As = np.logical_or(A, A.T)  # symmetrized adjmat

    for u in range(n - 3):
        # v1: neighbors of u (>u)
        V1 = np.append(np.zeros((u,), dtype=int), As[u, u + 1:n + 1])
        for v1 in np.where(V1)[0]:
            V2 = np.append(np.zeros((u,), dtype=int), As[v1, u + 1:n + 1])
            V2[V1] = 0  # not already in V1
            # and all neighbors of u (>v1)
            V2 = np.logical_or(
                np.append(np.zeros((v1,)), As[u, v1 + 1:n + 1]), V2)
            for v2 in np.where(V2)[0]:
                vz = np.max((v1, v2))  # vz: largest rank node
                # v3: all neighbors of v2 (>u)
                V3 = np.append(np.zeros((u,), dtype=int), As[v2, u + 1:n + 1])
                V3[V2] = 0  # not already in V1 and V2
                # and all neighbors of v1 (>v2)
                V3 = np.logical_or(
                    np.append(np.zeros((v2,)), As[v1, v2 + 1:n + 1]), V3)
                V3[V1] = 0  # not already in V1
                # and all neighbors of u (>vz)
                V3 = np.logical_or(
                    np.append(np.zeros((vz,)), As[u, vz + 1:n + 1]), V3)
                for v3 in np.where(V3)[0]:
                    a = np.array((A[v1, u], A[v2, u], A[v3, u], A[u, v1], A[v2, v1],
                                  A[v3, v1], A[u, v2], A[v1, v2], A[
                                      v3, v2], A[u, v3], A[v1, v3],
                                  A[v2, v3]))
                    s = np.uint64(
                        np.sum(np.power(10, np.arange(11, -1, -1)) * a))
                    # print np.shape(s),np.shape(m4n)
                    ix = np.squeeze(s == m4n)

                    w = np.array((W[v1, u], W[v2, u], W[v3, u], W[u, v1], W[v2, v1],
                                  W[v3, v1], W[u, v2], W[v1, v2], W[
                                      v3, v2], W[u, v3], W[v1, v3],
                                  W[v2, v3]))

                    M = w * m4[ix, :]
                    id = id4[ix] - 1
                    l = n4[ix]
                    x = np.sum(M, axis=1) / l  # arithmetic mean
                    M[M == 0] = 1  # enable geometric mean
                    i = np.prod(M, axis=1)**(1 / l)  # intensity
                    q = i / x  # coherence

                    # then add to cumulative count
                    I[id, u] += i
                    I[id, v1] += i
                    I[id, v2] += i
                    I[id, v3] += i
                    Q[id, u] += q
                    Q[id, v1] += q
                    Q[id, v2] += q
                    Q[id, v3] += q
                    F[id, u] += 1
                    F[id, v1] += 1
                    F[id, v2] += 1
                    F[id, v3] += 1

    return I, Q, F
