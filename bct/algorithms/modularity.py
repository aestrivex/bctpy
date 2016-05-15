from __future__ import division, print_function
import numpy as np
from bct.utils import BCTParamError, normalize


def ci2ls(ci):
    '''
    Convert from a community index vector to a 2D python list of modules
    The list is a pure python list, not requiring numpy.

    Parameters
    ----------
    ci : Nx1 np.ndarray
        the community index vector
    zeroindexed : bool
        If True, ci uses zero-indexing (lowest value is 0). Defaults to False.

    Returns
    -------
    ls : listof(list)
        pure python list with lowest value zero-indexed
        (regardless of zero-indexing parameter)
    '''
    if not np.size(ci):
        return ci  # list is empty
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    nr_indices = int(max(ci))
    ls = []
    for c in range(nr_indices):
        ls.append([])
    for i, x in enumerate(ci):
        ls[ci[i] - 1].append(i)
    return ls


def ls2ci(ls, zeroindexed=False):
    '''
    Convert from a 2D python list of modules to a community index vector.
    The list is a pure python list, not requiring numpy.

    Parameters
    ----------
    ls : listof(list)
        pure python list with lowest value zero-indexed
        (regardless of value of zeroindexed parameter)
    zeroindexed : bool
        If True, ci uses zero-indexing (lowest value is 0). Defaults to False.

    Returns
    -------
    ci : Nx1 np.ndarray
        community index vector
    '''
    if ls is None or np.size(ls) == 0:
        return ()  # list is empty
    nr_indices = sum(map(len, ls))
    ci = np.zeros((nr_indices,), dtype=int)
    z = int(not zeroindexed)
    for i, x in enumerate(ls):
        for j, y in enumerate(ls[i]):
            ci[ls[i][j]] = i + z
    return ci


def community_louvain(W, gamma=1, ci=None, B='modularity', seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes which maximizes the number of within-group
    edges and minimizes the number of between-group edges.

    This function is a fast an accurate multi-iterative generalization of the
    louvain community detection algorithm. This function subsumes and improves
    upon modularity_[louvain,finetune]_[und,dir]() and additionally allows to
    optimize other objective functions (includes built-in Potts Model i
    Hamiltonian, allows for custom objective-function matrices).

    Parameters
    ----------
    W : NxN np.array
        directed/undirected weighted/binary adjacency matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
        ignored if an objective function matrix is specified.
    ci : Nx1 np.arraylike
        initial community affiliation vector. default value=None
    B : str | NxN np.arraylike
        string describing objective function type, or provides a custom
        NxN objective-function matrix. builtin values 
            'modularity' uses Q-metric as objective function
            'potts' uses Potts model Hamiltonian.
            'negative_sym' symmetric treatment of negative weights
            'negative_asym' asymmetric treatment of negative weights
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.

    Returns
    -------
    ci : Nx1 np.array
        final community structure
    q : float
        optimized q-statistic (modularity only)
    '''
    np.random.seed(seed)

    n = len(W)
    s = np.sum(W)

    if np.min(W) < -1e-10:
        raise BCTParamError('adjmat must not contain negative weights')

    if ci is None:
        ci = np.arange(n) + 1
    else:
        if len(ci) != n:
            raise BCTParamError('initial ci vector size must equal N')
        _, ci = np.unique(ci, return_inverse=True)
        ci += 1
    Mb = ci.copy()

    if B in ('negative_sym', 'negative_asym'):
        W0 = W * (W > 0)
        s0 = np.sum(W0)
        B0 = W0 - gamma * np.outer(np.sum(W0, axis=1), np.sum(W, axis=0)) / s0

        W1 = W * (W < 0)
        s1 = np.sum(W1)
        if s1:
            B1 = (W1 - gamma * np.outer(np.sum(W1, axis=1), np.sum(W1, axis=0))
                / s1)
        else:
            B1 = 0

    elif np.min(W) < -1e-10:
        raise BCTParamError("Input connection matrix contains negative "
            'weights but objective function dealing with negative weights '
            'was not selected')

    if B == 'potts' and np.any(np.logical_not(np.logical_or(W == 0, W == 1))):
        raise BCTParamError('Potts hamiltonian requires binary input matrix')

    if B == 'modularity':
        B = W - gamma * np.outer(np.sum(W, axis=1), np.sum(W, axis=0)) / s
    elif B == 'potts':
        B = W - gamma * np.logical_not(W)
    elif B == 'negative_sym':
        B = B0 / (s0 + s1) - B1 / (s0 + s1)
    elif B == 'negative_asym':
        B = B0 / s0 - B1 / (s0 + s1)
    else:
        try:
            B = np.array(B)
        except:
            raise BCTParamError('unknown objective function type')

        if B.shape != W.shape:
            raise BCTParamError('objective function matrix does not match '
                                'size of adjacency matrix')
        if not np.allclose(B, B.T):
            print ('Warning: objective function matrix not symmetric, '
                   'symmetrizing')
            B = (B + B.T) / 2

    Hnm = np.zeros((n, n))
    for m in range(1, n + 1):
        Hnm[:, m - 1] = np.sum(B[:, ci == m], axis=1)  # node to module degree
    H = np.sum(Hnm, axis=1)  # node degree
    Hm = np.sum(Hnm, axis=0)  # module degree

    q0 = -np.inf
    # compute modularity
    q = np.sum(B[np.tile(ci, (n, 1)) == np.tile(ci, (n, 1)).T]) / s

    first_iteration = True

    while q - q0 > 1e-10:
        it = 0
        flag = True
        while flag:
            it += 1
            if it > 1000:
                raise BCTParamError('Modularity infinite loop style G. '
                                    'Please contact the developer.')
            flag = False
            for u in np.random.permutation(n):
                ma = Mb[u] - 1
                dQ = Hnm[u, :] - Hnm[u, ma] + B[u, u]  # algorithm condition
                dQ[ma] = 0

                max_dq = np.max(dQ)
                if max_dq > 1e-10:
                    flag = True
                    mb = np.argmax(dQ)

                    Hnm[:, mb] += B[:, u]
                    Hnm[:, ma] -= B[:, u]  # change node-to-module strengths

                    Hm[mb] += H[u]
                    Hm[ma] -= H[u]  # change module strengths

                    Mb[u] = mb + 1

        _, Mb = np.unique(Mb, return_inverse=True)
        Mb += 1

        M0 = ci.copy()
        if first_iteration:
            ci = Mb.copy()
            first_iteration = False
        else:
            for u in range(1, n + 1):
                ci[M0 == u] = Mb[u - 1]  # assign new modules

        n = np.max(Mb)
        b1 = np.zeros((n, n))
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                # pool weights of nodes in same module
                bm = np.sum(B[np.ix_(Mb == i, Mb == j)])
                b1[i - 1, j - 1] = bm
                b1[j - 1, i - 1] = bm
        B = b1.copy()

        Mb = np.arange(1, n + 1)
        Hnm = B.copy()
        H = np.sum(B, axis=0)
        Hm = H.copy()

        q0 = q
        q = np.trace(B) / s  # compute modularity

    return ci, q


def link_communities(W, type_clustering='single'):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes which maximizes the number of within-group
    edges and minimizes the number of between-group edges.

    This algorithm uncovers overlapping community structure via hierarchical
    clustering of network links. This algorithm is generalized for
    weighted/directed/fully-connected networks

    Parameters
    ----------
    W : NxN np.array
        directed weighted/binary adjacency matrix
    type_clustering : str
        type of hierarchical clustering. 'single' for single-linkage,
        'complete' for complete-linkage. Default value='single'

    Returns
    -------
    M : CxN np.ndarray
        nodal community affiliation matrix.
    '''
    n = len(W)
    W = normalize(W)

    if type_clustering not in ('single', 'complete'):
        raise BCTParamError('Unrecognized clustering type')

    # set diagonal to mean weights
    np.fill_diagonal(W, 0)
    W[range(n), range(n)] = (
        np.sum(W, axis=0) / np.sum(np.logical_not(W), axis=0) +
        np.sum(W.T, axis=0) / np.sum(np.logical_not(W.T), axis=0)) / 2

    # out/in norm squared
    No = np.sum(W**2, axis=1)
    Ni = np.sum(W**2, axis=0)

    # weighted in/out jaccard
    Jo = np.zeros((n, n))
    Ji = np.zeros((n, n))

    for b in range(n):
        for c in range(n):
            Do = np.dot(W[b, :], W[c, :].T)
            Jo[b, c] = Do / (No[b] + No[c] - Do)

            Di = np.dot(W[:, b].T, W[:, c])
            Ji[b, c] = Di / (Ni[b] + Ni[c] - Di)

    # get link similarity
    A, B = np.where(np.logical_and(np.logical_or(W, W.T),
                                   np.triu(np.ones((n, n)), 1)))
    m = len(A)
    Ln = np.zeros((m, 2), dtype=np.int32)  # link nodes
    Lw = np.zeros((m,))  # link weights

    for i in range(m):
        Ln[i, :] = (A[i], B[i])
        Lw[i] = (W[A[i], B[i]] + W[B[i], A[i]]) / 2

    ES = np.zeros((m, m), dtype=np.float32)  # link similarity
    for i in range(m):
        for j in range(m):
            if Ln[i, 0] == Ln[j, 0]:
                a = Ln[i, 0]
                b = Ln[i, 1]
                c = Ln[j, 1]
            elif Ln[i, 0] == Ln[j, 1]:
                a = Ln[i, 0]
                b = Ln[i, 1]
                c = Ln[j, 0]
            elif Ln[i, 1] == Ln[j, 0]:
                a = Ln[i, 1]
                b = Ln[i, 0]
                c = Ln[j, 1]
            elif Ln[i, 1] == Ln[j, 1]:
                a = Ln[i, 1]
                b = Ln[i, 0]
                c = Ln[j, 0]
            else:
                continue

            ES[i, j] = (W[a, b] * W[a, c] * Ji[b, c] +
                        W[b, a] * W[c, a] * Jo[b, c]) / 2

    np.fill_diagonal(ES, 0)

    # perform hierarchical clustering

    C = np.zeros((m, m), dtype=np.int32)  # community affiliation matrix

    Nc = C.copy()
    Mc = np.zeros((m, m), dtype=np.float32)
    Dc = Mc.copy()  # community nodes, links, density

    U = np.arange(m)  # initial community assignments
    C[0, :] = np.arange(m)

    import time

    for i in range(m - 1):
        print('hierarchy %i' % i)

        #time1 = time.time()

        for j in range(len(U)):  # loop over communities
            ixes = C[i, :] == U[j]  # get link indices

            links = np.sort(Lw[ixes])
            #nodes = np.sort(Ln[ixes,:].flat)

            nodes = np.sort(np.reshape(
                Ln[ixes, :], 2 * np.size(np.where(ixes))))

            # get unique nodes
            nodulo = np.append(nodes[0], (nodes[1:])[nodes[1:] != nodes[:-1]])
            #nodulo = ((nodes[1:])[nodes[1:] != nodes[:-1]])

            nc = len(nodulo)
            #nc = len(nodulo)+1
            mc = np.sum(links)
            min_mc = np.sum(links[:nc - 1])  # minimal weight
            dc = (mc - min_mc) / (nc * (nc - 1) /
                                  2 - min_mc)  # community density

            if np.array(dc).shape is not ():
                print(dc)
                print(dc.shape)

            Nc[i, j] = nc
            Mc[i, j] = mc
            Dc[i, j] = dc if not np.isnan(dc) else 0

        #time2 = time.time()
        #print('compute densities time', time2-time1)

        C[i + 1, :] = C[i, :]  # copy current partition

        #if i in (2693,):
        #    import pdb
        #    pdb.set_trace()

        # Profiling and debugging show that this line, finding
        # the max values in this matrix, take about 3x longer than the
        # corresponding matlab version. Can it be improved?

        u1, u2 = np.where(ES[np.ix_(U, U)] == np.max(ES[np.ix_(U, U)]))

        if np.size(u1) > 2:
            # pick one
            wehr, = np.where((u1 == u2[0]))

            uc = np.squeeze((u1[0], u2[0]))
            ud = np.squeeze((u1[wehr], u2[wehr]))

            u1 = uc
            u2 = ud

        #time25 = time.time()
        #print('copy and max time', time25-time2)

        # get unique links (implementation of matlab sortrows)
        #ugl = np.array((u1,u2))
        ugl = np.sort((u1, u2), axis=1)
        ug_rows = ugl[np.argsort(ugl, axis=0)[:, 0]]
        # implementation of matlab unique(A, 'rows')
        unq_rows = np.vstack({tuple(row) for row in ug_rows})
        V = U[unq_rows]

        #time3 = time.time()
        #print('sortrows time', time3-time25)

        for j in range(len(V)):
            if type_clustering == 'single':
                x = np.max(ES[V[j, :], :], axis=0)
            elif type_clustering == 'complete':
                x = np.min(ES[V[j, :], :], axis=0)

            # assign distances to whole clusters
#            import pdb
#            pdb.set_trace()
            ES[V[j, :], :] = np.array((x, x))
            ES[:, V[j, :]] = np.transpose((x, x))

            # clear diagonal
            ES[V[j, 0], V[j, 0]] = 0
            ES[V[j, 1], V[j, 1]] = 0

            # merge communities
            C[i + 1, C[i + 1, :] == V[j, 1]] = V[j, 0]
            V[V == V[j, 1]] = V[j, 0]

        #time4 = time.time()
        #print('get linkages time', time4-time3)

        U = np.unique(C[i + 1, :])
        if len(U) == 1:
            break

        #time5 = time.time()
        #print('get unique communities time', time5-time4)

    #ENDT HAIERARKIKL CLUSTRRINNG
    #ENDT HAIERARKIKL CLUSTRRINNG
    #ENDT HAIERARKIKL CLUSTRRINNG
    #ENDT HAIERARKIKL CLUSTRRINNG
    #ENDT HAIERARKIKL CLUSTRRINNG

    #Dc[ np.where(np.isnan(Dc)) ]=0
    i = np.argmax(np.sum(Dc * Mc, axis=1))

    U = np.unique(C[i, :])
    M = np.zeros((len(U), n))
    for j in range(len(U)):
        M[j, np.unique(Ln[C[i, :] == U[j], :])] = 1

    M = M[np.sum(M, axis=1) > 2, :]

    return M


def modularity_dir(A, gamma=1, kci=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.

    Parameters
    ----------
    W : NxN np.ndarray
        directed weighted/binary connection matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    kci : Nx1 np.ndarray | None
        starting community structure. If specified, calculates the Q-metric
        on the community structure giving, without doing any optimzation.
        Otherwise, if not specified, uses a spectral modularity maximization
        algorithm.

    Returns
    -------
    ci : Nx1 np.ndarray
        optimized community structure
    Q : float
        maximized modularity metric

    Notes
    -----
    This algorithm is deterministic. The matlab function bearing this
    name incorrectly disclaims that the outcome depends on heuristics
    involving a random seed. The louvain method does depend on a random seed,
    but this function uses a deterministic modularity maximization algorithm.
    '''
    from scipy import linalg
    n = len(A)  # number of vertices
    ki = np.sum(A, axis=0)  # in degree
    ko = np.sum(A, axis=1)  # out degree
    m = np.sum(ki)  # number of edges
    b = A - gamma * np.outer(ko, ki) / m
    B = b + b.T  # directed modularity matrix

    init_mod = np.arange(n)  # initial one big module
    modules = []  # output modules list

    def recur(module):
        n = len(module)
        modmat = B[module][:, module]

        vals, vecs = linalg.eig(modmat)  # biggest eigendecomposition
        rlvals = np.real(vals)
        max_eigvec = np.squeeze(vecs[:, np.where(rlvals == np.max(rlvals))])
        if max_eigvec.ndim > 1:  # if multiple max eigenvalues, pick one
            max_eigvec = max_eigvec[:, 0]
        # initial module assignments
        mod_asgn = np.squeeze((max_eigvec >= 0) * 2 - 1)
        q = np.dot(mod_asgn, np.dot(modmat, mod_asgn))  # modularity change

        if q > 0:  # change in modularity was positive
            qmax = q
            np.fill_diagonal(modmat, 0)
            it = np.ma.masked_array(np.ones((n,)), False)
            mod_asgn_iter = mod_asgn.copy()
            while np.any(it):  # do some iterative fine tuning
                # this line is linear algebra voodoo
                q_iter = qmax - 4 * mod_asgn_iter * \
                    (np.dot(modmat, mod_asgn_iter))
                qmax = np.max(q_iter * it)
                imax = np.argmax(q_iter * it)
                #imax, = np.where(q_iter == qmax)
                #if len(imax) > 0:
                #    imax = imax[0]
                #    print(imax)
                # does switching increase modularity?
                mod_asgn_iter[imax] *= -1
                it[imax] = np.ma.masked
                if qmax > q:
                    q = qmax
                    mod_asgn = mod_asgn_iter
            if np.abs(np.sum(mod_asgn)) == n:  # iteration yielded null module
                modules.append(np.array(module).tolist())
            else:
                mod1 = module[np.where(mod_asgn == 1)]
                mod2 = module[np.where(mod_asgn == -1)]

                recur(mod1)
                recur(mod2)
        else:  # change in modularity was negative or 0
            modules.append(np.array(module).tolist())

    # adjustment to one-based indexing occurs in ls2ci
    if kci is None:
        recur(init_mod)
        ci = ls2ci(modules)
    else:
        ci = kci
    s = np.tile(ci, (n, 1))
    q = np.sum(np.logical_not(s - s.T) * B / (2 * m))
    return ci, q


def modularity_finetune_dir(W, ci=None, gamma=1, seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.

    This algorithm is inspired by the Kernighan-Lin fine-tuning algorithm
    and is designed to refine a previously detected community structure.

    Parameters
    ----------
    W : NxN np.ndarray
        directed weighted/binary connection matrix
    ci : Nx1 np.ndarray | None
        initial community affiliation vector
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.

    Returns
    -------
    ci : Nx1 np.ndarray
        refined community affiliation vector
    Q : float
        optimized modularity metric

    Notes
    -----
    Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    '''
    np.random.seed(seed)

    n = len(W)  # number of nodes
    if ci is None:
        ci = np.arange(n) + 1
    else:
        _, ci = np.unique(ci, return_inverse=True)
        ci += 1

    s = np.sum(W)  # weight of edges
    knm_o = np.zeros((n, n))  # node-to-module out degree
    knm_i = np.zeros((n, n))  # node-to-module in degree

    for m in range(np.max(ci)):
        knm_o[:, m] = np.sum(W[:, ci == (m + 1)], axis=1)
        knm_i[:, m] = np.sum(W[ci == (m + 1), :], axis=0)

    k_o = np.sum(knm_o, axis=1)  # node out-degree
    k_i = np.sum(knm_i, axis=1)  # node in-degree
    km_o = np.sum(knm_o, axis=0)  # module out-degree
    km_i = np.sum(knm_i, axis=0)  # module out-degree

    flag = True
    while flag:
        flag = False
        for u in np.random.permutation(n):  # loop over nodes in random order
            ma = ci[u] - 1  # current module of u
            # algorithm condition
            dq_o = ((knm_o[u, :] - knm_o[u, ma] + W[u, u]) -
                    gamma * k_o[u] * (km_i - km_i[ma] + k_i[u]) / s)
            dq_i = ((knm_i[u, :] - knm_i[u, ma] + W[u, u]) -
                    gamma * k_i[u] * (km_o - km_o[ma] + k_o[u]) / s)
            dq = (dq_o + dq_i) / 2
            dq[ma] = 0

            max_dq = np.max(dq)  # find maximal modularity increase
            if max_dq > 1e-10:  # if maximal increase positive
                mb = np.argmax(dq)  # take only one value
                # print max_dq,mb

                knm_o[:, mb] += W[u, :].T  # change node-to-module out-degrees
                knm_o[:, ma] -= W[u, :].T
                knm_i[:, mb] += W[:, u]  # change node-to-module in-degrees
                knm_i[:, ma] -= W[:, u]
                km_o[mb] += k_o[u]  # change module out-degrees
                km_o[ma] -= k_o[u]
                km_i[mb] += k_i[u]  # change module in-degrees
                km_i[ma] -= k_i[u]

                ci[u] = mb + 1  # reassign module
                flag = True

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    m = np.max(ci)  # new number of modules
    w = np.zeros((m, m))  # new weighted matrix

    for u in range(m):
        for v in range(m):
            # pool weights of nodes in same module
            w[u, v] = np.sum(W[np.ix_(ci == u + 1, ci == v + 1)])

    q = np.trace(w) / s - gamma * np.sum(np.dot(w / s, w / s))
    return ci, q


def modularity_finetune_und(W, ci=None, gamma=1, seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.

    This algorithm is inspired by the Kernighan-Lin fine-tuning algorithm
    and is designed to refine a previously detected community structure.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted/binary connection matrix
    ci : Nx1 np.ndarray | None
        initial community affiliation vector
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.

    Returns
    -------
    ci : Nx1 np.ndarray
        refined community affiliation vector
    Q : float
        optimized modularity metric

    Notes
    -----
    Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    '''
    np.random.seed(seed)

    #import time
    n = len(W)  # number of nodes
    if ci is None:
        ci = np.arange(n) + 1
    else:
        _, ci = np.unique(ci, return_inverse=True)
        ci += 1

    s = np.sum(W)  # total weight of edges
    knm = np.zeros((n, n))  # node-to-module degree
    for m in range(np.max(ci)):
        knm[:, m] = np.sum(W[:, ci == (m + 1)], axis=1)
    k = np.sum(knm, axis=1)  # node degree
    km = np.sum(knm, axis=0)  # module degree

    flag = True
    while flag:
        flag = False

        for u in np.random.permutation(n):
            # for u in np.arange(n):
            ma = ci[u] - 1
            # time.sleep(1)
            # algorithm condition
            dq = (knm[u, :] - knm[u, ma] + W[u, u]) - \
                gamma * k[u] * (km - km[ma] + k[u]) / s
            # print
            # np.sum(knm[u,:],knm[u,ma],W[u,u],gamma,k[u],np.sum(km),km[ma],k[u],s
            dq[ma] = 0

            max_dq = np.max(dq)  # find maximal modularity increase
            if max_dq > 1e-10:  # if maximal increase positive
                mb = np.argmax(dq)  # take only one value

                # print max_dq, mb

                knm[:, mb] += W[:, u]  # change node-to-module degrees
                knm[:, ma] -= W[:, u]
                km[mb] += k[u]  # change module degrees
                km[ma] -= k[u]

                ci[u] = mb + 1
                flag = True

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    m = np.max(ci)
    w = np.zeros((m, m))
    for u in range(m):
        for v in range(m):
            # pool weights of nodes in same module
            wm = np.sum(W[np.ix_(ci == u + 1, ci == v + 1)])
            w[u, v] = wm
            w[v, u] = wm

    q = np.trace(w) / s - gamma * np.sum(np.dot(w / s, w / s))
    return ci, q


def modularity_finetune_und_sign(W, qtype='sta', gamma=1, ci=None, seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.

    This algorithm is inspired by the Kernighan-Lin fine-tuning algorithm
    and is designed to refine a previously detected community structure.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted/binary connection matrix with positive and
        negative weights.
    qtype : str
        modularity type. Can be 'sta' (default), 'pos', 'smp', 'gja', 'neg'.
        See Rubinov and Sporns (2011) for a description.
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    ci : Nx1 np.ndarray | None
        initial community affiliation vector
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.

    Returns
    -------
    ci : Nx1 np.ndarray
        refined community affiliation vector
    Q : float
        optimized modularity metric

    Notes
    -----
    Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    '''
    np.random.seed(seed)

    n = len(W)  # number of nodes/modules
    if ci is None:
        ci = np.arange(n) + 1
    else:
        _, ci = np.unique(ci, return_inverse=True)
        ci += 1

    W0 = W * (W > 0)  # positive weights matrix
    W1 = -W * (W < 0)  # negative weights matrix
    s0 = np.sum(W0)  # positive sum of weights
    s1 = np.sum(W1)  # negative sum of weights
    Knm0 = np.zeros((n, n))  # positive node-to-module-degree
    Knm1 = np.zeros((n, n))  # negative node-to-module degree

    for m in range(int(np.max(ci))):  # loop over modules
        Knm0[:, m] = np.sum(W0[:, ci == m + 1], axis=1)
        Knm1[:, m] = np.sum(W1[:, ci == m + 1], axis=1)

    Kn0 = np.sum(Knm0, axis=1)  # positive node degree
    Kn1 = np.sum(Knm1, axis=1)  # negative node degree
    Km0 = np.sum(Knm0, axis=0)  # positive module degree
    Km1 = np.sum(Knm1, axis=0)  # negative module degree

    if qtype == 'smp':
        d0 = 1 / s0
        d1 = 1 / s1  # dQ=dQ0/s0-dQ1/s1
    elif qtype == 'gja':
        d0 = 1 / (s0 + s1)
        d1 = 1 / (s0 + s1)  # dQ=(dQ0-dQ1)/(s0+s1)
    elif qtype == 'sta':
        d0 = 1 / s0
        d1 = 1 / (s0 + s1)  # dQ=dQ0/s0-dQ1/(s0+s1)
    elif qtype == 'pos':
        d0 = 1 / s0
        d1 = 0  # dQ=dQ0/s0
    elif qtype == 'neg':
        d0 = 0
        d1 = 1 / s1  # dQ=-dQ1/s1
    else:
        raise KeyError('modularity type unknown')

    if not s0:  # adjust for absent positive weights
        s0 = 1
        d0 = 0
    if not s1:  # adjust for absent negative weights
        s1 = 1
        d1 = 0

    flag = True  # flag for within hierarchy search
    h = 0
    while flag:
        h += 1
        if h > 1000:
            raise BCTParamError('Modularity infinite loop style D')
        flag = False
        for u in np.random.permutation(n):  # loop over nodes in random order
            ma = ci[u] - 1  # current module of u
            dq0 = ((Knm0[u, :] + W0[u, u] - Knm0[u, ma]) -
                   gamma * Kn0[u] * (Km0 + Kn0[u] - Km0[ma]) / s0)
            dq1 = ((Knm1[u, :] + W1[u, u] - Knm1[u, ma]) -
                   gamma * Kn1[u] * (Km1 + Kn1[u] - Km1[ma]) / s1)
            dq = d0 * dq0 - d1 * dq1  # rescaled changes in modularity
            dq[ma] = 0  # no changes for same module

            # print dq,ma,u

            max_dq = np.max(dq)  # maximal increase in modularity
            mb = np.argmax(dq)  # corresponding module
            if max_dq > 1e-10:  # if maximal increase is positive
                # print h,max_dq,mb,u
                flag = True
                ci[u] = mb + 1  # reassign module

                Knm0[:, mb] += W0[:, u]
                Knm0[:, ma] -= W0[:, u]
                Knm1[:, mb] += W1[:, u]
                Knm1[:, ma] -= W1[:, u]
                Km0[mb] += Kn0[u]
                Km0[ma] -= Kn0[u]
                Km1[mb] += Kn1[u]
                Km1[ma] -= Kn1[u]

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    m = np.tile(ci, (n, 1))
    q0 = (W0 - np.outer(Kn0, Kn0) / s0) * (m == m.T)
    q1 = (W1 - np.outer(Kn1, Kn1) / s1) * (m == m.T)
    q = d0 * np.sum(q0) - d1 * np.sum(q1)

    return ci, q


def modularity_louvain_dir(W, gamma=1, hierarchy=False, seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.

    The Louvain algorithm is a fast and accurate community detection
    algorithm (as of writing). The algorithm may also be used to detect
    hierarchical community structure.

    Parameters
    ----------
    W : NxN np.ndarray
        directed weighted/binary connection matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    hierarchy : bool
        Enables hierarchical output. Defalut value=False
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.

    Returns
    -------
    ci : Nx1 np.ndarray
        refined community affiliation vector. If hierarchical output enabled,
        it is an NxH np.ndarray instead with multiple iterations
    Q : float
        optimized modularity metric. If hierarchical output enabled, becomes
        an Hx1 array of floats instead.

    Notes
    -----
    Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    '''
    np.random.seed(seed)

    n = len(W)  # number of nodes
    s = np.sum(W)  # total weight of edges
    h = 0  # hierarchy index
    ci = []
    ci.append(np.arange(n) + 1)  # hierarchical module assignments
    q = []
    q.append(-1)  # hierarchical modularity index
    n0 = n

    while True:
        if h > 300:
            raise BCTParamError('Modularity Infinite Loop Style E.  Please '
                                'contact the developer with this error.')
        k_o = np.sum(W, axis=1)  # node in/out degrees
        k_i = np.sum(W, axis=0)
        km_o = k_o.copy()  # module in/out degrees
        km_i = k_i.copy()
        knm_o = W.copy()  # node-to-module in/out degrees
        knm_i = W.copy()

        m = np.arange(n) + 1  # initial module assignments

        flag = True  # flag for within hierarchy search
        it = 0
        while flag:
            it += 1
            if it > 1000:
                raise BCTParamError('Modularity Infinite Loop Style F.  Please '
                                    'contact the developer with this error.')
            flag = False

            # loop over nodes in random order
            for u in np.random.permutation(n):
                ma = m[u] - 1
                # algorithm condition
                dq_o = ((knm_o[u, :] - knm_o[u, ma] + W[u, u]) -
                        gamma * k_o[u] * (km_i - km_i[ma] + k_i[u]) / s)
                dq_i = ((knm_i[u, :] - knm_i[u, ma] + W[u, u]) -
                        gamma * k_i[u] * (km_o - km_o[ma] + k_o[u]) / s)
                dq = (dq_o + dq_i) / 2
                dq[ma] = 0

                max_dq = np.max(dq)  # find maximal modularity increase
                if max_dq > 1e-10:  # if maximal increase positive
                    mb = np.argmax(dq)  # take only one value

                    knm_o[:, mb] += W[u, :].T  # change node-to-module degrees
                    knm_o[:, ma] -= W[u, :].T
                    knm_i[:, mb] += W[:, u]
                    knm_i[:, ma] -= W[:, u]
                    km_o[mb] += k_o[u]  # change module out-degrees
                    km_o[ma] -= k_o[u]
                    km_i[mb] += k_i[u]
                    km_i[ma] -= k_i[u]

                    m[u] = mb + 1  # reassign module
                    flag = True

        _, m = np.unique(m, return_inverse=True)
        m += 1
        h += 1
        ci.append(np.zeros((n0,)))
        # for i,mi in enumerate(m):		#loop through module assignments
        for i in range(n):
            # ci[h][np.where(ci[h-1]==i)]=mi	#assign new modules
            ci[h][np.where(ci[h - 1] == i + 1)] = m[i]

        n = np.max(m)  # new number of modules
        W1 = np.zeros((n, n))  # new weighted matrix
        for i in range(n):
            for j in range(n):
                # pool weights of nodes in same module
                W1[i, j] = np.sum(W[np.ix_(m == i + 1, m == j + 1)])

        q.append(0)
        # compute modularity
        q[h] = np.trace(W1) / s - gamma * np.sum(np.dot(W1 / s, W1 / s))
        if q[h] - q[h - 1] < 1e-10:  # if modularity does not increase
            break

    ci = np.array(ci, dtype=int)
    if hierarchy:
        ci = ci[1:-1]
        q = q[1:-1]
        return ci, q
    else:
        return ci[h - 1], q[h - 1]


def modularity_louvain_und(W, gamma=1, hierarchy=False, seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.

    The Louvain algorithm is a fast and accurate community detection
    algorithm (as of writing). The algorithm may also be used to detect
    hierarchical community structure.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted/binary connection matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    hierarchy : bool
        Enables hierarchical output. Defalut value=False
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.

    Returns
    -------
    ci : Nx1 np.ndarray
        refined community affiliation vector. If hierarchical output enabled,
        it is an NxH np.ndarray instead with multiple iterations
    Q : float
        optimized modularity metric. If hierarchical output enabled, becomes
        an Hx1 array of floats instead.

    Notes
    -----
    Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    '''
    np.random.seed(seed)

    n = len(W)  # number of nodes
    s = np.sum(W)  # weight of edges
    h = 0  # hierarchy index
    ci = []
    ci.append(np.arange(n) + 1)  # hierarchical module assignments
    q = []
    q.append(-1)  # hierarchical modularity values
    n0 = n

    #knm = np.zeros((n,n))
    # for j in np.xrange(n0+1):
    #    knm[:,j] = np.sum(w[;,

    while True:
        if h > 300:
            raise BCTParamError('Modularity Infinite Loop Style B.  Please '
                                'contact the developer with this error.')
        k = np.sum(W, axis=0)  # node degree
        Km = k.copy()  # module degree
        Knm = W.copy()  # node-to-module degree

        m = np.arange(n) + 1  # initial module assignments

        flag = True  # flag for within-hierarchy search
        it = 0
        while flag:
            it += 1
            if it > 1000:
                raise BCTParamError('Modularity Infinite Loop Style C.  Please '
                                    'contact the developer with this error.')
            flag = False

            # loop over nodes in random order
            for i in np.random.permutation(n):
                ma = m[i] - 1
                # algorithm condition
                dQ = ((Knm[i, :] - Knm[i, ma] + W[i, i]) -
                      gamma * k[i] * (Km - Km[ma] + k[i]) / s)
                dQ[ma] = 0

                max_dq = np.max(dQ)  # find maximal modularity increase
                if max_dq > 1e-10:  # if maximal increase positive
                    j = np.argmax(dQ)  # take only one value
                    # print max_dq,j,dQ[j]

                    Knm[:, j] += W[:, i]  # change node-to-module degrees
                    Knm[:, ma] -= W[:, i]

                    Km[j] += k[i]  # change module degrees
                    Km[ma] -= k[i]

                    m[i] = j + 1  # reassign module
                    flag = True

        _, m = np.unique(m, return_inverse=True)  # new module assignments
        # print m,h
        m += 1
        h += 1
        ci.append(np.zeros((n0,)))
        # for i,mi in enumerate(m):	#loop through initial module assignments
        for i in range(n):
            # print i, m[i], n0, h, len(m), n
            # ci[h][np.where(ci[h-1]==i+1)]=mi	#assign new modules
            ci[h][np.where(ci[h - 1] == i + 1)] = m[i]

        n = np.max(m)  # new number of modules
        W1 = np.zeros((n, n))  # new weighted matrix
        for i in range(n):
            for j in range(i, n):
                # pool weights of nodes in same module
                wp = np.sum(W[np.ix_(m == i + 1, m == j + 1)])
                W1[i, j] = wp
                W1[j, i] = wp
        W = W1

        q.append(0)
        # compute modularity
        q[h] = np.trace(W) / s - gamma * np.sum(np.dot(W / s, W / s))
        if q[h] - q[h - 1] < 1e-10:  # if modularity does not increase
            break

    ci = np.array(ci, dtype=int)
    if hierarchy:
        ci = ci[1:-1]
        q = q[1:-1]
        return ci, q
    else:
        return ci[h - 1], q[h - 1]


def modularity_louvain_und_sign(W, gamma=1, qtype='sta', seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.

    The Louvain algorithm is a fast and accurate community detection
    algorithm (at the time of writing).

    Use this function as opposed to modularity_louvain_und() only if the
    network contains a mix of positive and negative weights.  If the network
    contains all positive weights, the output will be equivalent to that of
    modularity_louvain_und().

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted/binary connection matrix with positive and
        negative weights
    qtype : str
        modularity type. Can be 'sta' (default), 'pos', 'smp', 'gja', 'neg'.
        See Rubinov and Sporns (2011) for a description.
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.

    Returns
    -------
    ci : Nx1 np.ndarray
        refined community affiliation vector
    Q : float
        optimized modularity metric

    Notes
    -----
    Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    '''
    np.random.seed(seed)

    n = len(W)  # number of nodes

    W0 = W * (W > 0)  # positive weights matrix
    W1 = -W * (W < 0)  # negative weights matrix
    s0 = np.sum(W0)  # weight of positive links
    s1 = np.sum(W1)  # weight of negative links

    if qtype == 'smp':
        d0 = 1 / s0
        d1 = 1 / s1  # dQ=dQ0/s0-sQ1/s1
    elif qtype == 'gja':
        d0 = 1 / (s0 + s1)
        d1 = d0  # dQ=(dQ0-dQ1)/(s0+s1)
    elif qtype == 'sta':
        d0 = 1 / s0
        d1 = 1 / (s0 + s1)  # dQ=dQ0/s0-dQ1/(s0+s1)
    elif qtype == 'pos':
        d0 = 1 / s0
        d1 = 0  # dQ=dQ0/s0
    elif qtype == 'neg':
        d0 = 0
        d1 = 1 / s1  # dQ=-dQ1/s1
    else:
        raise KeyError('modularity type unknown')

    if not s0:  # adjust for absent positive weights
        s0 = 1
        d0 = 0
    if not s1:  # adjust for absent negative weights
        s1 = 1
        d1 = 0

    h = 1  # hierarchy index
    nh = n  # number of nodes in hierarchy
    ci = [None, np.arange(n) + 1]  # hierarchical module assignments
    q = [-1, 0]  # hierarchical modularity values
    while q[h] - q[h - 1] > 1e-10:
        if h > 300:
            raise BCTParamError('Modularity Infinite Loop Style A.  Please '
                                'contact the developer with this error.')
        kn0 = np.sum(W0, axis=0)  # positive node degree
        kn1 = np.sum(W1, axis=0)  # negative node degree
        km0 = kn0.copy()  # positive module degree
        km1 = kn1.copy()  # negative module degree
        knm0 = W0.copy()  # positive node-to-module degree
        knm1 = W1.copy()  # negative node-to-module degree

        m = np.arange(nh) + 1  # initial module assignments
        flag = True  # flag for within hierarchy search
        it = 0
        while flag:
            it += 1
            if it > 1000:
                raise BCTParamError('Infinite Loop was detected and stopped. '
                                    'This was probably caused by passing in a directed matrix.')
            flag = False
            # loop over nodes in random order
            for u in np.random.permutation(nh):
                ma = m[u] - 1
                dQ0 = ((knm0[u, :] + W0[u, u] - knm0[u, ma]) -
                       gamma * kn0[u] * (km0 + kn0[u] - km0[ma]) / s0)  # positive dQ
                dQ1 = ((knm1[u, :] + W1[u, u] - knm1[u, ma]) -
                       gamma * kn1[u] * (km1 + kn1[u] - km1[ma]) / s1)  # negative dQ

                dQ = d0 * dQ0 - d1 * dQ1  # rescaled changes in modularity
                dQ[ma] = 0  # no changes for same module

                max_dQ = np.max(dQ)  # maximal increase in modularity
                if max_dQ > 1e-10:  # if maximal increase is positive
                    flag = True
                    mb = np.argmax(dQ)

                    # change positive node-to-module degrees
                    knm0[:, mb] += W0[:, u]
                    knm0[:, ma] -= W0[:, u]
                    # change negative node-to-module degrees
                    knm1[:, mb] += W1[:, u]
                    knm1[:, ma] -= W1[:, u]
                    km0[mb] += kn0[u]  # change positive module degrees
                    km0[ma] -= kn0[u]
                    km1[mb] += kn1[u]  # change negative module degrees
                    km1[ma] -= kn1[u]

                    m[u] = mb + 1  # reassign module

        h += 1
        ci.append(np.zeros((n,)))
        _, m = np.unique(m, return_inverse=True)
        m += 1

        for u in range(nh):  # loop through initial module assignments
            ci[h][np.where(ci[h - 1] == u + 1)] = m[u]  # assign new modules

        nh = np.max(m)  # number of new nodes
        wn0 = np.zeros((nh, nh))  # new positive weights matrix
        wn1 = np.zeros((nh, nh))

        for u in range(nh):
            for v in range(u, nh):
                wn0[u, v] = np.sum(W0[np.ix_(m == u + 1, m == v + 1)])
                wn1[u, v] = np.sum(W1[np.ix_(m == u + 1, m == v + 1)])
                wn0[v, u] = wn0[u, v]
                wn1[v, u] = wn1[u, v]

        W0 = wn0
        W1 = wn1

        q.append(0)
        # compute modularity
        q0 = np.trace(W0) - np.sum(np.dot(W0, W0)) / s0
        q1 = np.trace(W1) - np.sum(np.dot(W1, W1)) / s1
        q[h] = d0 * q0 - d1 * q1

    _, ci_ret = np.unique(ci[-1], return_inverse=True)
    ci_ret += 1

    return ci_ret, q[-1]


def modularity_probtune_und_sign(W, qtype='sta', gamma=1, ci=None, p=.45,
                                 seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.
    High-modularity degeneracy is the presence of many topologically
    distinct high-modularity partitions of the network.

    This algorithm is inspired by the Kernighan-Lin fine-tuning algorithm
    and is designed to probabilistically refine a previously detected
    community by incorporating random node moves into a finetuning
    algorithm.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted/binary connection matrix with positive and
        negative weights
    qtype : str
        modularity type. Can be 'sta' (default), 'pos', 'smp', 'gja', 'neg'.
        See Rubinov and Sporns (2011) for a description.
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    ci : Nx1 np.ndarray | None
        initial community affiliation vector
    p : float
        probability of random node moves. Default value = 0.45
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.

    Returns
    -------
    ci : Nx1 np.ndarray
        refined community affiliation vector
    Q : float
        optimized modularity metric

    Notes
    -----
    Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    '''
    np.random.seed(seed)

    n = len(W)
    if ci is None:
        ci = np.arange(n) + 1
    else:
        _, ci = np.unique(ci, return_inverse=True)
        ci += 1

    W0 = W * (W > 0)  # positive weights matrix
    W1 = -W * (W < 0)  # negative weights matrix
    s0 = np.sum(W0)  # positive sum of weights
    s1 = np.sum(W1)  # negative sum of weights
    Knm0 = np.zeros((n, n))  # positive node-to-module degree
    Knm1 = np.zeros((n, n))  # negative node-to-module degree

    for m in range(int(np.max(ci))):  # loop over initial modules
        Knm0[:, m] = np.sum(W0[:, ci == m + 1], axis=1)
        Knm1[:, m] = np.sum(W1[:, ci == m + 1], axis=1)

    Kn0 = np.sum(Knm0, axis=1)  # positive node degree
    Kn1 = np.sum(Knm1, axis=1)  # negative node degree
    Km0 = np.sum(Knm0, axis=0)  # positive module degree
    Km1 = np.sum(Knm1, axis=0)  # negaitve module degree

    if qtype == 'smp':
        d0 = 1 / s0
        d1 = 1 / s1  # dQ=dQ0/s0-dQ1/s1
    elif qtype == 'gja':
        d0 = 1 / (s0 + s1)
        d1 = 1 / (s0 + s1)  # dQ=(dQ0-dQ1)/(s0+s1)
    elif qtype == 'sta':
        d0 = 1 / s0
        d1 = 1 / (s0 + s1)  # dQ=dQ0/s0-dQ1/(s0+s1)
    elif qtype == 'pos':
        d0 = 1 / s0
        d1 = 0  # dQ=dQ0/s0
    elif qtype == 'neg':
        d0 = 0
        d1 = 1 / s1  # dQ=-dQ1/s1
    else:
        raise KeyError('modularity type unknown')

    if not s0:  # adjust for absent positive weights
        s0 = 1
        d0 = 0
    if not s1:  # adjust for absent negative weights
        s1 = 1
        d1 = 0

    for u in np.random.permutation(n):  # loop over nodes in random order
        ma = ci[u] - 1  # current module
        r = np.random.random() < p
        if r:
            mb = np.random.randint(n)  # select new module randomly
        else:
            dq0 = ((Knm0[u, :] + W0[u, u] - Knm0[u, ma]) -
                   gamma * Kn0[u] * (Km0 + Kn0[u] - Km0[ma]) / s0)
            dq1 = ((Knm1[u, :] + W1[u, u] - Knm1[u, ma]) -
                   gamma * Kn1[u] * (Km1 + Kn1[u] - Km1[ma]) / s1)
            dq = d0 * dq0 - d1 * dq1
            dq[ma] = 0

            max_dq = np.max(dq)
            mb = np.argmax(dq)

        if r or max_dq > 1e-10:
            ci[u] = mb + 1

            Knm0[:, mb] += W0[:, u]
            Knm0[:, ma] -= W0[:, u]
            Knm1[:, mb] += W1[:, u]
            Knm1[:, ma] -= W1[:, u]
            Km0[mb] += Kn0[u]
            Km0[ma] -= Kn0[u]
            Km1[mb] += Kn1[u]
            Km1[ma] -= Kn1[u]

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    m = np.tile(ci, (n, 1))
    q0 = (W0 - np.outer(Kn0, Kn0) / s0) * (m == m.T)
    q1 = (W1 - np.outer(Kn1, Kn1) / s1) * (m == m.T)
    q = d0 * np.sum(q0) - d1 * np.sum(q1)

    return ci, q


def modularity_und(A, gamma=1, kci=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted/binary connection matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    kci : Nx1 np.ndarray | None
        starting community structure. If specified, calculates the Q-metric
        on the community structure giving, without doing any optimzation.
        Otherwise, if not specified, uses a spectral modularity maximization
        algorithm.

    Returns
    -------
    ci : Nx1 np.ndarray
        optimized community structure
    Q : float
        maximized modularity metric

    Notes
    -----
    This algorithm is deterministic. The matlab function bearing this
    name incorrectly disclaims that the outcome depends on heuristics
    involving a random seed. The louvain method does depend on a random seed,
    but this function uses a deterministic modularity maximization algorithm.
    '''
    from scipy import linalg
    n = len(A)  # number of vertices
    k = np.sum(A, axis=0)  # degree
    m = np.sum(k)  # number of edges (each undirected edge
    # is counted twice)
    B = A - gamma * np.outer(k, k) / m  # initial modularity matrix

    init_mod = np.arange(n)  # initial one big module
    modules = []  # output modules list

    def recur(module):
        n = len(module)
        modmat = B[module][:, module]
        modmat -= np.diag(np.sum(modmat, axis=0))

        vals, vecs = linalg.eigh(modmat)  # biggest eigendecomposition
        rlvals = np.real(vals)
        max_eigvec = np.squeeze(vecs[:, np.where(rlvals == np.max(rlvals))])
        if max_eigvec.ndim > 1:  # if multiple max eigenvalues, pick one
            max_eigvec = max_eigvec[:, 0]
        # initial module assignments
        mod_asgn = np.squeeze((max_eigvec >= 0) * 2 - 1)
        q = np.dot(mod_asgn, np.dot(modmat, mod_asgn))  # modularity change

        if q > 0:  # change in modularity was positive
            qmax = q
            np.fill_diagonal(modmat, 0)
            it = np.ma.masked_array(np.ones((n,)), False)
            mod_asgn_iter = mod_asgn.copy()
            while np.any(it):  # do some iterative fine tuning
                # this line is linear algebra voodoo
                q_iter = qmax - 4 * mod_asgn_iter * \
                    (np.dot(modmat, mod_asgn_iter))
                qmax = np.max(q_iter * it)
                imax = np.argmax(q_iter * it)
                #imax, = np.where(q_iter == qmax)
                #if len(imax) > 1:
                #    imax = imax[0]
                # does switching increase modularity?
                mod_asgn_iter[imax] *= -1
                it[imax] = np.ma.masked
                if qmax > q:
                    q = qmax
                    mod_asgn = mod_asgn_iter
            if np.abs(np.sum(mod_asgn)) == n:  # iteration yielded null module
                modules.append(np.array(module).tolist())
                return
            else:
                mod1 = module[np.where(mod_asgn == 1)]
                mod2 = module[np.where(mod_asgn == -1)]

                recur(mod1)
                recur(mod2)
        else:  # change in modularity was negative or 0
            modules.append(np.array(module).tolist())

    # adjustment to one-based indexing occurs in ls2ci
    if kci is None:
        recur(init_mod)
        ci = ls2ci(modules)
    else:
        ci = kci
    s = np.tile(ci, (n, 1))
    q = np.sum(np.logical_not(s - s.T) * B / m)
    return ci, q


def modularity_und_sign(W, ci, qtype='sta'):
    '''
    This function simply calculates the signed modularity for a given
    partition. It does not do automatic partition generation right now.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted/binary connection matrix with positive and
        negative weights
    ci : Nx1 np.ndarray
        community partition
    qtype : str
        modularity type. Can be 'sta' (default), 'pos', 'smp', 'gja', 'neg'.
        See Rubinov and Sporns (2011) for a description.

    Returns
    -------
    ci : Nx1 np.ndarray
        the partition which was input (for consistency of the API)
    Q : float
        maximized modularity metric

    Notes
    -----
    uses a deterministic algorithm
    '''
    n = len(W)
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    W0 = W * (W > 0)  # positive weights matrix
    W1 = -W * (W < 0)  # negative weights matrix
    s0 = np.sum(W0)  # positive sum of weights
    s1 = np.sum(W1)  # negative sum of weights
    Knm0 = np.zeros((n, n))  # positive node-to-module degree
    Knm1 = np.zeros((n, n))  # negative node-to-module degree

    for m in range(int(np.max(ci))):  # loop over initial modules
        Knm0[:, m] = np.sum(W0[:, ci == m + 1], axis=1)
        Knm1[:, m] = np.sum(W1[:, ci == m + 1], axis=1)

    Kn0 = np.sum(Knm0, axis=1)  # positive node degree
    Kn1 = np.sum(Knm1, axis=1)  # negative node degree
    Km0 = np.sum(Knm0, axis=0)  # positive module degree
    Km1 = np.sum(Knm1, axis=0)  # negaitve module degree

    if qtype == 'smp':
        d0 = 1 / s0
        d1 = 1 / s1  # dQ=dQ0/s0-dQ1/s1
    elif qtype == 'gja':
        d0 = 1 / (s0 + s1)
        d1 = 1 / (s0 + s1)  # dQ=(dQ0-dQ1)/(s0+s1)
    elif qtype == 'sta':
        d0 = 1 / s0
        d1 = 1 / (s0 + s1)  # dQ=dQ0/s0-dQ1/(s0+s1)
    elif qtype == 'pos':
        d0 = 1 / s0
        d1 = 0  # dQ=dQ0/s0
    elif qtype == 'neg':
        d0 = 0
        d1 = 1 / s1  # dQ=-dQ1/s1
    else:
        raise KeyError('modularity type unknown')

    if not s0:  # adjust for absent positive weights
        s0 = 1
        d0 = 0
    if not s1:  # adjust for absent negative weights
        s1 = 1
        d1 = 0

    m = np.tile(ci, (n, 1))

    q0 = (W0 - np.outer(Kn0, Kn0) / s0) * (m == m.T)
    q1 = (W1 - np.outer(Kn1, Kn1) / s1) * (m == m.T)
    q = d0 * np.sum(q0) - d1 * np.sum(q1)

    return ci, q


def partition_distance(cx, cy):
    '''
    This function quantifies the distance between pairs of community
    partitions with information theoretic measures.

    Parameters
    ----------
    cx : Nx1 np.ndarray
        community affiliation vector X
    cy : Nx1 np.ndarray
        community affiliation vector Y

    Returns
    -------
    VIn : Nx1 np.ndarray
        normalized variation of information
    MIn : Nx1 np.ndarray
        normalized mutual information

    Notes
    -----
    (Definitions:
       VIn = [H(X) + H(Y) - 2MI(X,Y)]/log(n)
       MIn = 2MI(X,Y)/[H(X)+H(Y)]
    where H is entropy, MI is mutual information and n is number of nodes)
    '''
    n = np.size(cx)
    _, cx = np.unique(cx, return_inverse=True)
    _, cy = np.unique(cy, return_inverse=True)
    _, cxy = np.unique(cx + cy * 1j, return_inverse=True)

    cx += 1
    cy += 1
    cxy += 1

    Px = np.histogram(cx, bins=np.max(cx))[0] / n
    Py = np.histogram(cy, bins=np.max(cy))[0] / n
    Pxy = np.histogram(cxy, bins=np.max(cxy))[0] / n

    Hx = -np.sum(Px * np.log(Px))
    Hy = -np.sum(Py * np.log(Py))
    Hxy = -np.sum(Pxy * np.log(Pxy))

    Vin = (2 * Hxy - Hx - Hy) / np.log(n)
    Min = 2 * (Hx + Hy - Hxy) / (Hx + Hy)
    return Vin, Min
