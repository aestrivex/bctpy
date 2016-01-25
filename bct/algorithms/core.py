
import numpy as np
from .degree import degrees_dir, degrees_und, strengths_dir, strengths_und


def assortativity_bin(CIJ, flag):
    '''
    The assortativity coefficient is a correlation coefficient between the
    degrees of all nodes on two opposite ends of a link. A positive
    assortativity coefficient indicates that nodes tend to link to other
    nodes with the same or similar degree.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed/undirected connection matrix
    flag : int
        0 : undirected graph; degree/degree correlation
        1 : directed graph; out-degree/in-degree correlation
        2 : directed graph; in-degree/out-degree correlation
        3 : directed graph; out-degree/out-degree correlation
        4 : directed graph; in-degree/in-degreen correlation

    Returns
    -------
    r : float
        assortativity coefficient

    Notes
    -----
    The function accepts weighted networks, but all connection
    weights are ignored. The main diagonal should be empty. For flag 1
    the function computes the directed assortativity described in Rubinov
    and Sporns (2010) NeuroImage.
    '''
    if flag == 0:  # undirected version
        deg = degrees_und(CIJ)
        i, j = np.where(np.triu(CIJ, 1) > 0)
        K = len(i)
        degi = deg[i]
        degj = deg[j]
    else:  # directed version
        id, od, deg = degrees_dir(CIJ)
        i, j = np.where(CIJ > 0)
        K = len(i)

        if flag == 1:
            degi = od[i]
            degj = id[j]
        elif flag == 2:
            degi = id[i]
            degj = od[j]
        elif flag == 3:
            degi = od[i]
            degj = od[j]
        elif flag == 4:
            degi = id[i]
            degj = id[j]
        else:
            raise ValueError('Flag must be 0-4')

    # compute assortativity
    term1 = np.sum(degi * degj) / K
    term2 = np.square(np.sum(.5 * (degi + degj)) / K)
    term3 = np.sum(.5 * (degi * degi + degj * degj)) / K
    r = (term1 - term2) / (term3 - term2)
    return r


def assortativity_wei(CIJ, flag):
    '''
    The assortativity coefficient is a correlation coefficient between the
    strengths (weighted degrees) of all nodes on two opposite ends of a link.
    A positive assortativity coefficient indicates that nodes tend to link to
    other nodes with the same or similar strength.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        weighted directed/undirected connection matrix
    flag : int
        0 : undirected graph; strength/strength correlation
        1 : directed graph; out-strength/in-strength correlation
        2 : directed graph; in-strength/out-strength correlation
        3 : directed graph; out-strength/out-strength correlation
        4 : directed graph; in-strength/in-strengthn correlation

    Returns
    -------
    r : float
        assortativity coefficient

    Notes
    -----
    The main diagonal should be empty. For flag 1
       the function computes the directed assortativity described in Rubinov
       and Sporns (2010) NeuroImage.
    '''
    if flag == 0:  # undirected version
        str = strengths_und(CIJ)
        i, j = np.where(np.triu(CIJ, 1) > 0)
        K = len(i)
        stri = str[i]
        strj = str[j]
    else:
        ist, ost = strengths_dir(CIJ)  # directed version
        i, j = np.where(CIJ > 0)
        K = len(i)

        if flag == 1:
            stri = ost[i]
            strj = ist[j]
        elif flag == 2:
            stri = ist[i]
            strj = ost[j]
        elif flag == 3:
            stri = ost[i]
            strj = ost[j]
        elif flag == 4:
            stri = ist[i]
            strj = ost[j]
        else:
            raise ValueError('Flag must be 0-4')

    # compute assortativity
    term1 = np.sum(stri * strj) / K
    term2 = np.square(np.sum(.5 * (stri + strj)) / K)
    term3 = np.sum(.5 * (stri * stri + strj * strj)) / K
    r = (term1 - term2) / (term3 - term2)
    return r


def kcore_bd(CIJ, k, peel=False):
    '''
    The k-core is the largest subnetwork comprising nodes of degree at
    least k. This function computes the k-core for a given binary directed
    connection matrix by recursively peeling off nodes with degree lower
    than k, until no such nodes remain.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed adjacency matrix
    k : int
        level of k-core
    peel : bool
        If True, additionally calculates peelorder and peellevel. Defaults to
        False.

    Returns
    -------
    CIJkcore : NxN np.ndarray
        connection matrix of the k-core. This matrix only contains nodes of
        degree at least k.
    kn : int
        size of k-core
    peelorder : Nx1 np.ndarray
        indices in the order in which they were peeled away during k-core
        decomposition. only returned if peel is specified.
    peellevel : Nx1 np.ndarray
        corresponding level - nodes in at the same level have been peeled
        away at the same time. only return if peel is specified

    Notes
    -----
    'peelorder' and 'peellevel' are similar the the k-core sub-shells
    described in Modha and Singh (2010).
    '''
    if peel:
        peelorder, peellevel = ([], [])
    iter = 0
    CIJkcore = CIJ.copy()

    while True:
        id, od, deg = degrees_dir(CIJkcore)  # get degrees of matrix

        # find nodes with degree <k
        ff, = np.where(np.logical_and(deg < k, deg > 0))

        if ff.size == 0:
            break  # if none found -> stop

        # else peel away found nodes
        iter += 1
        CIJkcore[ff, :] = 0
        CIJkcore[:, ff] = 0

        if peel:
            peelorder.append(ff)
        if peel:
            peellevel.append(iter * np.ones((len(ff),)))

    kn = np.sum(deg > 0)

    if peel:
        return CIJkcore, kn, peelorder, peellevel
    else:
        return CIJkcore, kn


def kcore_bu(CIJ, k, peel=False):
    '''
    The k-core is the largest subnetwork comprising nodes of degree at
    least k. This function computes the k-core for a given binary
    undirected connection matrix by recursively peeling off nodes with
    degree lower than k, until no such nodes remain.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary undirected connection matrix
    k : int
        level of k-core
    peel : bool
        If True, additionally calculates peelorder and peellevel. Defaults to
        False.

    Returns
    -------
    CIJkcore : NxN np.ndarray
        connection matrix of the k-core. This matrix only contains nodes of
        degree at least k.
    kn : int
        size of k-core
    peelorder : Nx1 np.ndarray
        indices in the order in which they were peeled away during k-core
        decomposition. only returned if peel is specified.
    peellevel : Nx1 np.ndarray
        corresponding level - nodes in at the same level have been peeled
        away at the same time. only return if peel is specified

    Notes
    -----
    'peelorder' and 'peellevel' are similar the the k-core sub-shells
    described in Modha and Singh (2010).
    '''
    if peel:
        peelorder, peellevel = ([], [])
    iter = 0
    CIJkcore = CIJ.copy()

    while True:
        deg = degrees_und(CIJkcore)  # get degrees of matrix

        # find nodes with degree <k
        ff, = np.where(np.logical_and(deg < k, deg > 0))

        if ff.size == 0:
            break  # if none found -> stop

        # else peel away found nodes
        iter += 1
        CIJkcore[ff, :] = 0
        CIJkcore[:, ff] = 0

        if peel:
            peelorder.append(ff)
        if peel:
            peellevel.append(iter * np.ones((len(ff),)))

    kn = np.sum(deg > 0)

    if peel:
        return CIJkcore, kn, peelorder, peellevel
    else:
        return CIJkcore, kn


def rich_club_bd(CIJ, klevel=None):
    '''
    The rich club coefficient, R, at level k is the fraction of edges that
    connect nodes of degree k or higher out of the maximum number of edges
    that such nodes might share.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed connection matrix
    klevel : int | None
        sets the maximum level at which the rich club coefficient will be
        calculated. If None (default), the maximum level is set to the
        maximum degree of the adjacency matrix

    Returns
    -------
    R : Kx1 np.ndarray
        vector of rich-club coefficients for levels 1 to klevel
    Nk : int
        number of nodes with degree > k
    Ek : int
        number of edges remaining in subgraph with degree > k
    '''
    # definition of degree as used for RC coefficients
    # degree is taken to be the sum of incoming and outgoing connections
    id, od, deg = degrees_dir(CIJ)

    if klevel is None:
        klevel = int(np.max(deg))

    R = np.zeros((klevel,))
    Nk = np.zeros((klevel,))
    Ek = np.zeros((klevel,))
    for k in range(klevel):
        SmallNodes, = np.where(deg <= k + 1)  # get small nodes with degree <=k
        subCIJ = np.delete(CIJ, SmallNodes, axis=0)
        subCIJ = np.delete(subCIJ, SmallNodes, axis=1)
        Nk[k] = np.size(subCIJ, axis=1)  # number of nodes with degree >k
        Ek[k] = np.sum(subCIJ)  # number of connections in subgraph
        # unweighted rich club coefficient
        R[k] = Ek[k] / (Nk[k] * (Nk[k] - 1))

    return R, Nk, Ek


def rich_club_bu(CIJ, klevel=None):
    '''
    The rich club coefficient, R, at level k is the fraction of edges that
    connect nodes of degree k or higher out of the maximum number of edges
    that such nodes might share.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary undirected connection matrix
    klevel : int | None
        sets the maximum level at which the rich club coefficient will be
        calculated. If None (default), the maximum level is set to the
        maximum degree of the adjacency matrix

    Returns
    -------
    R : Kx1 np.ndarray
        vector of rich-club coefficients for levels 1 to klevel
    Nk : int
        number of nodes with degree > k
    Ek : int
        number of edges remaining in subgraph with degree > k
    '''
    deg = degrees_und(CIJ)  # compute degree of each node

    if klevel == None:
        klevel = int(np.max(deg))

    R = np.zeros((klevel,))
    Nk = np.zeros((klevel,))
    Ek = np.zeros((klevel,))
    for k in range(klevel):
        SmallNodes, = np.where(deg <= k + 1)  # get small nodes with degree <=k
        subCIJ = np.delete(CIJ, SmallNodes, axis=0)
        subCIJ = np.delete(subCIJ, SmallNodes, axis=1)
        Nk[k] = np.size(subCIJ, axis=1)  # number of nodes with degree >k
        Ek[k] = np.sum(subCIJ)  # number of connections in subgraph
        # unweighted rich club coefficient
        R[k] = Ek[k] / (Nk[k] * (Nk[k] - 1))

    return R, Nk, Ek


def rich_club_wd(CIJ, klevel=None):
    '''
    Parameters
    ----------
    CIJ : NxN np.ndarray
        weighted directed connection matrix
    klevel : int | None
        sets the maximum level at which the rich club coefficient will be
        calculated. If None (default), the maximum level is set to the
        maximum degree of the adjacency matrix

    Returns
    -------
    Rw : Kx1 np.ndarray
        vector of rich-club coefficients for levels 1 to klevel
    '''
    nr_nodes = len(CIJ)
    # degree of each node is defined here as in+out
    deg = np.sum((CIJ != 0), axis=0) + np.sum((CIJ.T != 0), axis=0)

    if klevel is None:
        klevel = np.max(deg)
    Rw = np.zeros((klevel,))

    # sort the weights of the network, with the strongest connection first
    wrank = np.sort(CIJ.flat)[::-1]

    for k in range(klevel):
        SmallNodes, = np.where(deg < k + 1)
        if np.size(SmallNodes) == 0:
            Rw[k] = np.nan
            continue

        # remove small nodes with node degree < k
        cutCIJ = np.delete(
            np.delete(CIJ, SmallNodes, axis=0), SmallNodes, axis=1)
        # total weight of connections in subset E>r
        Wr = np.sum(cutCIJ)
        # total number of connections in subset E>r
        Er = np.size(np.where(cutCIJ.flat != 0), axis=1)
        # E>r number of connections with max weight in network
        wrank_r = wrank[:Er]
        # weighted rich-club coefficient
        Rw[k] = Wr / np.sum(wrank_r)
    return Rw


def rich_club_wu(CIJ, klevel=None):
    '''
    Parameters
    ----------
    CIJ : NxN np.ndarray
        weighted undirected connection matrix
    klevel : int | None
        sets the maximum level at which the rich club coefficient will be
        calculated. If None (default), the maximum level is set to the
        maximum degree of the adjacency matrix

    Returns
    -------
    Rw : Kx1 np.ndarray
        vector of rich-club coefficients for levels 1 to klevel
    '''
    nr_nodes = len(CIJ)
    deg = np.sum((CIJ != 0), axis=0)

    if klevel is None:
        klevel = np.max(deg)
    Rw = np.zeros((klevel,))

    # sort the weights of the network, with the strongest connection first
    wrank = np.sort(CIJ.flat)[::-1]

    for k in range(klevel):
        SmallNodes, = np.where(deg < k + 1)
        if np.size(SmallNodes) == 0:
            Rw[k] = np.nan
            continue

        # remove small nodes with node degree < k
        cutCIJ = np.delete(
            np.delete(CIJ, SmallNodes, axis=0), SmallNodes, axis=1)
        # total weight of connections in subset E>r
        Wr = np.sum(cutCIJ)
        # total number of connections in subset E>r
        Er = np.size(np.where(cutCIJ.flat != 0), axis=1)
        # E>r number of connections with max weight in network
        wrank_r = wrank[:Er]
        # weighted rich-club coefficient
        Rw[k] = Wr / np.sum(wrank_r)
    return Rw


def score_wu(CIJ, s):
    '''
    The s-core is the largest subnetwork comprising nodes of strength at
    least s. This function computes the s-core for a given weighted
    undirected connection matrix. Computation is analogous to the more
    widely used k-core, but is based on node strengths instead of node
    degrees.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        weighted undirected connection matrix
    s : float
        level of s-core. Note that can take on any fractional value.

    Returns
    -------
    CIJscore : NxN np.ndarray
        connection matrix of the s-core. This matrix contains only nodes with
        a strength of at least s.
    sn : int
        size of s-core
    '''
    CIJscore = CIJ.copy()
    while True:
        str = strengths_und(CIJscore)  # get strengths of matrix

        # find nodes with strength <s
        ff, = np.where(np.logical_and(str < s, str > 0))

        if ff.size == 0:
            break  # if none found -> stop

        # else peel away found nodes
        CIJscore[ff, :] = 0
        CIJscore[:, ff] = 0

    sn = np.sum(str > 0)
    return CIJscore, sn
