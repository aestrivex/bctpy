from __future__ import division, print_function
import numpy as np


def density_dir(CIJ):
    '''
    Density is the fraction of present connections to possible connections.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        directed weighted/binary connection matrix

    Returns
    -------
    kden : float
        density
    N : int
        number of vertices
    k : int
        number of edges

    Notes
    -----
    Assumes CIJ is directed and has no self-connections.
    Weight information is discarded.
    '''
    n = len(CIJ)
    k = np.size(np.where(CIJ.flatten()))
    kden = k / (n * n - n)
    return kden, n, k


def density_und(CIJ):
    '''
    Density is the fraction of present connections to possible connections.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        undirected (weighted/binary) connection matrix

    Returns
    -------
    kden : float
        density
    N : int
        number of vertices
    k : int
        number of edges

    Notes
    -----
    Assumes CIJ is undirected and has no self-connections.
            Weight information is discarded.
    '''
    n = len(CIJ)
    k = np.size(np.where(np.triu(CIJ).flatten()))
    kden = k / ((n * n - n) / 2)
    return kden, n, k


def rentian_scaling(A, xyz, n):
    '''
    Physical Rentian scaling (or more simply Rentian scaling) is a property
    of systems that are cost-efficiently embedded into physical space. It is
    what is called a "topo-physical" property because it combines information
    regarding the topological organization of the graph with information
    about the physical placement of connections. Rentian scaling is present
    in very large scale integrated circuits, the C. elegans neuronal network,
    and morphometric and diffusion-based graphs of human anatomical networks.
    Rentian scaling is determined by partitioning the system into cubes,
    counting the number of nodes inside of each cube (N), and the number of
    edges traversing the boundary of each cube (E). If the system displays
    Rentian scaling, these two variables N and E will scale with one another
    in loglog space. The Rent's exponent is given by the slope of log10(E)
    vs. log10(N), and can be reported alone or can be compared to the
    theoretical minimum Rent's exponent to determine how cost efficiently the
    network has been embedded into physical space. Note: if a system displays
    Rentian scaling, it does not automatically mean that the system is
    cost-efficiently embedded (although it does suggest that). Validation
    occurs when comparing to the theoretical minimum Rent's exponent for that
    system.

    Parameters
    ----------
    A : NxN np.ndarray
        unweighted, binary, symmetric adjacency matrix
    xyz : Nx3 np.ndarray
        vector of node placement coordinates
    n : int
        Number of partitions to compute. Each partition is a data point; you
        want a large enough number to adequately compute Rent's exponent.

    Returns
    -------
    N : Mx1 np.ndarray
        Number of nodes in each of the M partitions
    E : Mx1 np.ndarray

    Notes
    -----
    Subsequent Analysis:
    Rentian scaling plots are then created by: figure; loglog(E,N,'*');
    To determine the Rent's exponent, p, it is important not to use
    partitions which may
    be affected by boundary conditions. In Bassett et al. 2010 PLoS CB, only
    partitions with N<M/2 were used in the estimation of the Rent's exponent.
    Thus, we can define N_prime = N(find(N<M/2)) and
    E_prime = E(find(N<M/2)).
    Next we need to determine the slope of Eprime vs. Nprime in loglog space,
    which is the Rent's
    exponent. There are many ways of doing this with more or less statistical
    rigor. Robustfit in MATLAB is one such option:
       [b,stats] = robustfit(log10(N_prime),log10(E_prime))
    Then the Rent's exponent is b(1,2) and the standard error of the
    estimation is given by stats.se(1,2).

    Note: n=5000 was used in Bassett et al. 2010 in PLoS CB.
    '''
    m = np.size(xyz, axis=0)  # find number of nodes in system

    # rescale coordinates so they are all greater than unity
    xyzn = xyz - np.tile(np.min(xyz, axis=0) - 1, (m, 1))

    # find the absolute minimum and maximum over all directions
    nmax = np.max(xyzn)
    nmin = np.min(xyzn)

    count = 0
    N = np.zeros((n,))
    E = np.zeros((n,))

    # create partitions and count the number of nodes inside the partition (n)
    # and the number of edges traversing the boundary of the partition (e)
    while count < n:
        # define cube endpoints
        randx = np.sort((1 + nmax - nmin) * np.random.random((2,)))

        # find nodes in cube
        l1 = xyzn[:, 0] > randx[0]
        l2 = xyzn[:, 0] < randx[1]
        l3 = xyzn[:, 1] > randx[0]
        l4 = xyzn[:, 1] < randx[1]
        l5 = xyzn[:, 2] > randx[0]
        l6 = xyzn[:, 2] < randx[1]

        L, = np.where((l1 & l2 & l3 & l4 & l5 & l6).flatten())
        if np.size(L):
            # count edges crossing at the boundary of the cube
            E[count] = np.sum(A[np.ix_(L, np.setdiff1d(range(m), L))])
            # count nodes inside of the cube
            N[count] = np.size(L)
            count += 1

    return N, E
