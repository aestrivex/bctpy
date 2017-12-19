from __future__ import division, print_function
import numpy as np
from .miscellaneous_utilities import BCTParamError


def adjacency_plot_und(A, coor, tube=False):
    '''
    This function in matlab is a visualization helper which translates an
    adjacency matrix and an Nx3 matrix of spatial coordinates, and plots a
    3D isometric network connecting the undirected unweighted nodes using a
    specific plotting format. Including the formatted output is not useful at
    all for bctpy since matplotlib will not be able to plot it in quite the
    same way.

    Instead of doing this, I have included code that will plot the adjacency
    matrix onto nodes at the given spatial coordinates in mayavi

    This routine is basically a less featureful version of the 3D brain in
    cvu, the connectome visualization utility which I also maintain. cvu uses
    freesurfer surfaces and annotations to get the node coordinates (rather
    than leaving them up to the user) and has many other interactive
    visualization features not included here for the sake of brevity.

    There are other similar visualizations in the ConnectomeViewer and the
    UCLA multimodal connectivity database.

    Note that unlike other bctpy functions, this function depends on mayavi.

    Paramaters
    ----------
    A : NxN np.ndarray
        adjacency matrix
    coor : Nx3 np.ndarray
        vector of node coordinates
    tube : bool
        plots using cylindrical tubes for higher resolution image. If True,
        plots cylindrical tube sources. If False, plots line sources. Default
        value is False.

    Returns
    -------
    fig : Instance(Scene)
        handle to a mayavi figure.

    Notes
    -----
    To display the output interactively, call

    fig=adjacency_plot_und(A,coor)
    from mayavi import mlab
    mlab.show()

    Note: Thresholding the matrix is strongly recommended.  It is recommended
    that the input matrix have fewer than 5000 total connections in order to
    achieve reasonable performance and noncluttered visualization.
    '''
    from mayavi import mlab

    n = len(A)
    nr_edges = (n * n - 1) // 2

    #starts = np.zeros((nr_edges,3))
    #vecs = np.zeros((nr_edges,3))
    #adjdat = np.zeros((nr_edges,))

    ixes, = np.where(np.triu(np.ones((n, n)), 1).flat)

    # i=0
    # for r2 in xrange(n):
    #	for r1 in xrange(r2):
    #		starts[i,:] = coor[r1,:]
    #		vecs[i,:] = coor[r2,:] - coor[r1,:]
    #		adjdat[i,:]
    #		i+=1

    adjdat = A.flat[ixes]

    A_r = np.tile(coor, (n, 1, 1))
    starts = np.reshape(A_r, (n * n, 3))[ixes, :]

    vecs = np.reshape(A_r - np.transpose(A_r, (1, 0, 2)), (n * n, 3))[ixes, :]

    # plotting
    fig = mlab.figure()

    nodesource = mlab.pipeline.scalar_scatter(
        coor[:, 0], coor[:, 1], coor[:, 2], figure=fig)

    nodes = mlab.pipeline.glyph(nodesource, scale_mode='none',
                                scale_factor=3., mode='sphere', figure=fig)
    nodes.glyph.color_mode = 'color_by_scalar'

    vectorsrc = mlab.pipeline.vector_scatter(
        starts[:, 0], starts[:, 1], starts[
            :, 2], vecs[:, 0], vecs[:, 1], vecs[:, 2],
        figure=fig)
    vectorsrc.mlab_source.dataset.point_data.scalars = adjdat

    thres = mlab.pipeline.threshold(vectorsrc,
                                    low=0.0001, up=np.max(A), figure=fig)

    vectors = mlab.pipeline.vectors(thres, colormap='YlOrRd',
                                    scale_mode='vector', figure=fig)
    vectors.glyph.glyph.clamping = False
    vectors.glyph.glyph.color_mode = 'color_by_scalar'
    vectors.glyph.color_mode = 'color_by_scalar'
    vectors.glyph.glyph_source.glyph_position = 'head'
    vectors.actor.property.opacity = .7
    if tube:
        vectors.glyph.glyph_source.glyph_source = (vectors.glyph.glyph_source.
                                                   glyph_dict['cylinder_source'])
        vectors.glyph.glyph_source.glyph_source.radius = 0.015
    else:
        vectors.glyph.glyph_source.glyph_source.glyph_type = 'dash'

    return fig


def align_matrices(m1, m2, dfun='sqrdiff', verbose=False, H=1e6, Texp=1,
                   T0=1e-3, Hbrk=10):
    '''
    This function aligns two matrices relative to one another by reordering
    the nodes in M2.  The function uses a version of simulated annealing.

    Parameters
    ----------
    M1 : NxN np.ndarray
        first connection matrix
    M2 : NxN np.ndarray
        second connection matrix
    dfun : str
        distance metric to use for matching
            'absdiff' : absolute difference
            'sqrdiff' : squared difference (default)
            'cosang' : cosine of vector angle
    verbose : bool
        print out cost at each iteration. Default False.
    H : int
        annealing parameter, default value 1e6
    Texp : int
        annealing parameter, default value 1. Coefficient of H s.t.
        Texp0=1-Texp/H
    T0 : float
        annealing parameter, default value 1e-3
    Hbrk : int
        annealing parameter, default value = 10. Coefficient of H s.t.
        Hbrk0 = H/Hkbr

    Returns
    -------
    Mreordered : NxN np.ndarray
        reordered connection matrix M2
    Mindices : Nx1 np.ndarray
        reordered indices
    cost : float
        objective function distance between M1 and Mreordered

    Notes
    -----
    Connection matrices can be weighted or binary, directed or undirected.
    They must have the same number of nodes.  M1 can be entered in any
    node ordering.

    Note that in general, the outcome will depend on the initial condition
    (the setting of the random number seed).  Also, there is no good way to
    determine optimal annealing parameters in advance - these parameters
    will need to be adjusted "by hand" (particularly H, Texp, T0, and Hbrk).
    For large and/or dense matrices, it is highly recommended to perform
    exploratory runs varying the settings of 'H' and 'Texp' and then select
    the best values.

    Based on extensive testing, it appears that T0 and Hbrk can remain
    unchanged in most cases.  Texp may be varied from 1-1/H to 1-10/H, for
    example.  H is the most important parameter - set to larger values as
    the problem size increases.  Good solutions can be obtained for
    matrices up to about 100 nodes.  It is advisable to run this function
    multiple times and select the solution(s) with the lowest 'cost'.

    If the two matrices are related it may be very helpful to pre-align them
    by reordering along their largest eigenvectors:
       [v,~] = eig(M1); v1 = abs(v(:,end)); [a1,b1] = sort(v1);
       [v,~] = eig(M2); v2 = abs(v(:,end)); [a2,b2] = sort(v2);
       [a,b,c] = overlapMAT2(M1(b1,b1),M2(b2,b2),'dfun',1);

    Setting 'Texp' to zero cancels annealing and uses a greedy algorithm
    instead.
    '''
    n = len(m1)
    if n < 2:
        raise BCTParamError("align_matrix will infinite loop on a singleton "
                            "or null matrix.")

    # define maxcost (greatest possible difference) and lowcost
    if dfun in ('absdiff', 'absdff'):
        maxcost = np.sum(np.abs(np.sort(m1.flat) - np.sort(m2.flat)[::-1]))
        lowcost = np.sum(np.abs(m1 - m2)) / maxcost
    elif dfun in ('sqrdiff', 'sqrdff'):
        maxcost = np.sum((np.sort(m1.flat) - np.sort(m2.flat)[::-1])**2)
        lowcost = np.sum((m1 - m2)**2) / maxcost
    elif dfun == 'cosang':
        maxcost = np.pi / 2
        lowcost = np.arccos(np.dot(m1.flat, m2.flat) /
                            np.sqrt(np.dot(m1.flat, m1.flat) * np.dot(m2.flat, m2.flat))) / maxcost
    else:
        raise BCTParamError('dfun must be absdiff or sqrdiff or cosang')

    mincost = lowcost
    anew = np.arange(n)
    amin = np.arange(n)
    h = 0
    hcnt = 0

    # adjust annealing parameters from user provided coefficients
    # H determines the maximal number of steps (user-provided)
    # Texp determines the steepness of the temperature gradient
    Texp = 1 - Texp / H
    # T0 sets the initial temperature and scales the energy term (user provided)
    # Hbrk sets a break point for the stimulation
    Hbrk = H / Hbrk

    while h < H:
        h += 1
        hcnt += 1
        # terminate if no new mincost has been found for some time
        if hcnt > Hbrk:
            break
        # current temperature
        T = T0 * (Texp**h)

        # choose two positions at random and flip them
        atmp = anew.copy()
        r1, r2 = np.random.randint(n, size=(2,))
        while r1 == r2:
            r2 = np.random.randint(n)
        atmp[r1] = anew[r2]
        atmp[r2] = anew[r1]
        m2atmp = m2[np.ix_(atmp, atmp)]
        if dfun in ('absdiff', 'absdff'):
            costnew = np.sum(np.abs(m1 - m2atmp)) / maxcost
        elif dfun in ('sqrdiff', 'sqrdff'):
            costnew = np.sum((m1 - m2atmp)**2) / maxcost
        elif dfun == 'cosang':
            costnew = np.arccos(np.dot(m1.flat, m2atmp.flat) / np.sqrt(
                np.dot(m1.flat, m1.flat) * np.dot(m2.flat, m2.flat))) / maxcost

        if costnew < lowcost or np.random.random() < np.exp(-(costnew - lowcost) / T):
            anew = atmp
            lowcost = costnew
            # is this the absolute best?
            if lowcost < mincost:
                amin = anew
                mincost = lowcost
                if verbose:
                    print('step %i ... current lowest cost = %f' % (h, mincost))
                hcnt = 0
            # if the cost is 0 we're done
            if mincost == 0:
                break
    if verbose:
        print('step %i ... final lowest cost = %f' % (h, mincost))

    M_reordered = m2[np.ix_(amin, amin)]
    M_indices = amin
    cost = mincost
    return M_reordered, M_indices, cost


def backbone_wu(CIJ, avgdeg):
    '''
    The network backbone contains the dominant connections in the network
    and may be used to aid network visualization. This function computes
    the backbone of a given weighted and undirected connection matrix CIJ,
    using a minimum-spanning-tree based algorithm.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        weighted undirected connection matrix
    avgdeg : int
        desired average degree of backbone

    Returns
    -------
    CIJtree : NxN np.ndarray
        connection matrix of the minimum spanning tree of CIJ
    CIJclus : NxN np.ndarray
        connection matrix of the minimum spanning tree plus strongest
        connections up to some average degree 'avgdeg'. Identical to CIJtree
        if the degree requirement is already met.

    Notes
    -----
    NOTE: nodes with zero strength are discarded.
    NOTE: CIJclus will have a total average degree exactly equal to
         (or very close to) 'avgdeg'.
    NOTE: 'avgdeg' backfill is handled slightly differently than in Hagmann
         et al 2008.
    '''
    n = len(CIJ)
    if not np.all(CIJ == CIJ.T):
        raise BCTParamError('backbone_wu can only be computed for undirected '
                            'matrices.  If your matrix is has noise, correct it with np.around')
    CIJtree = np.zeros((n, n))

    # find strongest edge (if multiple edges are tied, use only first one)
    i, j = np.where(np.max(CIJ) == CIJ)
    im = [i[0], i[1]]  # what?  why take two values?  doesnt that mess up multiples?
    jm = [j[0], j[1]]

    # copy into tree graph
    CIJtree[im, jm] = CIJ[im, jm]
    in_ = im
    out = np.setdiff1d(range(n), in_)

    # repeat n-2 times
    for ix in range(n - 2):
        CIJ_io = CIJ[np.ix_(in_, out)]
        i, j = np.where(np.max(CIJ_io) == CIJ_io)
        # i,j=np.where(np.max(CIJ[in_,out])==CIJ[in_,out])
        print(i, j)
        im = in_[i[0]]
        jm = out[j[0]]

        # copy into tree graph
        CIJtree[im, jm] = CIJ[im, jm]
        CIJtree[jm, im] = CIJ[jm, im]
        in_ = np.append(in_, jm)
        out = np.setdiff1d(range(n), in_)

    # now add connections back with the total number of added connections
    # determined by the desired avgdeg

    CIJnotintree = CIJ * np.logical_not(CIJtree)
    ix, = np.where(CIJnotintree.flat)
    a = np.sort(CIJnotintree.flat[ix])[::-1]
    cutoff = avgdeg * n - 2 * (n - 1) - 1
    # if the avgdeg req is already satisfied, skip this
    if cutoff >= np.size(a):
        CIJclus = CIJtree.copy()
    else:
        thr = a[cutoff]
        CIJclus = CIJtree + CIJnotintree * (CIJnotintree >= thr)

    return CIJtree, CIJclus


def grid_communities(c):
    '''
    (X,Y,INDSORT) = GRID_COMMUNITIES(C) takes a vector of community
    assignments C and returns three output arguments for visualizing the
    communities. The third is INDSORT, which is an ordering of the vertices
    so that nodes with the same community assignment are next to one
    another. The first two arguments are vectors that, when overlaid on the
    adjacency matrix using the PLOT function, highlight the communities.

    Parameters
    ----------
    c : Nx1 np.ndarray
        community assignments

    Returns
    -------
    bounds : list
        list containing the communities
    indsort : np.ndarray
        indices

    Notes
    -----
    Note: This function returns considerably different values than in
    matlab due to differences between matplotlib and matlab.  This function
    has been designed to work with matplotlib, as in the following example:

    ci,_=modularity_und(adj)
    bounds,ixes=grid_communities(ci)
    pylab.imshow(adj[np.ix_(ixes,ixes)],interpolation='none',cmap='BuGn')
    for b in bounds:
      pylab.axvline(x=b,color='red')
      pylab.axhline(y=b,color='red')

    Note that I adapted the idea from the matlab function of the same name,
    and have not tested the functionality extensively.
    '''
    c = c.copy()
    nr_c = np.max(c)
    ixes = np.argsort(c)
    c = c[ixes]

    bounds = []

    for i in range(nr_c):
        ind = np.where(c == i + 1)
        if np.size(ind):
            mn = np.min(ind) - .5
            mx = np.max(ind) + .5
            bounds.extend([mn, mx])

    bounds = np.unique(bounds)
    return bounds, ixes


def reorderMAT(m, H=5000, cost='line'):
    '''
    This function reorders the connectivity matrix in order to place more
    edges closer to the diagonal. This often helps in displaying community
    structure, clusters, etc.

    Parameters
    ----------
    MAT : NxN np.ndarray
        connection matrix
    H : int
        number of reordering attempts
    cost : str
        'line' or 'circ' for shape of lattice (linear or ring lattice).
        Default is linear lattice.

    Returns
    -------
    MATreordered : NxN np.ndarray
        reordered connection matrix
    MATindices : Nx1 np.ndarray
        reordered indices
    MATcost : float
        objective function cost of reordered matrix

    Notes
    -----
    I'm not 100% sure how the algorithms between this and reorder_matrix
    differ, but this code looks a ton sketchier and might have had some minor
    bugs in it.  Considering reorder_matrix() does the same thing using a well
    vetted simulated annealing algorithm, just use that. ~rlaplant
    '''
    from scipy import linalg, stats
    m = m.copy()
    n = len(m)
    np.fill_diagonal(m, 0)

    # generate cost function
    if cost == 'line':
        profile = stats.norm.pdf(range(1, n + 1), 0, n / 2)[::-1]
    elif cost == 'circ':
        profile = stats.norm.pdf(range(1, n + 1), n / 2, n / 4)[::-1]
    else:
        raise BCTParamError('dfun must be line or circ')
    costf = linalg.toeplitz(profile, r=profile)

    lowcost = np.sum(costf * m)

    # keep track of starting configuration
    m_start = m.copy()
    starta = np.arange(n)
    # reorder
    for h in range(H):
        a = np.arange(n)
        # choose two positions and flip them
        r1, r2 = np.random.randint(n, size=(2,))
        a[r1] = r2
        a[r2] = r1
        costnew = np.sum((m[np.ix_(a, a)]) * costf)
        # if this reduced the overall cost
        if costnew < lowcost:
            m = m[np.ix_(a, a)]
            r2_swap = starta[r2]
            r1_swap = starta[r1]
            starta[r1] = r2_swap
            starta[r2] = r1_swap
            lowcost = costnew

    M_reordered = m_start[np.ix_(starta, starta)]
    m_indices = starta
    cost = lowcost
    return M_reordered, m_indices, cost


def reorder_matrix(m1, cost='line', verbose=False, H=1e4, Texp=10, T0=1e-3, Hbrk=10):
    '''
    This function rearranges the nodes in matrix M1 such that the matrix
    elements are squeezed along the main diagonal.  The function uses a
    version of simulated annealing.

    Parameters
    ----------
    M1 : NxN np.ndarray
        connection matrix weighted/binary directed/undirected
    cost : str
        'line' or 'circ' for shape of lattice (linear or ring lattice).
        Default is linear lattice.
    verbose : bool
        print out cost at each iteration. Default False.
    H : int
        annealing parameter, default value 1e6
    Texp : int
        annealing parameter, default value 1. Coefficient of H s.t.
        Texp0=1-Texp/H
    T0 : float
        annealing parameter, default value 1e-3
    Hbrk : int
        annealing parameter, default value = 10. Coefficient of H s.t.
        Hbrk0 = H/Hkbr

    Returns
    -------
    Mreordered : NxN np.ndarray
        reordered connection matrix
    Mindices : Nx1 np.ndarray
        reordered indices
    Mcost : float
        objective function cost of reordered matrix

    Notes
    -----
    Note that in general, the outcome will depend on the initial condition
    (the setting of the random number seed).  Also, there is no good way to
    determine optimal annealing parameters in advance - these paramters
    will need to be adjusted "by hand" (particularly H, Texp, and T0).
    For large and/or dense matrices, it is highly recommended to perform
    exploratory runs varying the settings of 'H' and 'Texp' and then select
    the best values.

    Based on extensive testing, it appears that T0 and Hbrk can remain
    unchanged in most cases.  Texp may be varied from 1-1/H to 1-10/H, for
    example.  H is the most important parameter - set to larger values as
    the problem size increases.  It is advisable to run this function
    multiple times and select the solution(s) with the lowest 'cost'.

    Setting 'Texp' to zero cancels annealing and uses a greedy algorithm
    instead.
    '''
    from scipy import linalg, stats
    n = len(m1)
    if n < 2:
        raise BCTParamError("align_matrix will infinite loop on a singleton "
                            "or null matrix.")

    # generate cost function
    if cost == 'line':
        profile = stats.norm.pdf(range(1, n + 1), loc=0, scale=n / 2)[::-1]
    elif cost == 'circ':
        profile = stats.norm.pdf(
            range(1, n + 1), loc=n / 2, scale=n / 4)[::-1]
    else:
        raise BCTParamError('cost must be line or circ')

    costf = linalg.toeplitz(profile, r=profile) * np.logical_not(np.eye(n))
    costf /= np.sum(costf)

    # establish maxcost, lowcost, mincost
    maxcost = np.sum(np.sort(costf.flat) * np.sort(m1.flat))
    lowcost = np.sum(m1 * costf) / maxcost
    mincost = lowcost

    # initialize
    anew = np.arange(n)
    amin = np.arange(n)
    h = 0
    hcnt = 0

    # adjust annealing parameters
    # H determines the maximal number of steps (user specified)
    # Texp determines the steepness of the temperature gradient
    Texp = 1 - Texp / H
    # T0 sets the initial temperature and scales the energy term (user provided)
    # Hbrk sets a break point for the stimulation
    Hbrk = H / Hbrk

    while h < H:
        h += 1
        hcnt += 1
        # terminate if no new mincost has been found for some time
        if hcnt > Hbrk:
            break
        T = T0 * Texp**h
        atmp = anew.copy()
        r1, r2 = np.random.randint(n, size=(2,))
        while r1 == r2:
            r2 = np.random.randint(n)
        atmp[r1] = anew[r2]
        atmp[r2] = anew[r1]
        costnew = np.sum((m1[np.ix_(atmp, atmp)]) * costf) / maxcost
        # annealing
        if costnew < lowcost or np.random.random() < np.exp(-(costnew - lowcost) / T):
            anew = atmp
            lowcost = costnew
            # is this a new absolute best?
            if lowcost < mincost:
                amin = anew
                mincost = lowcost
                if verbose:
                    print('step %i ... current lowest cost = %f' % (h, mincost))
                hcnt = 0

    if verbose:
        print('step %i ... final lowest cost = %f' % (h, mincost))

    M_reordered = m1[np.ix_(amin, amin)]
    M_indices = amin
    cost = mincost
    return M_reordered, M_indices, cost


def reorder_mod(A, ci):
    '''
    This function reorders the connectivity matrix by modular structure and
    may hence be useful in visualization of modular structure.

    Parameters
    ----------
    A : NxN np.ndarray
        binary/weighted connectivity matrix
    ci : Nx1 np.ndarray
        module affiliation vector

    Returns
    -------
    On : Nx1 np.ndarray
        new node order
    Ar : NxN np.ndarray
        reordered connectivity matrix
    '''
    # TODO update function with 2015 changes

    from scipy import stats
    _, max_module_size = stats.mode(ci)
    u, ci = np.unique(ci, return_inverse=True)  # make consecutive
    n = np.size(ci)  # number of nodes
    m = np.size(u)  # number of modules

    nm = np.zeros((m,))  # number of nodes in modules
    knm = np.zeros((n, m))  # degree to other modules

    for i in range(m):
        nm[i] = np.size(np.where(ci == i))
        knm[:, i] = np.sum(A[:, ci == i], axis=1)

    am = np.zeros((m, m))  # relative intermodular connectivity
    for i in range(m):
        am[i, :] = np.sum(knm[ci == i, :], axis=0)
    am /= np.outer(nm, nm)

    # 1. Arrange densely connected modules together
    # symmetrized intermodular connectivity
    i, j = np.where(np.tril(am, -1) + 1)
    s = (np.tril(am, -1) + 1)[i, j]
    ord = np.argsort(s)[::-1]  # sort by high relative connectivity
    i = i[ord]
    j = j[ord]
    i += 1
    j += 1  # fix off by 1 error so np.where doesnt
    om = np.array((i[0], j[0]))  # catch module 0
    i[0] = 0
    j[0] = 0
    while len(om) < m:  # while not all modules ordered
        ui, = np.where(np.logical_and(
            i, np.logical_or(j == om[0], j == om[-1])))
        uj, = np.where(np.logical_and(
            j, np.logical_or(i == om[0], i == om[-1])))

        if np.size(ui):
            ui = ui[0]
        if np.size(uj):
            uj = uj[0]

        if ui == uj:
            i[ui] = 0
            j[uj] = 0
            continue

        if not np.size(ui):
            ui = np.inf
        if not np.size(uj):
            uj = np.inf
        if ui < uj:
            old = j[ui]
            new = i[ui]
        if uj < ui:
            old = i[uj]
            new = j[uj]
        if old == om[0]:
            om = np.append((new,), om)
        if old == om[-1]:
            om = np.append(om, (new,))

        i[i == old] = 0
        j[j == old] = 0

    print(om)

    # 2. Reorder nodes within modules
    on = np.zeros((n,), dtype=int)
    for y, x in enumerate(om):
        ind, = np.where(ci == x - 1)  # indices
        pos, = np.where(om == x)  # position
        # NOT DONE! OE NOES

        mod_imp = np.array((om, np.sign(np.arange(m) - pos),
                            np.abs(np.arange(m) - pos), am[x - 1, om - 1])).T
        print(np.shape((mod_imp[:, 3][::-1], mod_imp[:, 2])))
        ix = np.lexsort((mod_imp[:, 3][::-1], mod_imp[:, 2]))
        mod_imp = mod_imp[ix]
        # at this point mod_imp agrees with the matlab version
        signs = mod_imp[:, 1]
        mod_imp = np.abs(mod_imp[:, 0] * mod_imp[:, 1])
        mod_imp = np.append(mod_imp[1:], x)
        mod_imp = np.array(mod_imp - 1, dtype=int)
        print(mod_imp, signs)
        # at this point mod_imp is the absolute value of that in the matlab
        # version.  this limitation comes from sortrows ability to deal with
        # negative indices, which we would have to do manually.

        # instead, i punt on its importance; i only bother to order by the
        # principal dimension.  some within-module orderings
        # may potentially be a little bit out of order.

        # ksmi=knm[ind,:].T[mod_imp[::-1]]
        # reverse mod_imp to sort by the first column first and so on
        # print ksmi
        # for i,sin in enumerate(signs):
        #	if sin==-1:
        #		ksmi[i,:]=ksmi[i,:][::-1]
        # print ksmi
        # print np.shape(ksmi)

        # ^ this is unworkable and wrong, lexsort alone cannot handle the
        # negative indices problem of sortrows.  you would pretty much need
        # to rewrite sortrows to do lexsort plus negative indices; the algorithm
        # cant be further simplified.

        ord = np.lexsort(knm[np.ix_(ind, mod_imp[::-1])])
        # ord=np.lexsort(knm[ind,:].T[mod_imp[::-1]])
        if signs[mod_imp[0]] == -1:
            ord = ord[::-1]
            # reverse just the principal level and punt on the other levels.
            # this will basically be fine for most purposes and probably won't
            # ever show a difference for weighted graphs.
        on[ind[ord]] = y * int(max_module_size) + \
            np.arange(nm[x - 1], dtype=int)

    on = np.argsort(on)
    ar = A[np.ix_(on, on)]

    return on, ar


def writetoPAJ(CIJ, fname, directed):
    '''
    This function writes a Pajek .net file from a numpy matrix

    Parameters
    ----------
    CIJ : NxN np.ndarray
        adjacency matrix
    fname : str
        filename
    directed : bool
        True if the network is directed and False otherwise. The data format
        may be required to know this for some reason so I am afraid to just
        use directed as the default value.
    '''
    n = np.size(CIJ, axis=0)
    with open(fname, 'w') as fd:
        fd.write('*vertices %i \r' % n)
        for i in range(1, n + 1):
            fd.write('%i "%i" \r' % (i, i))
        if directed:
            fd.write('*arcs \r')
        else:
            fd.write('*edges \r')
        for i in range(n):
            for j in range(n):
                if CIJ[i, j] != 0:
                    fd.write('%i %i %.6f \r' % (i + 1, j + 1, CIJ[i, j]))
