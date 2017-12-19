from __future__ import division, print_function
import numpy as np
from .miscellaneous_utilities import BCTParamError


def threshold_absolute(W, thr, copy=True):
    '''
    This function thresholds the connectivity matrix by absolute weight
    magnitude. All weights below the given threshold, and all weights
    on the main diagonal (self-self connections) are set to 0.

    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    thr : float
        absolute weight threshold
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        thresholded connectivity matrix
    '''
    if copy:
        W = W.copy()
    np.fill_diagonal(W, 0)  # clear diagonal
    W[W < thr] = 0  # apply threshold
    return W


def threshold_proportional(W, p, copy=True):
    '''
    This function "thresholds" the connectivity matrix by preserving a
    proportion p (0<p<1) of the strongest weights. All other weights, and
    all weights on the main diagonal (self-self connections) are set to 0.

    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    p : float
        proportional weight threshold (0<p<1)
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        thresholded connectivity matrix

    Notes
    -----
    The proportion of elements set to 0 is a fraction of all elements
    in the matrix, whether or not they are already 0. That is, this function
    has the following behavior:

    >> x = np.random.random((10,10))
    >> x_25 = threshold_proportional(x, .25)
    >> np.size(np.where(x_25)) #note this double counts each nonzero element
    46
    >> x_125 = threshold_proportional(x, .125)
    >> np.size(np.where(x_125))
    22
    >> x_test = threshold_proportional(x_25, .5)
    >> np.size(np.where(x_test))
    46

    That is, the 50% thresholding of x_25 does nothing because >=50% of the
    elements in x_25 are aleady <=0. This behavior is the same as in BCT. Be
    careful with matrices that are both signed and sparse.
    '''
    from .miscellaneous_utilities import teachers_round as round

    if p > 1 or p < 0:
        raise BCTParamError('Threshold must be in range [0,1]')
    if copy:
        W = W.copy()
    n = len(W)						# number of nodes
    np.fill_diagonal(W, 0)			# clear diagonal

    if np.allclose(W, W.T):				# if symmetric matrix
        W[np.tril_indices(n)] = 0		# ensure symmetry is preserved
        ud = 2						# halve number of removed links
    else:
        ud = 1

    ind = np.where(W)					# find all links

    I = np.argsort(W[ind])[::-1]		# sort indices by magnitude

    en = int(round((n * n - n) * p / ud))		# number of links to be preserved

    W[(ind[0][I][en:], ind[1][I][en:])] = 0  # apply threshold
    #W[np.ix_(ind[0][I][en:], ind[1][I][en:])]=0

    if ud == 2:						# if symmetric matrix
        W[:, :] = W + W.T						# reconstruct symmetry

    return W


def weight_conversion(W, wcm, copy=True):
    '''
    W_bin = weight_conversion(W, 'binarize');
    W_nrm = weight_conversion(W, 'normalize');
    L = weight_conversion(W, 'lengths');

    This function may either binarize an input weighted connection matrix,
    normalize an input weighted connection matrix or convert an input
    weighted connection matrix to a weighted connection-length matrix.

    Binarization converts all present connection weights to 1.

    Normalization scales all weight magnitudes to the range [0,1] and
    should be done prior to computing some weighted measures, such as the
    weighted clustering coefficient.

    Conversion of connection weights to connection lengths is needed
    prior to computation of weighted distance-based measures, such as
    distance and betweenness centrality. In a weighted connection network,
    higher weights are naturally interpreted as shorter lengths. The
    connection-lengths matrix here is defined as the inverse of the
    connection-weights matrix.

    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : NxN np.ndarray
        weighted connectivity matrix
    wcm : str
        weight conversion command.
        'binarize' : binarize weights
        'normalize' : normalize weights
        'lengths' : convert weights to lengths (invert matrix)
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : NxN np.ndarray
        connectivity matrix with specified changes

    Notes
    -----
    This function is included for compatibility with BCT. But there are
    other functions binarize(), normalize() and invert() which are simpler to
    call directly.
    '''
    if wcm == 'binarize':
        return binarize(W, copy)
    elif wcm == 'normalize':
        return normalize(W, copy)
    elif wcm == 'lengths':
        return invert(W, copy)
    else:
        raise NotImplementedError('Unknown weight conversion command.')


def binarize(W, copy=True):
    '''
    Binarizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*

    Parameters
    ----------
    W : NxN np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : NxN np.ndarray
        binary connectivity matrix
    '''
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W


def normalize(W, copy=True):
    '''
    Normalizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        normalized connectivity matrix
    '''
    if copy:
        W = W.copy()
    W /= np.max(np.abs(W))
    return W


def invert(W, copy=True):
    '''
    Inverts elementwise the weights in an input connection matrix.
    In other words, change the from the matrix of internode strengths to the
    matrix of internode distances.

    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        inverted connectivity matrix
    '''
    if copy:
        W = W.copy()
    E = np.where(W)
    W[E] = 1. / W[E]
    return W


def autofix(W, copy=True):
    '''
    Fix a bunch of common problems. More specifically, remove Inf and NaN,
    ensure exact binariness and symmetry (i.e. remove floating point
    instability), and zero diagonal.


    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        connectivity matrix with fixes applied
    '''
    if copy:
        W = W.copy()

    # zero diagonal
    np.fill_diagonal(W, 0)

    # remove np.inf and np.nan
    W[np.logical_or(np.where(np.isinf(W)), np.where(np.isnan(W)))] = 0

    # ensure exact binarity
    u = np.unique(W)
    if np.all(np.logical_or(np.abs(u) < 1e-8, np.abs(u - 1) < 1e-8)):
        W = np.around(W, decimal=5)

    # ensure exact symmetry
    if np.allclose(W, W.T):
        W = np.around(W, decimals=5)

    return W
