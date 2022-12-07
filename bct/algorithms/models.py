from __future__ import division, print_function
import numpy as np

from bct.utils import BCTParamError, get_rng
from ..due import due, BibTex

def mleme_constraint_model(nr_samples, W, ci=None, lo=None, li=None, lm=None, 
                           seed=None):
    '''
    This function returns an ensemble of unbiasedly sampled networks with
    weighted node-strength and module-weight constraints. These constraints
    are soft in that they are satisfied on average for the full network
    ensemble but not, in general, for each individual network.

    Parameters
    ----------
    W : np.ndarray
        NxN square directed weighted connectivity matrix. All inputs must be
        nonnegative integers. Real valued weights could be converted to
        integers through rescaling and rounding.
    ci : np.ndarray
        Nx1 module affiliation vector. Can be None if there are no module
        constraints. Must contain nonnegative integers. The default value
        is None.
    lo : np.ndarray
        Nx1 out strength constraing logical vector. This vector specifies
        out strength constraints for each node. Alternately, it can be
        True to constrain all out-strengths or None for no constraints.
        The default value is None.
    li : np.ndarray
        Nx1 in strength constraing logical vector. This vector specifies
        in strength constraints for each node. Alternately, it can be
        True to constrain all in-strengths or None for no constraints.
        The default value is None.
    lm : np.ndarray
        Mx1 module-weight constraint logical matrix where M is the number of
        modules. Specifies module-weight constraints for all pairs of modules.
        Can be True, 'all', or 2, to constrain all inter-module and 
        intra-module weights, 'intra' or 1 to constrain all intra-module
        weights only, or None for no constraints. The default value is None.
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.

    Returns
    -------
    W0 : np.ndarray
        NxNxnr_samples an ensemble of sampled networks with constraints
    E0 : np.ndarray
        expected weights matrix
    P0 : np.ndarray
        probability matrix
    delt0 : float
        algorithm convergence error
    '''
    rng = get_rng(seed)
    raise NotImplementedError()
