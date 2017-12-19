from __future__ import division, print_function
import numpy as np
from bct.utils import binarize


def degrees_dir(CIJ):
    '''
    Node degree is the number of links connected to the node. The indegree
    is the number of inward links and the outdegree is the number of
    outward links.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        directed binary/weighted connection matrix

    Returns
    -------
    id : Nx1 np.ndarray
        node in-degree
    od : Nx1 np.ndarray
        node out-degree
    deg : Nx1 np.ndarray
        node degree (in-degree + out-degree)

    Notes
    -----
    Inputs are assumed to be on the columns of the CIJ matrix.
           Weight information is discarded.
    '''
    CIJ = binarize(CIJ, copy=True)  # ensure CIJ is binary
    id = np.sum(CIJ, axis=0)  # indegree = column sum of CIJ
    od = np.sum(CIJ, axis=1)  # outdegree = row sum of CIJ
    deg = id + od  # degree = indegree+outdegree
    return id, od, deg


def degrees_und(CIJ):
    '''
    Node degree is the number of links connected to the node.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        undirected binary/weighted connection matrix

    Returns
    -------
    deg : Nx1 np.ndarray
        node degree

    Notes
    -----
    Weight information is discarded.
    '''
    CIJ = binarize(CIJ, copy=True)  # ensure CIJ is binary
    return np.sum(CIJ, axis=0)


def jdegree(CIJ):
    '''
    This function returns a matrix in which the value of each element (u,v)
    corresponds to the number of nodes that have u outgoing connections
    and v incoming connections.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        directed binary/weighted connnection matrix

    Returns
    -------
    J : ZxZ np.ndarray
        joint degree distribution matrix
        (shifted by one, replicates matlab one-based-indexing)
    J_od : int
        number of vertices with od>id
    J_id : int
        number of vertices with id>od
    J_bl : int
        number of vertices with id==od

    Notes
    -----
    Weights are discarded.
    '''
    CIJ = binarize(CIJ, copy=True)  # ensure CIJ is binary
    n = len(CIJ)
    id = np.sum(CIJ, axis=0)  # indegree = column sum of CIJ
    od = np.sum(CIJ, axis=1)  # outdegree = row sum of CIJ

    # create the joint degree distribution matrix
    # note: the matrix is shifted by one, to accomodate zero id and od in the
    # first row/column
    # upper triangular part of the matrix has vertices with od>id
    # lower triangular part has vertices with id>od
    # main diagonal has units with id=od

    szJ = np.max((id, od)) + 1
    J = np.zeros((szJ, szJ))

    for i in range(n):
        J[id[i], od[i]] += 1

    J_od = np.sum(np.triu(J, 1))
    J_id = np.sum(np.tril(J, -1))
    J_bl = np.sum(np.diag(J))
    return J, J_od, J_id, J_bl


def strengths_dir(CIJ):
    '''
    Node strength is the sum of weights of links connected to the node. The
    instrength is the sum of inward link weights and the outstrength is the
    sum of outward link weights.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        directed weighted connection matrix

    Returns
    -------
    is : Nx1 np.ndarray
        node in-strength
    os : Nx1 np.ndarray
        node out-strength
    str : Nx1 np.ndarray
        node strength (in-strength + out-strength)

    Notes
    -----
    Inputs are assumed to be on the columns of the CIJ matrix.
    '''
    istr = np.sum(CIJ, axis=0)
    ostr = np.sum(CIJ, axis=1)
    return istr + ostr


def strengths_und(CIJ):
    '''
    Node strength is the sum of weights of links connected to the node.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        undirected weighted connection matrix

    Returns
    -------
    str : Nx1 np.ndarray
        node strengths
    '''
    return np.sum(CIJ, axis=0)


def strengths_und_sign(W):
    '''
    Node strength is the sum of weights of links connected to the node.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected connection matrix with positive and negative weights

    Returns
    -------
    Spos : Nx1 np.ndarray
        nodal strength of positive weights
    Sneg : Nx1 np.ndarray
        nodal strength of positive weights
    vpos : float
        total positive weight
    vneg : float
        total negative weight
    '''
    W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)  # clear diagonal
    Spos = np.sum(W * (W > 0), axis=0)  # positive strengths
    Sneg = np.sum(W * (W < 0), axis=0) # negative strengths

    vpos = np.sum(W[W > 0])  # positive weight
    vneg = np.sum(W[W < 0])  # negative weight
    return Spos, Sneg, vpos, vneg
