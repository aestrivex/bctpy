from __future__ import division, print_function
import numpy as np

from bct.utils import BCTParamError
from .similarity import matching_ind
from .clustering import clustering_coef_bu
from .centrality import betweenness_bin


def generative_model(A, D, m, eta, gamma=None, model_type='matching', 
    model_var='powerlaw', epsilon=1e-6, copy=True):
    '''
    Generates synthetic networks using the models described in
    Betzel et al. (2016) Neuroimage. See this paper for more details.

    Succinctly, the probability of forming a connection between nodes u and v is
    P(u,v) = E(u,v)**eta * K(u,v)**gamma
    where eta and gamma are hyperparameters, E(u,v) is the euclidean or similar
    distance measure, and K(u,v) is the algorithm that defines the model.

    This describes the power law formulation, an alternative formulation uses
    the exponential function
    P(u,v) = exp(E(u,v)*eta) * exp(K(u,v)*gamma)

    Parameters
    ----------
    A : np.ndarray
        Binary network of seed connections
    D : np.ndarray
        Matrix of euclidean distances or other distances between nodes
    m : int
        Number of connections that should be present in the final synthetic 
        network
    eta : np.ndarray
        A vector describing a range of values to estimate for eta, the 
        hyperparameter describing exponential weighting of the euclidean
        distance.
    gamma : np.ndarray
        A vector describing a range of values to estimate for theta, the
        hyperparameter describing exponential weighting of the basis
        algorithm. If model_type='euclidean' or another distance metric,
        this can be None.
    model_type : Enum(str)
        euclidean : Uses only euclidean distances to generate connection 
            probabilities
        neighbors : count of common neighbors
        matching : matching index, the normalized overlap in neighborhoods
        clu-avg : Average clustering coefficient
        clu-min : Minimum clustering coefficient
        clu-max : Maximum clustering coefficient
        clu-diff : Difference in clustering coefficient
        clu-prod : Product of clustering coefficient
        deg-avg : Average degree
        deg-min : Minimum degree
        deg-max : Maximum degree
        deg-diff : Difference in degree
        deg-prod : Product of degrees
    model_var : Enum(str)
        Default value is powerlaw. If so, uses formulation of P(u,v) as
        described above. Alternate value is exponential. If so, uses
        P(u,v) = exp(E(u,v)*eta) * exp(K(u,v)*gamma)
    epsilon : float
        A small positive value added to all P(u,v). The default value is 1e-6
    copy : bool
        Some algorithms add edges directly to the input matrix. Set this flag
        to make a copy of the input matrix instead. Defaults to True.
    '''

    if copy:
        A = A.copy()

    n = len(D)
    
    #These parameters don't do any of the voronoi narrowing.
    #Its a list of eta values paired with gamma values.
    #To try 3 eta and 3 gamma pairs, should use 9 list values.
    if len(eta) != len(gamma):
        raise BCTParamError('Eta and gamma hyperparameters must be lists of '
            'the same size')

    nparams = len(eta)

    B = np.zeros((n, n, nparams))

    def k_avg(K):
        return ((np.tile(K, (n, 1)) + np.transpose(np.tile(K, (n, 1))))/2 +
            epsilon)

    def k_diff(K):
        return np.abs(np.tile(K, (n, 1)) - 
                      np.transpose(np.tile(K, (n, 1)))) + epsilon

    def k_max(K):
        return np.max(np.dstack((np.tile(K, (n, 1)),
                                 np.transpose(np.tile(K, (n, 1))))),
                      axis=2) + epsilon

    def k_min(K):
        return np.min(np.dstack((np.tile(K, (n, 1)),
                                 np.transpose(np.tile(K, (n, 1))))),
                      axis=2) + epsilon

    def k_prod(K):
        return np.outer(K, np.transpose(K)) + epsilon

    def s_avg(K, sc):
        return (K+sc) / 2 + epsilon

    def s_diff(K, sc):
        return np.abs(K-sc) + epsilon

    def s_min(K, sc):
        return np.where(K < sc, K + epsilon, sc + epsilon)
    
    def s_max(K, sc):
        #return np.max((K, sc.T), axis=0)
        return np.where(K > sc, K + epsilon, sc + epsilon)

    def s_prod(K, sc):
        return K * sc + epsilon

    def x_avg(K, ixes):
        nr_ixes = np.size(np.where(ixes))
        Ksc = np.tile(K, (nr_ixes, 1))
        Kix = np.transpose(np.tile(K[ixes], (n, 1)))
        return s_avg(Ksc, Kix)

    def x_diff(K, ixes):
        nr_ixes = np.size(np.where(ixes))
        Ksc = np.tile(K, (nr_ixes, 1))
        Kix = np.transpose(np.tile(K[ixes], (n, 1)))
        return s_diff(Ksc, Kix)

    def x_max(K, ixes):
        nr_ixes = np.size(np.where(ixes))
        Ksc = np.tile(K, (nr_ixes, 1))
        Kix = np.transpose(np.tile(K[ixes], (n, 1)))
        return s_max(Ksc, Kix)

    def x_min(K, ixes):
        nr_ixes = np.size(np.where(ixes))
        Ksc = np.tile(K, (nr_ixes, 1))
        Kix = np.transpose(np.tile(K[ixes], (n, 1)))
        return s_min(Ksc, Kix)

    def x_prod(K, ixes):
        nr_ixes = np.size(np.where(ixes))
        Ka = np.reshape(K[ixes], (nr_ixes, 1))
        Kb = np.reshape(np.transpose(K), (1, n))
        return np.outer(Ka, Kb) + epsilon


    def clu_gen(A, K, D, m, eta, gamma, model_var, x_fun):
        mseed = np.size(np.where(A.flat))//2

        A = A>0

        if type(model_var) == tuple:
            mv1, mv2 = model_var
        else:
            mv1, mv2 = model_var, model_var

        if mv1 in ('powerlaw', 'power_law'):
            Fd = D**eta
        elif mv1 in ('exponential',):
            Fd = np.exp(eta*D) 

        if mv2 in ('powerlaw', 'power_law'):
            Fk = K**gamma
        elif mv2 in ('exponential',):
            Fk = np.exp(gamma*K) 

        c = clustering_coef_bu(A)
        k = np.sum(A, axis=1)

        Ff = Fd * Fk * np.logical_not(A)
        u,v = np.where(np.triu(np.ones((n,n)), 1))

        #print(mseed, m)
        for i in range(mseed+1, m):
            C = np.append(0, np.cumsum(Ff[u,v]))
            r = np.sum(np.random.random()*C[-1] >= C)
            uu = u[r]
            vv = v[r]
            A[uu,vv] = A[vv,uu] = 1
            k[uu] += 1
            k[vv] += 1

            bu = A[uu,:].astype(bool)
            bv = A[vv,:].astype(bool)
            su = A[np.ix_(bu, bu)]
            sv = A[np.ix_(bu, bu)]

            bth = np.logical_and(bu, bv)
            c[bth] += 2/(k[bth]**2 - k[bth])
            c[uu] = np.size(np.where(su.flat))/(k[uu]*(k[uu]-1))
            c[vv] = np.size(np.where(sv.flat))/(k[vv]*(k[vv]-1))
            c[k<=1] = 0
            bth[uu] = 1
            bth[vv] = 1
    
            k_result = x_fun(c, bth)

            #print(np.shape(k_result))
            #print(np.shape(K))
            #print(K)
            #print(np.shape(K[bth,:]))

            K[bth,:] = k_result
            K[:,bth] = k_result.T

            if mv2 in ('powerlaw', 'power_law'):
                Ff[bth,:] = Fd[bth,:] * K[bth,:]**gamma
                Ff[:,bth] = Fd[:,bth] * K[:,bth]**gamma
            elif mv2 in ('exponential',):
                Ff[bth,:] = Fd[bth,:] * np.exp(K[bth,:])*gamma
                Ff[:,bth] = Fd[:,bth] * np.exp(K[:,bth])*gamma

            Ff = Ff * np.logical_not(A)

        return A

    def deg_gen(A, K, D, m, eta, gamma, model_var, s_fun):
        mseed = np.size(np.where(A.flat))//2

        k = np.sum(A, axis=1)

        if type(model_var) == tuple:
            mv1, mv2 = model_var
        else:
            mv1, mv2 = model_var, model_var

        if mv1 in ('powerlaw', 'power_law'):
            Fd = D**eta
        elif mv1 in ('exponential',):
            Fd = np.exp(eta*D) 

        if mv2 in ('powerlaw', 'power_law'):
            Fk = K**gamma
        elif mv2 in ('exponential',):
            Fk = np.exp(gamma*K) 

        P = Fd * Fk * np.logical_not(A)
        u,v = np.where(np.triu(np.ones((n,n)), 1))

        b = np.zeros((m,), dtype=int)

#        print(mseed)
#        print(np.shape(u),np.shape(v))
#        print(np.shape(b))
#        print(np.shape(A[u,v]))
#        print(np.shape(np.where(A[u,v])), 'sqishy')
#        print(np.shape(P), 'squnnaq')

        #b[:mseed] = np.where(A[np.ix_(u,v)]) 
        b[:mseed] = np.squeeze(np.where(A[u,v]))
        #print(mseed, m)
        for i in range(mseed, m):
            C = np.append(0, np.cumsum(P[u,v]))
            r = np.sum(np.random.random()*C[-1] >= C)
            uu = u[r]
            vv = v[r]
            k[uu] += 1
            k[vv] += 1

            if mv2 in ('powerlaw', 'power_law'):
                Fk[:,uu] = Fk[uu,:] = s_fun(k, k[uu]) ** gamma
                Fk[:,vv] = Fk[vv,:] = s_fun(k, k[vv]) ** gamma
            elif mv2 in ('exponential',):
                Fk[:,uu] = Fk[uu,:] = np.exp(s_fun(k, k[uu]) * gamma)
                Fk[:,vv] = Fk[vv,:] = np.exp(s_fun(k, k[vv]) * gamma)

            P = Fd * Fk

            b[i] = r

            P[u[b[:i]], v[b[:i]]] = P[v[b[:i]], u[b[:i]]] = 0

            A[u[r], v[r]] = A[v[r], u[r]] = 1
            #P[b[u[:i]], b[v[:i]]] = P[b[v[:i]], b[u[:i]]] = 0

            #A[uu,vv] = A[vv,uu] = 1


#        indx = v*n + u
#        indx[b]
#
#        nH = np.zeros((n,n))
#        nH.ravel()[indx[b]]=1
#
#        nG = np.zeros((n,n))
#        nG[ u[b], v[b] ]=1
#        nG = nG + nG.T
#
#        print(np.shape(np.where(A != nG)))
#
#        import pdb
#        pdb.set_trace()

        return A

    def matching_gen(A, K, D, m, eta, gamma, model_var):
        K += epsilon

        mseed = np.size(np.where(A.flat))//2

        if type(model_var) == tuple:
            mv1, mv2 = model_var
        else:
            mv1, mv2 = model_var, model_var

        if mv1 in ('powerlaw', 'power_law'):
            Fd = D**eta
        elif mv1 in ('exponential',):
            Fd = np.exp(eta*D) 

        if mv2 in ('powerlaw', 'power_law'):
            Fk = K**gamma
        elif mv2 in ('exponential',):
            Fk = np.exp(gamma*K) 

        Ff = Fd * Fk * np.logical_not(A)
        u,v = np.where(np.triu(np.ones((n,n)), 1))
    
        for ii in range(mseed, m):
            C = np.append(0, np.cumsum(Ff[u,v]))
            r = np.sum(np.random.random()*C[-1] >= C)
            uu = u[r]
            vv = v[r]
            A[uu,vv] = A[vv,uu] = 1

            updateuu, = np.where(np.inner(A, A[:,uu]))
            np.delete(updateuu, np.where(updateuu == uu))
            np.delete(updateuu, np.where(updateuu == vv))

            c1 = np.append(A[:,uu], A[uu,:])
            for i in range(len(updateuu)):
                j = updateuu[i]
                c2 = np.append(A[:,j], A[j,:])
    
                use = np.logical_or(c1, c2)
                use[uu] = use[uu+n] = use[j] = use[j+n] = 0
                ncon = np.sum(c1[use]) + np.sum(c2[use])
                if ncon == 0:
                    K[uu, j] = K[j, uu] = epsilon
                else:
                    K[uu, j] = K[j, uu] = (2 / ncon *
                        np.sum(np.logical_and(c1[use], c2[use])) + epsilon)

            updatevv, = np.where(np.inner(A, A[:,vv]))
            np.delete(updatevv, np.where(updatevv == uu))
            np.delete(updatevv, np.where(updatevv == vv))
        
            c1 = np.append(A[:,vv], A[vv,:])
            for i in range(len(updatevv)):
                j = updatevv[i]
                c2 = np.append(A[:,j], A[j,:])
    
                use = np.logical_or(c1, c2)
                use[vv] = use[vv+n] = use[j] = use[j+n] = 0
                ncon = np.sum(c1[use]) + np.sum(c2[use])
                if ncon == 0:
                    K[vv, j] = K[j, vv] = epsilon
                else:
                    K[vv, j] = K[j, vv] = (2 / ncon *
                        np.sum(np.logical_and(c1[use], c2[use])) + epsilon)

            Ff = Fd * Fk * np.logical_not(A)

        return A
    
    def neighbors_gen(A, K, D, m, eta, gamma, model_var):
        K += epsilon

        mseed = np.size(np.where(A.flat))//2

        if type(model_var) == tuple:
            mv1, mv2 = model_var
        else:
            mv1, mv2 = model_var, model_var

        if mv1 in ('powerlaw', 'power_law'):
            Fd = D**eta
        elif mv1 in ('exponential',):
            Fd = np.exp(eta*D) 

        if mv2 in ('powerlaw', 'power_law'):
            Fk = K**gamma
        elif mv2 in ('exponential',):
            Fk = np.exp(gamma*K) 

        Ff = Fd * Fk * np.logical_not(A)
        u,v = np.where(np.triu(np.ones((n,n)), 1))
    
        for ii in range(mseed, m):
            C = np.append(0, np.cumsum(Ff[u,v]))
            r = np.sum(np.random.random()*C[-1] >= C)
            uu = u[r]
            vv = v[r]
            A[uu, vv] = A[vv, uu] = 1

            x = A[uu, :].astype(int)
            y = A[:, vv].astype(int)
    
            K[uu, y] += 1
            K[y, uu] += 1
            K[vv, x] += 1
            K[x, vv] += 1

            if mv2 in ('powerlaw', 'power_law'):
                Fk = K**gamma
            elif mv2 in ('exponential',):
                Fk = np.exp(gamma*K) 

            if mv2 in ('powerlaw', 'power_law'):
                Ff[uu, y] = Ff[y, uu] = Fd[uu, y] * (K[uu, y] ** gamma)
                Ff[vv, x] = Ff[x, vv] = Fd[vv, x] * (K[vv, x] ** gamma)
            elif mv2 in ('exponential',):
                Ff[uu, y] = Ff[y, uu] = Fd[uu, y] * np.exp(gamma * K[uu, y])
                Ff[vv, x] = Ff[x, vv] = Fd[vv, x] * np.exp(gamma * K[vv, x])

            Ff[np.where(A)] = 0

        return A

    def euclidean_gen(A, D, m, eta, model_var):
        mseed = np.size(np.where(A.flat))//2

        if type(model_var) == tuple:
            mv1, mv2 = model_var
        else:
            mv1, mv2 = model_var, model_var

        if mv1 != mv2:
            raise BCTParamError('Too many hyperparameters specified')

        if mv1 in ('powerlaw', 'power_law'):
            Fd = D ** eta
        elif mv1 in ('exponential',):
            Fd = np.exp(eta ** D)

        u,v = np.where(np.triu(np.ones((n,n)), 1))
        P = Fd * np.logical_not(A)

        b = np.zeros((m,), dtype=int)
        b[:mseed] = np.squeeze(np.where(A[u, v]))
        for i in range(mseed, m):
            C = np.append(0, np.cumsum(P[u, v]))
            r = np.sum(np.random.random()*C[-1] >= C)
            b[i] = r
            P = Fd
            P[u[b[:i]], v[b[:i]]] = P[v[b[:i]], u[b[:i]]] = 0

            A[u[r], v[r]] = A[v[r], u[r]] = 1

        return A

    if model_type in ('clu-avg', 'clu_avg'):
        Kseed = k_avg(clustering_coef_bu(A))
        for j, (ep, gp) in enumerate(zip(eta, gamma)):
            B[:,:,j] = clu_gen(A, Kseed, D, m, ep, gp, model_var, x_avg)

    elif model_type in ('clu-diff', 'clu_diff'):
        Kseed = k_diff(clustering_coef_bu(A))
        for j, (ep, gp) in enumerate(zip(eta, gamma)):
            B[:,:,j] = clu_gen(A, Kseed, D, m, ep, gp, model_var, x_diff)

    elif model_type in ('clu-max', 'clu_max'):
        Kseed = k_max(clustering_coef_bu(A))
        for j, (ep, gp) in enumerate(zip(eta, gamma)):
            B[:,:,j] = clu_gen(A, Kseed, D, m, ep, gp, model_var, x_max) 

    elif model_type in ('clu-min', 'clu_min'):
        Kseed = k_min(clustering_coef_bu(A))
        for j, (ep, gp) in enumerate(zip(eta, gamma)):
            B[:,:,j] = clu_gen(A, Kseed, D, m, ep, gp, model_var, x_min) 

    elif model_type in ('clu-prod', 'clu_prod'):
        Kseed = k_prod(clustering_coef_bu(A))
        for j, (ep, gp) in enumerate(zip(eta, gamma)):
            B[:,:,j] = clu_gen(A, Kseed, D, m, ep, gp, model_var, x_prod)

    elif model_type in ('deg-avg', 'deg_avg'):
        Kseed = k_avg(np.sum(A, axis=1))
        for j, (ep, gp) in enumerate(zip(eta, gamma)):
            B[:,:,j] = deg_gen(A, Kseed, D, m, ep, gp, model_var, s_avg)

    elif model_type in ('deg-diff', 'deg_diff'):
        Kseed = k_diff(np.sum(A, axis=1))
        for j, (ep, gp) in enumerate(zip(eta, gamma)):
            B[:,:,j] = deg_gen(A, Kseed, D, m, ep, gp, model_var, s_diff)
    
    elif model_type in ('deg-max', 'deg_max'):
        Kseed = k_max(np.sum(A, axis=1))
        for j, (ep, gp) in enumerate(zip(eta, gamma)):
            B[:,:,j] = deg_gen(A, Kseed, D, m, ep, gp, model_var, s_max)

    elif model_type in ('deg-min', 'deg_min'):
        Kseed = k_min(np.sum(A, axis=1))
        for j, (ep, gp) in enumerate(zip(eta, gamma)):
            B[:,:,j] = deg_gen(A, Kseed, D, m, ep, gp, model_var, s_min)

    elif model_type in ('deg-prod', 'deg_prod'):
        Kseed = k_prod(np.sum(A, axis=1))
        for j, (ep, gp) in enumerate(zip(eta, gamma)):
            B[:,:,j] = deg_gen(A, Kseed, D, m, ep, gp, model_var, s_prod)

    elif model_type in ('neighbors',):
        Kseed = np.inner(A, A)
        np.fill_diagonal(Kseed, 0)
        for j, (ep, gp) in enumerate(zip(eta, gamma)):
            B[:,:,j] = neighbors_gen(A, Kseed, D, m, ep, gp, model_var)

    elif model_type in ('matching', 'matching-ind', 'matching_ind'):
        mi, _, _ = matching_ind(A)
        Kseed = mi + mi.T
        for j, (ep, gp) in enumerate(zip(eta, gamma)):
            B[:,:,j] = matching_gen(A, Kseed, D, m, ep, gp, model_var)

    elif model_type in ('spatial', 'geometric', 'euclidean'):
        for j, ep in enumerate(eta):
            B[:,:,j] = euclidean_gen(A, D, m, ep, model_var) 

    return np.squeeze(B)

def evaluate_generative_model(A, Atgt, D, eta, gamma=None, 
    model_type='matching', model_var='powerlaw', epsilon=1e-6):
    '''
    Generates synthetic networks with parameters provided and evaluates their
    energy function. The energy function is defined as in Betzel et al. 2016.
    Basically it takes the Kolmogorov-Smirnov statistics of 4 network
    measures; comparing the degree distributions, clustering coefficients,
    betweenness centrality, and Euclidean distances between connected regions.
    
    The energy is globally low if the synthetic network matches the target.
    Energy is defined as the maximum difference across the four statistics.
    '''
    m = np.size(np.where(Atgt.flat))//2
    n = len(Atgt)
    xk = np.sum(Atgt, axis=1)
    xc = clustering_coef_bu(Atgt)
    xb = betweenness_bin(Atgt)
    xe = D[np.triu(Atgt, 1) > 0]

    B = generative_model(A, D, m, eta, gamma, model_type=model_type, 
                         model_var=model_var, epsilon=epsilon, copy=True)

    #if eta != gamma then an error is thrown within generative model
    
    nB = len(eta)

    if nB == 1:
        B = np.reshape(B, np.append(np.shape(B), 1))

    K = np.zeros((nB, 4))

    def kstats(x, y):
        bin_edges = np.concatenate([[-np.inf],
                                    np.sort(np.concatenate((x, y))), 
                                    [np.inf]])

        bin_x,_ = np.histogram(x, bin_edges)
        bin_y,_ = np.histogram(y, bin_edges)

        #print(np.shape(bin_x))

        sum_x = np.cumsum(bin_x) / np.sum(bin_x)
        sum_y = np.cumsum(bin_y) / np.sum(bin_y)

        cdfsamp_x = sum_x[:-1]
        cdfsamp_y = sum_y[:-1]

        delta_cdf = np.abs(cdfsamp_x - cdfsamp_y)

        print(np.shape(delta_cdf))
        #print(delta_cdf)
        print(np.argmax(delta_cdf), np.max(delta_cdf))

        return np.max(delta_cdf)

    for ib in range(nB):
        Bc = B[:,:,ib]
        yk = np.sum(Bc, axis=1)
        yc = clustering_coef_bu(Bc)
        yb = betweenness_bin(Bc)
        ye = D[np.triu(Bc, 1) > 0]

        K[ib, 0] = kstats(xk, yk)
        K[ib, 1] = kstats(xc, yc)
        K[ib, 2] = kstats(xb, yb)
        K[ib, 3] = kstats(xe, ye)

    return np.max(K, axis=1)
