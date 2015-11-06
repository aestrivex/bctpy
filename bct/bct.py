#	(C) Roan LaPlante 2013 rlaplant@nmr.mgh.harvard.edu
#
#	This program is BCT-python, the Brain Connectivity Toolbox for python.
#
#	BCT-python is based on BCT, the Brain Connectivity Toolbox.  BCT is the
# 	collaborative work of many contributors, and is maintained by Olaf Sporns
#	and Mikail Rubinov.  For the full list, see the associated contributors.
#
#	This program is free software; you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np

class BCTParamError(RuntimeError): pass

##############################################################################
# CENTRALITY
##############################################################################

def betweenness_bin(G):
    '''
    Node betweenness centrality is the fraction of all shortest paths in 
    the network that contain a given node. Nodes with high values of 
    betweenness centrality participate in a large number of shortest paths.

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed/undirected connection matrix

    BC : Nx1 np.ndarray
        node betweenness centrality vector

    Notes
    -----
    Betweenness centrality may be normalised to the range [0,1] as
    BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    '''
    G=np.array(G, dtype=float)      #force G to have float type so it can be
                                    #compared to float np.inf
    
    n=len(G)						#number of nodes
    I=np.eye(n)						#identity matrix
    d=1								#path length
    NPd=G.copy()					#number of paths of length |d|
    NSPd=G.copy()					#number of shortest paths of length |d|
    NSP=G.copy();					#number of shortest paths of any length
    L=G.copy();						#length of shortest paths
    
    NSP[np.where(I)]=1; L[np.where(I)]=1

    #calculate NSP and L
    while np.any(NSPd):
        d+=1
        NPd=np.dot(NPd,G)
        NSPd=NPd*(L==0)
        NSP+=NSPd
        L=L+d*(NSPd!=0)

    L[L==0]=np.inf					#L for disconnected vertices is inf
    L[np.where(I)]=0
    NSP[NSP==0]=1					#NSP for disconnected vertices is 1

    DP=np.zeros((n,n))				#vertex on vertex dependency
    diam=d-1

    #calculate DP
    for d in range(diam,1,-1):
        DPd1=np.dot(((L==d)*(1+DP)/NSP),G.T)*((L==(d-1))*NSP)	
        DP+=DPd1
        
    return np.sum(DP,axis=0)

def betweenness_wei(G):
    '''
    Node betweenness centrality is the fraction of all shortest paths in 
    the network that contain a given node. Nodes with high values of 
    betweenness centrality participate in a large number of shortest paths.

    Parameters
    ----------
    L : NxN np.ndarray
        directed/undirected weighted connection matrix

    Returns
    -------
    BC : Nx1 np.ndarray
        node betweenness centrality vector

    Notes
    -----
       The input matrix must be a connection-length matrix, typically
        obtained via a mapping from weight to length. For instance, in a
        weighted correlation network higher correlations are more naturally
        interpreted as shorter distances and the input matrix should
        consequently be some inverse of the connectivity matrix. 
       Betweenness centrality may be normalised to the range [0,1] as
        BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    '''
    n=len(G)
    BC=np.zeros((n,))					#vertex betweenness

    for u in range(n):
        D=np.tile(np.inf,(n,)); D[u]=0	#distance from u
        NP=np.zeros((n,)); NP[u]=1		#number of paths from u
        S=np.ones((n,),dtype=bool)		#distance permanence
        P=np.zeros((n,n))				#predecessors
        Q=np.zeros((n,)); q=n-1			#order of non-increasing distance

        G1=G.copy()
        V=[u]
        while True:
            S[V]=0						#distance u->V is now permanent	
            G1[:,V]=0					#no in-edges as already shortest
            for v in V:
                Q[q]=v
                q-=1
                W,=np.where(G1[v,:])		#neighbors of v
                for w in W:
                    Duw=D[v]+G1[v,w]	#path length to be tested
                    if Duw<D[w]:		#if new u->w shorter than old
                        D[w]=Duw
                        NP[w]=NP[v]		#NP(u->w) = NP of new path
                        P[w,:]=0
                        P[w,v]=1		#v is the only predecessor
                    elif Duw==D[w]:		#if new u->w equal to old
                        NP[w]+=NP[v]	#NP(u->w) sum of old and new
                        P[w,v]=1		#v is also predecessor

            if D[S].size==0:
                break					#all nodes were reached
            if np.isinf(np.min(D[S])):	#some nodes cannot be reached
                Q[:q+1],=np.where(np.isinf(D)) #these are first in line
                break
            V,=np.where(D==np.min(D[S]))

        DP=np.zeros((n,))
        for w in Q[:n-1]:
            BC[w]+=DP[w]
            for v in np.where(P[w,:])[0]:
                DP[v]+=(1+DP[w])*NP[v]/NP[w]

    return BC

def diversity_coef_sign(W,ci):
    '''
    The Shannon-entropy based diversity coefficient measures the diversity
    of intermodular connections of individual nodes and ranges from 0 to 1.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected connection matrix with positive and negative weights
    ci : Nx1 np.ndarray
        community affiliation vector

    Returns
    -------
    Hpos : Nx1 np.ndarray
        diversity coefficient based on positive connections
    Hneg : Nx1 np.ndarray
        diversity coefficient based on negative connections
    '''
    n=len(W)							#number of nodes

    _,ci=np.unique(ci,return_inverse=True)
    ci += 1

    m=np.max(ci)						#number of modules		

    def entropy(w_):
        S=np.sum(w_,axis=1)				#strength
        Snm=np.zeros((n,m))				#node-to-module degree
        for i in range(m):
            Snm[:,i]=np.sum(w_[:,ci==i+1],axis=1)
        pnm=Snm/(np.tile(S,(m,1)).T)
        pnm[np.isnan(pnm)]=0
        pnm[np.logical_not(pnm)]=1
        return -np.sum(pnm*np.log(pnm),axis=1)/np.log(m)

    Hpos=entropy(W*(W>0))
    Hneg=entropy(-W*(W<0))

    return Hpos,Hneg

def edge_betweenness_bin(G):
    '''
    Edge betweenness centrality is the fraction of all shortest paths in 
    the network that contain a given edge. Edges with high values of 
    betweenness centrality participate in a large number of shortest paths.

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed/undirected connection matrix

    Returns
    -------
    EBC : NxN np.ndarray
        edge betweenness centrality matrix
    BC : Nx1 np.ndarray
        node betweenness centrality vector

    Notes
    -----
    Betweenness centrality may be normalised to the range [0,1] as
    BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    '''
    n=len(G)
    BC=np.zeros((n,))					#vertex betweenness
    EBC=np.zeros((n,n))					#edge betweenness

    for u in range(n):
        D=np.zeros((n,)); D[u]=1		#distance from u
        NP=np.zeros((n,)); NP[u]=1		#number of paths from u
        P=np.zeros((n,n))				#predecessors
        Q=np.zeros((n,)); q=n-1			#order of non-increasing distance

        Gu=G.copy()	
        V=np.array([u])
        while V.size:
            Gu[:,V]=0					#remove remaining in-edges
            for v in V:
                Q[q]=v;
                q-=1
                W,=np.where(Gu[v,:])		#neighbors of V
                for w in W:
                    if D[w]:
                        NP[w]+=NP[v]	#NP(u->w) sum of old and new
                        P[w,v]=1		#v is a predecessor
                    else:
                        D[w]=1
                        NP[w]=NP[v]		#NP(u->v) = NP of new path
                        P[w,v]=1		#v is a predecessor
            V,=np.where(np.any(Gu[V,:],axis=0))

        if np.any(np.logical_not(D)):	# if some vertices unreachable
            Q[:q],=np.where(np.logical_not(D))	#...these are first in line

        DP=np.zeros((n,))				# dependency
        for w in Q[:n-1]:
            BC[w]+=DP[w]
            for v in np.where(P[w,:])[0]:
                DPvw=(1+DP[w])*NP[v]/NP[w]	
                DP[v]+=DPvw
                EBC[v,w]+=DPvw

    return EBC,BC

def edge_betweenness_wei(G):
    '''
    Edge betweenness centrality is the fraction of all shortest paths in 
    the network that contain a given edge. Edges with high values of 
    betweenness centrality participate in a large number of shortest paths.

    Parameters
    ----------
    L : NxN np.ndarray
        directed/undirected weighted connection matrix

    Returns
    -------
    EBC : NxN np.ndarray
        edge betweenness centrality matrix
    BC : Nx1 np.ndarray
        nodal betweenness centrality vector

    Notes
    -----
    The input matrix must be a connection-length matrix, typically
        obtained via a mapping from weight to length. For instance, in a
        weighted correlation network higher correlations are more naturally
        interpreted as shorter distances and the input matrix should
        consequently be some inverse of the connectivity matrix. 
    Betweenness centrality may be normalised to the range [0,1] as
        BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    '''
    n=len(G)	
    BC=np.zeros((n,))					#vertex betweenness
    EBC=np.zeros((n,n))					#edge betweenness

    for u in range(n):
        D=np.tile(np.inf,n); D[u]=0		#distance from u
        NP=np.zeros((n,)); NP[u]=1		#number of paths from u
        S=np.ones((n,),dtype=bool)		#distance permanence
        P=np.zeros((n,n))				#predecessors
        Q=np.zeros((n,)); q=n-1			#order of non-increasing distance

        G1=G.copy()
        V=[u]
        while True:
            S[V]=0						#distance u->V is now permanent
            G1[:,V]=0					#no in-edges as already shortest
            for v in V:
                Q[q]=v
                q-=1
                W,=np.where(G1[v,:])	#neighbors of v
                for w in W:
                    Duw=D[v]+G1[v,w]	#path length to be tested
                    if Duw<D[w]:		#if new u->w shorter than old
                        D[w]=Duw
                        NP[w]=NP[v]		#NP(u->w) = NP of new path
                        P[w,:]=0
                        P[w,v]=1		#v is the only predecessor
                    elif Duw==D[w]:		#if new u->w equal to old
                        NP[w]+=NP[v]	#NP(u->w) sum of old and new
                        P[w,v]=1		#v is also a predecessor

            if D[S].size==0:
                break					#all nodes reached, or
            if np.isinf(np.min(D[S])):	#some cannot be reached
                Q[:q],=np.where(np.isinf(D)) #these are first in line
                break
            V,=np.where(D==np.min(D[S]))

        DP=np.zeros((n,))				#dependency
        for w in Q[:n-1]:
            BC[w]+=DP[w]
            for v in np.where(P[w,:])[0]:
                DPvw=(1+DP[w])*NP[v]/NP[w]
                DP[v]+=DPvw
                EBC[v,w]+=DPvw

    return EBC,BC

def eigenvector_centrality_und(CIJ):
    '''
    Eigenector centrality is a self-referential measure of centrality:
    nodes have high eigenvector centrality if they connect to other nodes
    that have high eigenvector centrality. The eigenvector centrality of
    node i is equivalent to the ith element in the eigenvector 
    corresponding to the largest eigenvalue of the adjacency matrix.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary/weighted undirected adjacency matrix

    v : Nx1 np.ndarray
        eigenvector associated with the largest eigenvalue of the matrix
    '''
    from scipy import linalg

    n=len(CIJ)	
    vals,vecs=linalg.eig(CIJ)
    i=np.argmax(vals)
    return np.abs(vecs[:,i])

def erange(CIJ):
    '''
    Shortcuts are central edges which significantly reduce the
    characteristic path length in the network.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    Erange : NxN np.ndarray
        range for each edge, i.e. the length of the shortest path from i to j
        for edge c(i,j) after the edge has been removed from the graph
    eta : float
        average range for the entire graph
    Eshort : NxN np.ndarray
        entries are ones for shortcut edges
    fs : float
        fractions of shortcuts in the graph

    Follows the treatment of 'shortcuts' by Duncan Watts
    '''
    N=len(CIJ)	
    K=np.size(np.where(CIJ)[1])
    Erange=np.zeros((N,N))
    i,j=np.where(CIJ)

    for c in range(len(i)):
        CIJcut=CIJ.copy()
        CIJcut[i[c],j[c]]=0
        R,D=reachdist(CIJcut)
        Erange[i[c],j[c]] = D[i[c],j[c]]

    #average range (ignore Inf)
    eta=(np.sum(Erange[np.logical_and(Erange>0,Erange<np.inf)])/
        len(Erange[np.logical_and(Erange>0,Erange<np.inf)]))

    # Original entries of D are ones, thus entries of Erange 
    # must be two or greater.
    # If Erange(i,j) > 2, then the edge is a shortcut.
    # 'fshort' is the fraction of shortcuts over the entire graph.

    Eshort=Erange>2
    fs=len(np.where(Eshort))/K

    return Erange,eta,Eshort,fs

def flow_coef_bd(CIJ):
    '''
    Computes the flow coefficient for each node and averaged over the
    network, as described in Honey et al. (2007) PNAS. The flow coefficient
    is similar to betweenness centrality, but works on a local
    neighborhood. It is mathematically related to the clustering
    coefficient  (cc) at each node as, fc+cc <= 1.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    fc : Nx1 np.ndarray
        flow coefficient for each node
    FC : float
        average flow coefficient over the network
    total_flo : int
        number of paths that "flow" across the central node
    '''
    N=len(CIJ)
    
    fc=np.zeros((N,))
    total_flo=np.zeros((N,))
    max_flo=np.zeros((N,))

    #loop over nodes
    for v in range(N):
        #find neighbors - note: both incoming and outgoing connections
        nb,=np.where(CIJ[v,:]+CIJ[:,v].T)
        fc[v]=0
        if np.where(nb)[0].size:
            CIJflo=-CIJ[np.ix_(nb,nb)]
            for i in range(len(nb)):
                for j in range(len(nb)):
                    if CIJ[nb[i],v] and CIJ[v,nb[j]]:
                        CIJflo[i,j]+=1
            total_flo[v]=np.sum((CIJflo==1)*np.logical_not(np.eye(len(nb))))
            max_flo[v]=len(nb)*len(nb)-len(nb)
            fc[v]=total_flo[v]/max_flo[v]
    
    fc[np.isnan(fc)]=0
    FC=np.mean(fc)

    return fc,FC,total_flo

def kcoreness_centrality_bd(CIJ):
    '''
    The k-core is the largest subgraph comprising nodes of degree at least
    k. The coreness of a node is k if the node belongs to the k-core but
    not to the (k+1)-core. This function computes k-coreness of all nodes
    for a given binary directed connection matrix.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    coreness : Nx1 np.ndarray
        node coreness
    kn : int
        size of k-core
    '''
    N=len(CIJ)
    
    coreness=np.zeros((N,))
    kn=np.zeros((N,))
    
    for k in range(N):
        CIJkcore,kn[k]=kcore_bd(CIJ,k)
        ss=np.sum(CIJkcore,axis=0)>0
        coreness[ss]=k

    return coreness,kn

def	kcoreness_centrality_bu(CIJ):
    '''
    The k-core is the largest subgraph comprising nodes of degree at least
    k. The coreness of a node is k if the node belongs to the k-core but
    not to the (k+1)-core. This function computes the coreness of all nodes
    for a given binary undirected connection matrix.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary undirected connection matrix
   
    Returns
    -------
    coreness : Nx1 np.ndarray
        node coreness
    kn : int
        size of k-core
    '''
    N=len(CIJ)

    #determine if the network is undirected -- if not, compute coreness
    #on the corresponding undirected network
    CIJund=CIJ+CIJ.T
    if np.any(CIJund>1):
        CIJ=np.array(CIJund>0,dtype=float)
    
    coreness=np.zeros((N,))
    kn=np.zeros((N,))
    for k in range(N):
        CIJkcore,kn[k]=kcore_bu(CIJ,k)
        ss=np.sum(CIJkcore,axis=0)>0
        coreness[ss]=k

    return coreness,kn

def module_degree_zscore(W,ci,flag=0):
    '''
    The within-module degree z-score is a within-module version of degree
    centrality.

    Parameters
    ----------
    W : NxN np.narray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.array_like
        community affiliation vector
    flag : int
        Graph type. 0: undirected graph (default)
                    1: directed graph in degree
                    2: directed graph out degree
                    3: directed graph in and out degree

    Returns
    -------
    Z : Nx1 np.ndarray
        within-module degree Z-score
    '''
    _,ci=np.unique(ci,return_inverse=True)
    ci+=1

    if flag==2:
        W=W.copy(); W=W.T
    elif flag==3:
        W=W.copy(); W=W+W.T

    n=len(W)
    Z=np.zeros((n,))					#number of vertices
    for i in range(1,int(np.max(ci)+1)):
        Koi=np.sum(W[np.ix_(ci==i,ci==i)],axis=1)
        Z[np.where(ci==i)]=(Koi-np.mean(Koi))/np.std(Koi)

    Z[np.where(np.isnan(Z))]=0
    return Z

def pagerank_centrality(A,d,falff=None):
    '''
    The PageRank centrality is a variant of eigenvector centrality. This
    function computes the PageRank centrality of each vertex in a graph.

    Formally, PageRank is defined as the stationary distribution achieved
    by instantiating a Markov chain on a graph. The PageRank centrality of
    a given vertex, then, is proportional to the number of steps (or amount
    of time) spent at that vertex as a result of such a process. 

    The PageRank index gets modified by the addition of a damping factor,
    d. In terms of a Markov chain, the damping factor specifies the
    fraction of the time that a random walker will transition to one of its
    current state's neighbors. The remaining fraction of the time the
    walker is restarted at a random vertex. A common value for the damping
    factor is d = 0.85.

    Parameters
    ----------
    A : NxN np.narray
        adjacency matrix
    d : float
        damping factor (see description)
    falff : Nx1 np.ndarray | None
        Initial page rank probability, non-negative values. Default value is
        None. If not specified, a naive bayesian prior is used.

    Returns
    -------
    r : Nx1 np.ndarray
        vectors of page rankings
    
    Notes
    -----
    Note: The algorithm will work well for smaller matrices (number of
    nodes around 1000 or less) 
    '''
    from scipy import linalg

    N=len(A)
    if falff is None:
        norm_falff=np.ones((N,))/N
    else:
        norm_falff=falff/np.sum(falff)

    deg=np.sum(A,axis=0)
    deg[deg==0]=1
    D1=np.diag(1/deg)
    B=np.eye(N)-d*np.dot(A,D1)
    b=(1-d)*norm_falff
    r=linalg.solve(B,b)
    r/=np.sum(r)
    return r

def participation_coef(W,ci,degree='undirected'):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.

    Parameters
    ----------
    W : NxN np.ndarray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector
    degree : str
        Flag to describe nature of graph 'undirected': For undirected graphs
                                         'in': Uses the in-degree
                                         'out': Uses the out-degree

    Returns
    -------
    P : Nx1 np.ndarray
        participation coefficient
    '''
    if degree=='in':
        W = W.T

    _,ci=np.unique(ci,return_inverse=True)
    ci+=1

    n=len(W)						#number of vertices
    Ko=np.sum(W,axis=1)				#(out) degree
    Gc=np.dot((W!=0),np.diag(ci))	#neighbor community affiliation
    Kc2=np.zeros((n,))				#community-specific neighbors

    for i in range(1,int(np.max(ci))+1):
        Kc2+=np.square(np.sum(W*(Gc==i),axis=1))

    P=np.ones((n,))-Kc2/np.square(Ko)
    P[np.where(np.logical_not(Ko))]=0 #P=0 if for nodes with no (out) neighbors

    return P

def participation_coef_sign(W,ci):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected connection matrix with positive and negative weights
    ci : Nx1 np.ndarray
        community affiliation vector

    Returns
    -------
    Ppos : Nx1 np.ndarray
        participation coefficient from positive weights
    Pneg : Nx1 np.ndarray
        participation coefficient from negative weights
    '''
    _,ci=np.unique(ci,return_inverse=True)
    ci+=1

    n=len(W)						#number of vertices

    def pcoef(W_):
        S=np.sum(W_,axis=1)			#strength
        Gc=np.dot(np.logical_not(W_==0),np.diag(ci)) #neighbor community affil.
        Sc2=np.zeros((n,))

        for i in range(1,int(np.max(ci)+1)):
            Sc2+=np.square(np.sum(W_*(Gc==i),axis=1))

        P=np.ones((n,))-Sc2/np.square(S)
        P[np.where(np.isnan(P))]=0
        P[np.where(np.logical_not(P))]=0	#p_ind=0 if no (out)neighbors
        return P

    Ppos=pcoef(W*(W>0))
    Pneg=pcoef(-W*(W<0))

    return Ppos, Pneg

def subgraph_centrality(CIJ):
    '''
    The subgraph centrality of a node is a weighted sum of closed walks of
    different lengths in the network starting and ending at the node. This
    function returns a vector of subgraph centralities for each node of the
    network.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary adjacency matrix

    Cs : Nx1 np.ndarray
        subgraph centrality
    '''
    from scipy import linalg
    
    vals,vecs=linalg.eig(CIJ)				#compute eigendecomposition
    #lambdas=np.diag(vals)
    Cs=np.real(np.dot(vecs*vecs, np.exp( vals ))) #compute eigenvector centr.
    return Cs							#imaginary part from precision error

##############################################################################
# CLUSTERING
##############################################################################

def agreement(ci,buffsz=None):
    '''
    Takes as input a set of vertex partitions CI of
    dimensions [vertex x partition]. Each column in CI contains the
    assignments of each vertex to a class/community/module. This function
    aggregates the partitions in CI into a square [vertex x vertex]
    agreement matrix D, whose elements indicate the number of times any two
    vertices were assigned to the same class.
 
    In the case that the number of nodes and partitions in CI is large
    (greater than ~1000 nodes or greater than ~1000 partitions), the script
    can be made faster by computing D in pieces. The optional input BUFFSZ
    determines the size of each piece. Trial and error has found that
    BUFFSZ ~ 150 works well.

    Parameters
    ----------
    ci : MxN np.ndarray
        set of M (possibly degenerate) partitions of N nodes
    buffsz : int | None
        sets buffer size. If not specified, defaults to 1000
 
    Returns
    -------
    D : NxN np.ndarray
        agreement matrix
    '''
    ci = np.array(ci)
    m,n=ci.shape

    if buffsz is None: buffsz=1000	

    if m<=buffsz:
        ind=dummyvar(ci)
        D=np.dot(ind,ind.T)
    else:
        a=np.arange(0,m,buffsz)
        b=np.arange(buffsz,m,buffsz)
        if len(a)!=len(b):
            b=np.append(b,m)
        D=np.zeros((n,))
        for i,j in zip(a,b):
            y=ci[:,i:j+1]
            ind=dummyvar(y)
            D+=np.dot(ind,ind.T)

    np.fill_diagonal(D,0)
    return D

def agreement_weighted(ci,wts):
    '''
    D = AGREEMENT_WEIGHTED(CI,WTS) is identical to AGREEMENT, with the 
    exception that each partitions contribution is weighted according to 
    the corresponding scalar value stored in the vector WTS. As an example,
    suppose CI contained partitions obtained using some heuristic for 
    maximizing modularity. A possible choice for WTS might be the Q metric
    (Newman's modularity score). Such a choice would add more weight to 
    higher modularity partitions.
 
    NOTE: Unlike AGREEMENT, this script does not have the input argument
    BUFFSZ.
 
    Parameters
    ----------
    ci : MxN np.ndarray
        set of M (possibly degenerate) partitions of N nodes
    wts : Mx1 np.ndarray
        relative weight of each partition
 
    Returns
    -------
    D : NxN np.ndarray
        weighted agreement matrix
    '''
    ci = np.array(ci)
    m,n=ci.shape
    wts=np.array(wts)/np.sum(wts)

    D=np.zeros((n,n))
    for i in range(m):
        d=dummyvar(ci[i,:].reshape(1,n))
        D+=np.dot(d,d.T)*wts[i]
    return D

def clustering_coef_bd(A):
    '''
    The clustering coefficient is the fraction of triangles around a node
    (equiv. the fraction of nodes neighbors that are neighbors of each other).

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector

    Notes
    -----
    Methodological note: In directed graphs, 3 nodes generate up to 8 
    triangles (2*2*2 edges). The number of existing triangles is the main 
    diagonal of S^3/2. The number of all (in or out) neighbour pairs is 
    K(K-1)/2. Each neighbour pair may generate two triangles. "False pairs" 
    are i<->j edge pairs (these do not generate triangles). The number of 
    false pairs is the main diagonal of A^2.
    Thus the maximum possible number of triangles = 
           = (2 edges)*([ALL PAIRS] - [FALSE PAIRS])
           = 2 * (K(K-1)/2 - diag(A^2))
           = K(K-1) - 2(diag(A^2))
    '''
    S=A+A.T								#symmetrized input graph
    K=np.sum(S,axis=1)					#total degree (in+out)
    cyc3=np.diag(np.dot(S,np.dot(S,S)))/2 #number of 3-cycles
    K[np.where(cyc3==0)]=np.inf			#if no 3-cycles exist, make C=0
    CYC3=K*(K-1)-2*np.diag(np.dot(A,A))	#number of all possible 3 cycles
    C=cyc3/CYC3
    return C

def clustering_coef_bu(G):
    '''
    The clustering coefficient is the fraction of triangles around a node
    (equiv. the fraction of nodes neighbors that are neighbors of each other).

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected connection matrix
    
    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector
    '''
    n=len(G)
    C=np.zeros((n,))

    for u in range(n):
        V,=np.where(G[u,:])
        k=len(V)
        if k>=2:						#degree must be at least 2
            S=G[np.ix_(V,V)]
            C[u]=np.sum(S)/(k*k-k)

    return C

def clustering_coef_wd(W):
    '''
    The weighted clustering coefficient is the average "intensity" of 
    triangles around a node.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted directed connection matrix

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector

    Notes
    -----
    Methodological note (also see clustering_coef_bd)
    The weighted modification is as follows:
    - The numerator: adjacency matrix is replaced with weights matrix ^ 1/3
    - The denominator: no changes from the binary version

    The above reduces to symmetric and/or binary versions of the clustering 
    coefficient for respective graphs.
    '''
    A=np.logical_not(W==0).astype(float)	#adjacency matrix
    S=_cuberoot(W)+_cuberoot(W.T)			#symmetrized weights matrix ^1/3
    K=np.sum(A+A.T,axis=1)					#total degree (in+out)
    cyc3=np.diag(np.dot(S,np.dot(S,S)))/2	#number of 3-cycles
    K[np.where(cyc3==0)]=np.inf				#if no 3-cycles exist, make C=0
    CYC3=K*(K-1)-2*np.diag(np.dot(A,A))		#number of all possible 3 cycles
    C=cyc3/CYC3								#clustering coefficient
    return C	

def clustering_coef_wu(W):
    '''
    The weighted clustering coefficient is the average "intensity" of 
    triangles around a node.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted undirected connection matrix

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector
    '''
    K=np.array(np.sum(np.logical_not(W==0),axis=1),dtype=float)
    ws=_cuberoot(W)
    cyc3=np.diag(np.dot(ws,np.dot(ws,ws)))
    K[np.where(cyc3==0)]=np.inf					#if no 3-cycles exist, set C=0
    C=cyc3/(K*(K-1))
    return C

def consensus_und(D,tau,reps=1000):
    '''
    This algorithm seeks a consensus partition of the 
    agreement matrix D. The algorithm used here is almost identical to the
    one introduced in Lancichinetti & Fortunato (2012): The agreement
    matrix D is thresholded at a level TAU to remove an weak elements. The
    resulting matrix is then partitions REPS number of times using the
    Louvain algorithm (in principle, any clustering algorithm that can
    handle weighted matrixes is a suitable alternative to the Louvain
    algorithm and can be substituted in its place). This clustering
    produces a set of partitions from which a new agreement is built. If
    the partitions have not converged to a single representative partition,
    the above process repeats itself, starting with the newly built
    agreement matrix.

    NOTE: In this implementation, the elements of the agreement matrix must
    be converted into probabilities.

    NOTE: This implementation is slightly different from the original
    algorithm proposed by Lanchichinetti & Fortunato. In its original
    version, if the thresholding produces singleton communities, those
    nodes are reconnected to the network. Here, we leave any singleton
    communities disconnected.

    Parameters
    ----------
    D : NxN np.ndarray
        agreement matrix with entries between 0 and 1 denoting the probability
        of finding node i in the same cluster as node j
    tau : float
        threshold which controls the resolution of the reclustering
    reps : int
        number of times the clustering algorithm is reapplied. default value
        is 1000.

    Returns
    -------
    ciu : Nx1 np.ndarray
        consensus partition
    '''
    def unique_partitions(cis):
        #relabels the partitions to recognize different numbers on same topology

        n,r = np.shape(cis)			#ci represents one vector for each rep
        ci_tmp = np.zeros(n)

        for i in range(r):
            for j,u in enumerate(sorted(
                    np.unique(cis[:,i],return_index=True)[1])):
                ci_tmp[np.where(cis[:,i]==cis[u,i])]=j
            cis[:,i]=ci_tmp
            #so far no partitions have been deleted from ci

        #now squash any of the partitions that are completely identical
        #do not delete them from ci which needs to stay same size, so make copy
        ciu = []
        cis = cis.copy()
        c = np.arange(r)
        #count=0
        while (c!=0).sum() > 0:
            ciu.append(cis[:,0])
            dup = np.where(np.sum(np.abs(cis.T-cis[:,0]),axis=1)==0)
            cis = np.delete(cis,dup,axis=1)
            c = np.delete(c,dup)
            #count+=1
            #print count,c,dup
            #if count>10:
            #	class QualitativeError(): pass
            #	raise QualitativeError()
        return np.transpose(ciu)

    n=len(D)
    flag=True
    while flag:
        flag=False
        dt = D*(D >= tau)
        np.fill_diagonal(dt,0)

        if np.size(np.where(dt==0)) == 0:
            ciu = np.arange(1,n+1)
        else:
            cis = np.zeros((n,reps))
            for i in np.arange(reps):
                cis[:,i],_ = modularity_louvain_und_sign(dt)		
            ciu = unique_partitions(cis)
            nu = np.size(ciu, axis=1)
            if nu > 1:
                flag = True
                D = agreement(cis)/reps

    return np.squeeze(ciu+1)

def get_components(A, no_depend=False):
    '''	
    Returns the components of an undirected graph specified by the binary and 
    undirected adjacency matrix adj. Components and their constitutent nodes 
    are assigned the same index and stored in the vector, comps. The vector, 
    comp_sizes, contains the number of nodes beloning to each component.

    Parameters
    ----------
    adj : NxN np.ndarray
        binary undirected adjacency matrix
    no_depend : bool
        If true, doesn't import networkx to do the calculation. Default value
        is false.

    Returns
    -------
    comps : Nx1 np.ndarray
        vector of component assignments for each node
    comp_sizes : Mx1 np.ndarray
        vector of component sizes

    Notes
    -----
    Note: disconnected nodes will appear as components with a component
    size of 1

    Note: The identity of each component (i.e. its numerical value in the 
    result) is not guaranteed to be identical the value returned in BCT, 
    although the component topology is.

    Note: networkx is used to do the computation efficiently. If networkx is 
    not available a breadth-first search that does not depend on networkx is 
    used instead, but this is less efficient. The corresponding BCT function 
    does the computation by computing the Dulmage-Mendelsohn decomposition. I 
    don't know what a Dulmage-Mendelsohn decomposition is and there doesn't 
    appear to be a python equivalent. If you think of a way to implement this 
    better, let me know.
        '''
        #nonsquare matrices cannot be symmetric; no need to check

    if not np.all(A==A.T):			#ensure matrix is undirected
        raise BCTParamError('get_components can only be computed for undirected'
            ' matrices.  If your matrix is noisy, correct it with np.around')

    A=binarize(A,copy=True)
    n=len(A)
    np.fill_diagonal(A, 1)

    try:
        if no_depend:
            raise ImportError()
        else:
            import networkx as nx
        net=nx.from_numpy_matrix(A)
        cpts=list(nx.connected_components(net))
        
        cptvec=np.zeros((n,))
        cptsizes=np.zeros(len(cpts))
        for i,cpt in enumerate(cpts):
            cptsizes[i]=len(cpt)
            for node in cpt:
                cptvec[node]=i+1
    
    except ImportError:
        #if networkx is not available use less efficient breadth first search
        cptvec=np.zeros((n,))
        r,_ = breadthdist(A)
        for node,reach in enumerate(r):
            if cptvec[node] > 0:
                continue
            else:
                cptvec[np.where(reach)] = np.max(cptvec)+1

        cptsizes=np.zeros(np.max(cptvec))
        for i in np.arange(np.max(cptvec)):
            cptsizes[i]=np.size(np.where(cptvec==i+1))

    return cptvec,cptsizes

def number_of_components(A):
    _,csizes = get_components(A)
    return len(csizes)
    
def transitivity_bd(A):
    '''
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    T : float
        transitivity scalar

    Notes
    -----
    Methodological note: In directed graphs, 3 nodes generate up to 8 
    triangles (2*2*2 edges). The number of existing triangles is the main 
    
    diagonal of S^3/2. The number of all (in or out) neighbour pairs is 
    K(K-1)/2. Each neighbour pair may generate two triangles. "False pairs"
    are i<->j edge pairs (these do not generate triangles). The number of 
    false pairs is the main diagonal of A^2. Thus the maximum possible 
    number of triangles = (2 edges)*([ALL PAIRS] - [FALSE PAIRS])
                        = 2 * (K(K-1)/2 - diag(A^2))
                        = K(K-1) - 2(diag(A^2))
    '''
    S=A+A.T									#symmetrized input graph
    K=np.sum(S,axis=1)						#total degree (in+out)
    cyc3=np.diag(np.dot(S,np.dot(S,S)))/2	#number of 3-cycles
    K[np.where(cyc3==0)]=np.inf				#if no 3-cycles exist, make C=0
    CYC3=K*(K-1)-2*np.diag(np.dot(A,A))		#number of all possible 3-cycles
    return np.sum(cyc3)/np.sum(CYC3)

def transitivity_bu(A):
    '''
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected connection matrix
    
    Returns
    -------
    T : float
        transitivity scalar
    '''
    tri3=np.trace(np.dot(A,np.dot(A,A)))
    tri2=np.sum(np.dot(A,A))-np.trace(np.dot(A,A))
    return tri3/tri2

def transitivity_wd(W):
    '''
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).

    Parameters
    ----------
    W : NxN np.ndarray
        weighted directed connection matrix

    Returns
    -------
    T : int
        transitivity scalar

    Methodological note (also see note for clustering_coef_bd)
    The weighted modification is as follows:
    - The numerator: adjacency matrix is replaced with weights matrix ^ 1/3
    - The denominator: no changes from the binary version

    The above reduces to symmetric and/or binary versions of the clustering
    coefficient for respective graphs.
    '''
    A=np.logical_not(W==0).astype(float)	#adjacency matrix
    S=_cuberoot(W)+_cuberoot(W.T)			#symmetrized weights matrix ^1/3
    K=np.sum(A+A.T,axis=1)			 		#total degree (in+out)
    cyc3=np.diag(np.dot(S,np.dot(S,S)))/2	#number of 3-cycles
    K[np.where(cyc3==0)]=np.inf				#if no 3-cycles exist, make T=0
    CYC3=K*(K-1)-2*np.diag(np.dot(A,A))		#number of all possible 3-cycles
    return np.sum(cyc3)/np.sum(CYC3)		#transitivity

def transitivity_wu(W):
    '''	
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).

    Parameters
    ----------
    W : NxN np.ndarray
        weighted undirected connection matrix

    Returns
    -------
    T : int
        transitivity scalar
    '''
    K=np.sum(np.logical_not(W==0),axis=1)
    ws=_cuberoot(W)
    cyc3=np.diag(np.dot(ws,np.dot(ws,ws)))
    return np.sum(cyc3,axis=0)/np.sum(K*(K-1),axis=0)	

##############################################################################
# CORE
##############################################################################

def assortativity_bin(CIJ,flag):
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
    if flag==0:								#undirected version
        deg=degrees_und(CIJ)
        i,j=np.where(np.triu(CIJ,1)>0)
        K=len(i)
        degi=deg[i]; degj=deg[j]	
    else:									#directed version
        id,od,deg=degrees_dir(CIJ)
        i,j=np.where(CIJ>0)
        K=len(i)

        if flag==1: degi=od[i]; degj=id[j]
        elif flag==2: degi=id[i]; degj=od[j]
        elif flag==3: degi=od[i]; degj=od[j]
        elif flag==4: degi=id[i]; degj=id[j]
        else: raise ValueError('Flag must be 0-4')

    #compute assortativity
    term1 = np.sum(degi*degj)/K
    term2 = np.square(np.sum(.5*(degi+degj))/K)
    term3 = np.sum(.5*(degi*degi+degj*degj))/K
    r = (term1 - term2) / (term3 - term2)
    return r

def assortativity_wei(CIJ,flag):
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
    if flag==0:							#undirected version
        str=strengths_und(CIJ)
        i,j=np.where(np.triu(CIJ,1)>0)
        K=len(i)
        stri=str[i]
        strj=str[j]
    else:
        ist,ost=strengths_dir(CIJ)		#directed version
        i,j=np.where(CIJ>0)
        K=len(i)
        
        if flag==1: stri=ost[i]; strj=ist[j]
        elif flag==2: stri=ist[i]; strj=ost[j]
        elif flag==3: stri=ost[i]; strj=ost[j]
        elif flag==4: stri=ist[i]; strj=ost[j]
        else: raise ValueError('Flag must be 0-4')

    #compute assortativity
    term1 = np.sum(stri*strj)/K
    term2 = np.square(np.sum(.5*(stri+strj))/K)
    term3 = np.sum(.5*(stri*stri+strj*strj))/K
    r = (term1 - term2) / (term3 - term2)
    return r

def kcore_bd(CIJ,k,peel=False):
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
    if peel: peelorder,peellevel=([],[])
    iter=0
    CIJkcore=CIJ.copy()

    while True:
        id,od,deg=degrees_dir(CIJkcore)		#get degrees of matrix
    
        #find nodes with degree <k	
        ff,=np.where(np.logical_and(deg<k,deg>0))

        if ff.size==0: break				#if none found -> stop

        #else peel away found nodes
        iter+=1
        CIJkcore[ff,:]=0
        CIJkcore[:,ff]=0

        if peel: peelorder.append(ff)
        if peel: peellevel.append(iter*np.ones((len(ff),)))
    
    kn=np.sum(deg>0)

    if peel: return CIJkcore,kn,peelorder,peellevel
    else: return CIJkcore,kn

def kcore_bu(CIJ,k,peel=False):
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
    if peel: peelorder,peellevel=([],[])
    iter=0
    CIJkcore=CIJ.copy()

    while True:
        deg=degrees_und(CIJkcore)		#get degrees of matrix
        
        #find nodes with degree <k
        ff,=np.where(np.logical_and(deg<k,deg>0))

        if ff.size==0: break			#if none found -> stop
            
        #else peel away found nodes
        iter+=1
        CIJkcore[ff,:]=0
        CIJkcore[:,ff]=0

        if peel: peelorder.append(ff)
        if peel: peellevel.append(iter*np.ones((len(ff),)))

    kn=np.sum(deg>0)

    if peel: return CIJkcore,kn,peelorder,peellevel
    else: return CIJkcore,kn

def rich_club_bd(CIJ,klevel=None):
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
    #definition of degree as used for RC coefficients
    #degree is taken to be the sum of incoming and outgoing connections
    id,od,deg=degrees_dir(CIJ)

    if klevel is None:
        klevel=int(np.max(deg))
    
    R=np.zeros((klevel,))
    Nk=np.zeros((klevel,))
    Ek=np.zeros((klevel,))
    for k in range(klevel):
        SmallNodes,=np.where(deg<=k+1)		#get small nodes with degree <=k
        subCIJ=np.delete(CIJ,SmallNodes,axis=0)
        subCIJ=np.delete(subCIJ,SmallNodes,axis=1)
        Nk[k]=np.size(subCIJ,axis=1)		#number of nodes with degree >k
        Ek[k]=np.sum(subCIJ)				#number of connections in subgraph
        R[k]=Ek[k]/(Nk[k]*(Nk[k]-1))		#unweighted rich club coefficient

    return R,Nk,Ek

def rich_club_bu(CIJ,klevel=None):
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
    deg=degrees_und(CIJ)					#compute degree of each node

    if klevel==None:
        klevel=int(np.max(deg))
    
    R=np.zeros((klevel,))
    Nk=np.zeros((klevel,))
    Ek=np.zeros((klevel,))
    for k in range(klevel):
        SmallNodes,=np.where(deg<=k+1)		#get small nodes with degree <=k
        subCIJ=np.delete(CIJ,SmallNodes,axis=0)
        subCIJ=np.delete(subCIJ,SmallNodes,axis=1)
        Nk[k]=np.size(subCIJ,axis=1)		#number of nodes with degree >k
        Ek[k]=np.sum(subCIJ)				#number of connections in subgraph
        R[k]=Ek[k]/(Nk[k]*(Nk[k]-1))		#unweighted rich club coefficient

    return R,Nk,Ek
    
def rich_club_wd(CIJ,klevel=None):
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
    #degree of each node is defined here as in+out
    deg = np.sum((CIJ != 0),axis=0)+np.sum((CIJ.T != 0),axis=0)

    if klevel is None: klevel = np.max(deg)
    Rw=np.zeros((klevel,))

    #sort the weights of the network, with the strongest connection first
    wrank = np.sort(CIJ.flat)[::-1]

    for k in range(klevel):
        SmallNodes,=np.where(deg<k+1)
        if np.size(SmallNodes) == 0:
            Rw[k]=np.nan
            continue

        #remove small nodes with node degree < k
        cutCIJ=np.delete(np.delete(CIJ,SmallNodes,axis=0),SmallNodes,axis=1)
        #total weight of connections in subset E>r
        Wr = np.sum(cutCIJ)
        #total number of connections in subset E>r
        Er = np.size(np.where(cutCIJ.flat != 0),axis=1)
        #E>r number of connections with max weight in network
        wrank_r = wrank[:Er]
        #weighted rich-club coefficient
        Rw[k] = Wr/np.sum(wrank_r)
    return Rw

def rich_club_wu(CIJ,klevel=None):
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
    deg = np.sum((CIJ != 0),axis=0)

    if klevel is None: klevel = np.max(deg)
    Rw=np.zeros((klevel,))

    #sort the weights of the network, with the strongest connection first
    wrank = np.sort(CIJ.flat)[::-1]

    for k in range(klevel):
        SmallNodes,=np.where(deg<k+1)
        if np.size(SmallNodes) == 0:
            Rw[k]=np.nan
            continue

        #remove small nodes with node degree < k
        cutCIJ=np.delete(np.delete(CIJ,SmallNodes,axis=0),SmallNodes,axis=1)
        #total weight of connections in subset E>r
        Wr = np.sum(cutCIJ)
        #total number of connections in subset E>r
        Er = np.size(np.where(cutCIJ.flat != 0),axis=1)
        #E>r number of connections with max weight in network
        wrank_r = wrank[:Er]
        #weighted rich-club coefficient
        Rw[k] = Wr/np.sum(wrank_r)
    return Rw
    
def score_wu(CIJ,s):
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
    CIJscore=CIJ.copy()
    while True:
        str=strengths_und(CIJscore)			#get strengths of matrix

        #find nodes with strength <s
        ff,=np.where(np.logical_and(str<s,str>0))

        if ff.size==0: break			#if none found -> stop

        #else peel away found nodes
        CIJscore[ff,:]=0
        CIJscore[:,ff]=0

    sn=np.sum(str>0)
    return CIJscore,sn

##############################################################################
# DEGREE
##############################################################################

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
    CIJ=binarize(CIJ,copy=True)		#ensure CIJ is binary
    id=np.sum(CIJ,axis=0)			#indegree = column sum of CIJ
    od=np.sum(CIJ,axis=1)			#outdegree = row sum of CIJ
    deg=id+od						#degree = indegree+outdegree
    return id,od,deg

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
    CIJ=binarize(CIJ,copy=True)		#ensure CIJ is binary
    return np.sum(CIJ,axis=0)

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
    CIJ=binarize(CIJ,copy=True)		#ensure CIJ is binary
    n=len(CIJ)
    id=np.sum(CIJ,axis=0)			#indegree = column sum of CIJ
    od=np.sum(CIJ,axis=1)			#outdegree = row sum of CIJ

    #create the joint degree distribution matrix
    #note: the matrix is shifted by one, to accomodate zero id and od in the 
    #first row/column
    #upper triangular part of the matrix has vertices with od>id
    #lower triangular part has vertices with id>od
    #main diagonal has units with id=od

    szJ=np.max((id,od))+1
    J=np.zeros((szJ,szJ))
    
    for i in range(n):
        J[id[i],od[i]]+=1

    J_od=np.sum(np.triu(J,1))
    J_id=np.sum(np.tril(J,-1))
    J_bl=np.sum(np.diag(J))
    return J,J_od,J_id,J_bl
    

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
    istr=np.sum(CIJ,axis=0)
    ostr=np.sum(CIJ,axis=1)
    return istr+ostr

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
    return np.sum(CIJ,axis=0)	

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
    W=W.copy()
    n=len(W)
    np.fill_diagonal(W, 0)				#clear diagonal
    Spos=W*(W>0)						#positive strengths
    Sneg=W*(W<0)						#negative strengths

    vpos=np.sum(W[W>0])					#positive weight
    vneg=np.sum(W[W<0])					#negative weight
    return Spos, Sneg, vpos, vneg

##############################################################################
# DISTANCE
##############################################################################

def breadthdist(CIJ):
    '''
    The binary reachability matrix describes reachability between all pairs
    of nodes. An entry (u,v)=1 means that there exists a path from node u
    to node v; alternatively (u,v)=0.

    The distance matrix contains lengths of shortest paths between all
    pairs of nodes. An entry (u,v) represents the length of shortest path 
    from node u to  node v. The average shortest path length is the 
    characteristic path length of the network.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed/undirected connection matrix

    Returns
    -------
    R : NxN np.ndarray
        binary reachability matrix
    D : NxN np.ndarray 
        distance matrix

    Notes
    -----
    slower but less memory intensive than "reachdist.m".
    '''
    n=len(CIJ)

    D=np.zeros((n,n))
    for i in range(n):
        D[i,:],_=breadth(CIJ,i)

    D[D==0]=np.inf;
    R=(D!=np.inf);
    return R,D

def breadth(CIJ,source):
    '''
    Implementation of breadth-first search.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed/undirected connection matrix
    source : int
        source vertex

    Returns
    -------
    distance : Nx1 np.ndarray
        vector of distances between source and ith vertex (0 for source)
    branch : Nx1 np.ndarray
        vertex that precedes i in the breadth-first search (-1 for source)

    Notes
    -----
    Breadth-first search tree does not contain all paths (or all 
    shortest paths), but allows the determination of at least one path with
    minimum distance. The entire graph is explored, starting from source 
    vertex 'source'.
    '''
    n=len(CIJ)

    #colors: white,gray,black
    white=0; gray=1; black=2

    color=np.zeros((n,))
    distance=np.inf*np.ones((n,))
    branch=np.zeros((n,))

    #start on vertex source
    color[source]=gray
    distance[source]=0
    branch[source]=-1
    Q=[source]

    #keep going until the entire graph is explored
    while Q:
        u=Q[0]
        ns,=np.where(CIJ[u,:])
        for v in ns:
            #this allows the source distance itself to be recorded
            if distance[v]==0:
                distance[v]=distance[u]+1
            if color[v]==white:
                color[v]=gray
                distance[v]=distance[u]+1
                branch[v]=u
                Q.append(v)
        Q=Q[1:]
        color[u]=black

    return distance,branch

def charpath(D, include_diagonal=False):
    '''
    The characteristic path length is the average shortest path length in 
    the network. The global efficiency is the average inverse shortest path
    length in the network.

    Parameters
    ----------
    D : NxN np.ndarray
        distance matrix
    include_diagonal : bool
        If True, include the weights on the diagonal. Default value is False.

    Returns
    -------
    lambda : float
        characteristic path length
    efficiency : float
        global efficiency
    ecc : Nx1 np.ndarray
        eccentricity at each vertex
    radius : float
        radius of graph
    diameter : float
        diameter of graph

    Notes
    -----
    The input distance matrix may be obtained with any of the distance
    functions, e.g. distance_bin, distance_wei.
    Characteristic path length is calculated as the global mean of 
    the distance matrix D, excludings any 'Infs' but including distances on
    the main diagonal.
    '''
    D = D.copy()

    if not include_diagonal:
        np.fill_diagonal(D, 0)

    #mean of finite entries of D[G]
    lambda_=np.sum(D[D!=np.inf])/len(np.where(D!=np.inf)[0])	

    #eccentricity for each vertex (ignore inf)
    ecc=np.max(D*(D!=np.inf),axis=1)

    #radius of graph
    radius=np.min(ecc)	#but what about zeros?

    #diameter of graph
    diameter=np.max(ecc)

    #efficiency: mean of inverse entries of D[G]
    n=len(D)
    D=1/D								#invert distance
    np.fill_diagonal(D, 0)				#set diagonal to 0
    efficiency=np.sum(D)/(n*(n-1))		#compute global efficiency

    return lambda_,efficiency,ecc,radius,diameter

def cycprob(Pq):
    '''
    Cycles are paths which begin and end at the same node. Cycle 
    probability for path length d, is the fraction of all paths of length 
    d-1 that may be extended to form cycles of length d.

    Parameters
    ----------
    Pq : NxNxQ np.ndarray
        Path matrix with Pq[i,j,q] = number of paths from i to j of length q.
        Produced by findpaths()

    Returns
    -------
    fcyc : Qx1 np.ndarray
        fraction of all paths that are cycles for each path length q
    pcyc : Qx1 np.ndarray
        probability that a non-cyclic path of length q-1 can be extended to
        form a cycle of length q for each path length q
    '''

    #note: fcyc[1] must be zero, as there cannot be cycles of length 1
    fcyc=np.zeros(np.size(Pq,axis=2))
    for q in range(np.size(Pq,axis=2)):
        if np.sum(Pq[:,:,q])>0:
            fcyc[q]=np.sum(np.diag(Pq[:,:,q]))/np.sum(Pq[:,:,q])
        else:
            fcyc[q]=0

    #note: pcyc[1] is not defined (set to zero)
    #note: pcyc[2] is equal to the fraction of reciprocal connections
    #note: there are no non-cyclic paths of length N and no cycles of len N+1
    pcyc=np.zeros(np.size(Pq,axis=2))
    for q in range(np.size(Pq,axis=2)):
        if np.sum(Pq[:,:,q-1])-np.sum(np.diag(Pq[:,:,q-1]))>0:
            pcyc[q]=(np.sum(np.diag(Pq[:,:,q-1]))/
                np.sum(Pq[:,:,q-1])-np.sum(np.diag(Pq[:,:,q-1])))
        else:
            pcyc[q]=0
    
    return fcyc,pcyc

def distance_bin(G):
    '''	
    The distance matrix contains lengths of shortest paths between all
    pairs of nodes. An entry (u,v) represents the length of shortest path 
    from node u to node v. The average shortest path length is the 
    characteristic path length of the network.

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed/undirected connection matrix
    
    Returns
    -------
    D : NxN 
        distance matrix

    Notes
    -----
    Lengths between disconnected nodes are set to Inf.
    Lengths on the main diagonal are set to 0.
    Algorithm: Algebraic shortest paths.
    '''
    G=binarize(G,copy=True)
    D=np.eye(len(G))
    n=1
    nPATH=G.copy()						#n path matrix
    L=(nPATH!=0)						#shortest n-path matrix
    
    while np.any(L):
        D+=n*L
        n+=1
        nPATH=np.dot(nPATH,G)
        L=(nPATH!=0)*(D==0)

    D[D==0]=np.inf						#disconnected nodes are assigned d=inf
    np.fill_diagonal(D, 0)
    return D

def distance_wei(G):
    '''
    The distance matrix contains lengths of shortest paths between all
    pairs of nodes. An entry (u,v) represents the length of shortest path 
    from node u to node v. The average shortest path length is the 
    characteristic path length of the network.

    Parameters
    ----------
    L : NxN np.ndarray
        Directed/undirected connection-length matrix.
        NB L is not the adjacency matrix. See below.

    Returns
    -------
    D : NxN np.ndarray
        distance (shortest weighted path) matrix
    B : NxN np.ndarray
        matrix of number of edges in shortest weighted path

    Notes
    -----
       The input matrix must be a connection-length matrix, typically
    obtained via a mapping from weight to length. For instance, in a
    weighted correlation network higher correlations are more naturally
    interpreted as shorter distances and the input matrix should
    consequently be some inverse of the connectivity matrix. 
       The number of edges in shortest weighted paths may in general 
    exceed the number of edges in shortest binary paths (i.e. shortest
    paths computed on the binarized connectivity matrix), because shortest 
    weighted paths have the minimal weighted distance, but not necessarily 
    the minimal number of edges.
       Lengths between disconnected nodes are set to Inf.
       Lengths on the main diagonal are set to 0.

    Algorithm: Dijkstra's algorithm.
    '''
    n=len(G)						
    D=np.zeros((n,n))					#distance matrix
    D[np.logical_not(np.eye(n))]=np.inf	
    B=np.zeros((n,n))					#number of edges matrix

    for u in range(n):
        S=np.ones((n,),dtype=bool)		#distance permanence (true is temporary)
        G1=G.copy()
        V=[u]
        while True:
            S[V]=0						#distance u->V is now permanent
            G1[:,V]=0					#no in-edges as already shortest
            for v in V:
                W,=np.where(G1[v,:])	#neighbors of shortest nodes

                td=np.array([D[u,W].flatten(),(D[u,v]+G1[v,W]).flatten()])
                d=np.min(td,axis=0)
                wi=np.argmin(td,axis=0)

                D[u,W]=d				#smallest of old/new path lengths
                ind=W[np.where(wi==1)]	#indices of lengthened paths
                B[u,ind]=B[u,v]+1		#increment nr_edges for lengthened paths

            if D[u,S].size==0:			#all nodes reached
                break 
            minD=np.min(D[u,S])
            if np.isinf(minD):			#some nodes cannot be reached
                break

            V,=np.where(D[u,:]==minD)

    return D,B

def efficiency_bin(G,local=False):
    '''
    The global efficiency is the average of inverse shortest path length, 
    and is inversely related to the characteristic path length.

    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected connection matrix
    local : bool
        If True, computes local efficiency instead of global efficiency.
        Default value = False.

    Returns
    -------
    Eglob : float
        global efficiency, only if local=False
    Eloc : Nx1 np.ndarray
        local efficiency, only if local=True
    '''
    def distance_inv(g):
        D=np.eye(len(g))
        n=1
        nPATH=g.copy()
        L=(nPATH!=0)
        
        while np.any(L):
            D+=n*L
            n+=1
            nPATH=np.dot(nPATH,g)
            L=(nPATH!=0)*(D==0)
        D[np.logical_not(D)]=np.inf
        D=1/D
        np.fill_diagonal(D, 0)
        return D

    n=len(G)							#number of nodes
    if local:							
        E=np.zeros((n,))				#local efficiency	

        for u in range(n):
            #V,=np.where(G[u,:])			#neighbors
            #k=len(V)					#degree
            #if k>=2:					#degree must be at least 2
            #	e=distance_inv(G[V].T[V])
            #	E[u]=np.sum(e)/(k*k-k)	#local efficiency computation

            #find pairs of neighbors 
            V,=np.where(np.logical_or(G[u,:], G[u,:].T))
            #inverse distance matrix
            e=distance_inv(G[np.ix_(V,V)])
            #symmetrized inverse distance matrix
            se=e+e.T

            #symmetrized adjacency vector
            sa=G[u,V]+G[V,u].T
            numer = np.sum(np.dot(sa.T,sa)*se)/2
            if numer!=0:
                denom = np.sum(sa)**2 - np.sum(sa*sa)
                E[u] = numer/denom		#local efficiency

    else:
        e=distance_inv(G)
        E=np.sum(e)/(n*n-n)				#global efficiency
    return E

def efficiency_wei(Gw,local=False):
    '''
    The global efficiency is the average of inverse shortest path length, 
    and is inversely related to the characteristic path length.

    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted connection matrix
        (all weights in W must be between 0 and 1)
    local : bool
        If True, computes local efficiency instead of global efficiency.
        Default value = False.

    Returns
    -------
    Eglob : float
        global efficiency, only if local=False
    Eloc : Nx1 np.ndarray
        local efficiency, only if local=True

    Notes
    -----
       The  efficiency is computed using an auxiliary connection-length
    matrix L, defined as L_ij = 1/W_ij for all nonzero L_ij; This has an
    intuitive interpretation, as higher connection weights intuitively
    correspond to shorter lengths.
       The weighted local efficiency broadly parallels the weighted
    clustering coefficient of Onnela et al. (2005) and distinguishes the
    influence of different paths based on connection weights of the
    corresponding neighbors to the node in question. In other words, a path
    between two neighbors with strong connections to the node in question
    contributes more to the local efficiency than a path between two weakly
    connected neighbors. Note that this weighted variant of the local
    efficiency is hence not a strict generalization of the binary variant.

    Algorithm:  Dijkstra's algorithm
    '''
    def distance_inv_wei(G):
        n=len(G)
        D=np.zeros((n,n))				#distance matrix
        D[np.logical_not(np.eye(n))]=np.inf

        for u in range(n):
            S=np.ones((n,),dtype=bool)	#distance permanence (true is temporary)
            G1=G.copy()
            V=[u]
            while True:
                S[V]=0					#distance u->V is now permanent
                G1[:,V]=0				#no in-edges as already shortest
                for v in V:
                    W,=np.where(G1[v,:])	#neighbors of smallest nodes
                    td=np.array([D[u,W].flatten(),(D[u,v]+G1[v,W]).flatten()])
                    D[u,W]=np.min(td,axis=0)

                if D[u,S].size==0:		#all nodes reached
                    break				
                minD=np.min(D[u,S])
                if np.isinf(minD):		#some nodes cannot be reached
                    break
                V,=np.where(D[u,:]==minD)
        
        np.fill_diagonal(D, 1)
        D=1/D
        np.fill_diagonal(D, 0)
        return D

    n=len(Gw)
    Gl=invert(Gw,copy=True)				#connection length matrix
    A=np.array((Gw!=0),dtype=int)
    if local:
        E=np.zeros((n,))				#local efficiency
        for u in range(n):
            #V,=np.where(Gw[u,:])		#neighbors
            #k=len(V)					#degree
            #if k>=2:					#degree must be at least 2
            #	e=(distance_inv_wei(Gl[V].T[V])*np.outer(Gw[V,u],Gw[u,V]))**1/3
            #	E[u]=np.sum(e)/(k*k-k)
            
            #find pairs of neighbors
            V,=np.where(np.logical_or(Gw[u,:],Gw[:,u].T))
            #symmetrized vector of weights
            sw=_cuberoot(Gw[u,V])+_cuberoot(Gw[V,u].T)
            #inverse distance matrix
            e=distance_inv_wei(Gl[np.ix_(V,V)])
            #symmetrized inverse distance matrix
            se=_cuberoot(e)+_cuberoot(e.T)

            numer=np.sum(np.outer(sw.T,sw)*se)/2
            if numer!=0:
                #symmetrized adjacency vector
                sa=A[u,V]+A[V,u].T
                denom=np.sum(sa)**2 - np.sum(sa*sa)
                #print numer,denom
                E[u] = numer/denom		#local efficiency
                
    else:
        e=distance_inv_wei(Gl)
        E=np.sum(e)/(n*n-n)
    return E

def findpaths(CIJ,qmax,sources,savepths=False):
    '''	
    Paths are sequences of linked nodes, that never visit a single node
    more than once. This function finds all paths that start at a set of 
    source nodes, up to a specified length. Warning: very memory-intensive.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed/undirected connection matrix
    qmax : int
        maximal path length
    sources : Nx1 np.ndarray
        source units from which paths are grown
    savepths : bool
        True if all paths are to be collected and returned. This functionality
        is currently not enabled.

    Returns
    -------
    Pq : NxNxQ np.ndarray
        Path matrix with P[i,j,jq] = number of paths from i to j with length q
    tpath : int
        total number of paths found
    plq : Qx1 np.ndarray
        path length distribution as a function of q
    qstop : int
        path length at which findpaths is stopped
    allpths : None
        a matrix containing all paths up to qmax. This function is extremely
        complicated and reimplementing it in bctpy is not straightforward.
    util : NxQ np.ndarray
        node use index

    Notes
    -----
    Note that Pq(:,:,N) can only carry entries on the diagonal, as all
    "legal" paths of length N-1 must terminate.  Cycles of length N are
    possible, with all vertices visited exactly once (except for source and
    target). 'qmax = N' can wreak havoc (due to memory problems).

    Note: Weights are discarded.
    Note: I am certain that this algorithm is rather inefficient -
    suggestions for improvements are welcome.

    '''
    CIJ=binarize(CIJ,copy=True)				#ensure CIJ is binary
    n=len(CIJ)
    k=np.sum(CIJ)
    pths=[]
    Pq=np.zeros((n,n,qmax))
    util=np.zeros((n,qmax))

    #this code is for pathlength=1
    #paths are seeded from sources
    q=1
    for j in range(n):
        for i in range(len(sources)):
            i_s=sources[i]
            if CIJ[i_s,j]==1:
                pths.append([i_s,j])
    pths=np.array(pths)

    #calculate the use index per vertex (for paths of length 1)
    util[:,q],_=np.histogram(pths,bins=n)
    #now enter the found paths of length 1 into the pathmatrix Pq
    for nrp in range(np.size(pths,axis=0)):
        Pq[pths[nrp,0],pths[nrp,q],q-1]+=1

    #begin saving allpths
    if savepths:
        allpths=pths.copy()
    else:
        allpths=[]

    npthscnt=k

    #big loop for all other pathlengths q
    for q in range(2,qmax+1):
        #to keep track of time...
        print(('current pathlength (q=i, number of paths so far (up to q-1)=i'			%(q,np.sum(Pq))))

        #old paths are now in 'pths'
        #new paths are about to be collected in 'npths'
        #estimate needed allocation for new paths
        len_npths=np.min((np.ceil(1.1*npthscnt*k/n),100000000))
        npths=np.zeros((q+1,len_npths))

        #find the unique set of endpoints of 'pths'
        endp=np.unique(pths[:,q-1])
        npthscnt=0

        for i in endp:	#set of endpoints of previous paths
            #in 'pb' collect all previous paths with 'i' as their endpoint
            pb,=np.where(pths[:,q-1]==i)
            #find the outgoing connections from i (breadth-first)
            nendp,=np.where(CIJ[i,:]==1)
            #if i is not a dead end
            if nendp.size:
                for j in nendp:			#endpoints of next edge
                    #find new paths -- only legal ones, no vertex twice visited
                    pb_temp=pb[np.sum(j==pths[pb,1:q],axis=1)==0]

                    #add new paths to 'npths'
                    pbx=pths[pb_temp-1,:]
                    npx=np.ones((len(pb_temp),1))*j
                    npths[:,npthscnt:npthscnt+len(pb_temp)]=np.append(
                        pbx,npx,axis=1).T
                    npthscnt+=len(pb_temp)
                    #count new paths and add the number to P
                    Pq[:n,j,q-1]+=np.histogram(pths[pb_temp-1,0],bins=n)[0]

        #note: 'npths' now contains a list of all the paths of length q
        if len_npths>npthscnt:
            npths=npths[:,:npthscnt]

        #append the matrix of all paths
        #FIXME
        if savepths:
            raise NotImplementedError("Sorry allpaths is not yet implemented")
        
        #calculate the use index per vertex (correct for cycles, count
        #source/target only once)
        util[:,q-1]+=(np.histogram(npths[:,:npthscnt],bins=n)[0]-
            np.diag(Pq[:,:,q-1]))

        #elininate cycles from "making it" to the next level, so that "pths"
        #contains all the paths that have a chance of being continued
        if npths.size:
            pths=np.squeeze(npths[:,np.where(npths[0,:]!=npths[q,:])]).T
        else:
            pths=[]

        #if there are no 'pths' paths left, end the search
        if not pths.size:
            qstop=q
            tpath=np.sum(Pq)
            plq=np.sum(np.sum(Pq,axis=0),axis=0)
            return
    
    qstop=q
    tpath=np.sum(Pq)						#total number of paths
    plq=np.sum(np.sum(Pq,axis=0),axis=0)	#path length distribution

    return Pq,tpath,plq,qstop,allpths,util

def findwalks(CIJ):
    '''
    Walks are sequences of linked nodes, that may visit a single node more
    than once. This function finds the number of walks of a given length, 
    between any two nodes.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed/undirected connection matrix

    Returns
    -------
    Wq : NxNxQ np.ndarray
        Wq[i,j,q] is the number of walks from i to j of length q
    twalk : int
        total number of walks found
    wlq : Qx1 np.ndarray
        walk length distribution as a function of q

    Notes
    -----
    Wq grows very quickly for larger N,K,q. Weights are discarded.
    '''
    CIJ=binarize(CIJ,copy=True)
    n=len(CIJ)
    Wq=np.zeros((n,n,n))
    CIJpwr=CIJ.copy()
    Wq[:,:,1]=CIJ
    for q in range(n):
        CIJpwr=np.dot(CIJpwr,CIJ)
        Wq[:,:,q]=CIJpwr

    twalk=np.sum(Wq)						#total number of walks
    wlq=np.sum(np.sum(Wq,axis=0),axis=0)
    return Wq,twalk,wlq

def reachdist(CIJ):
    '''
    The binary reachability matrix describes reachability between all pairs

    of nodes. An entry (u,v)=1 means that there exists a path from node u
    to node v; alternatively (u,v)=0.

    The distance matrix contains lengths of shortest paths between all
    pairs of nodes. An entry (u,v) represents the length of shortest path 
    from node u to  node v. The average shortest path length is the 
    characteristic path length of the network.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed/undirected connection matrix

    Returns
    -------
    R : NxN np.ndarray
        binary reachability matrix
    D : NxN np.ndarray
        distance matrix

    Notes
    -----
    faster but more memory intensive than "breadthdist.m".
    '''
    def reachdist2(CIJ,CIJpwr,R,D,n,powr,col,row):
        CIJpwr=np.dot(CIJpwr,CIJ)
        R=np.logical_or(R,CIJpwr!=0)
        D+=R
        if powr<=n and np.any(R[row,col]==0):
            powr+=1
            R,D,powr=reachdist2(CIJ,CIJpwr,R,D,N,powr,col,row)
        return R,D,powr

    R=CIJ.copy()
    D=CIJ.copy()
    powr=2
    n=len(CIJ)
    CIJpwr=CIJ.copy()

    #check for vertices that have no incoming or outgoing connections
    #these are ignored by reachdist
    id=np.sum(CIJ,axis=0)
    od=np.sum(CIJ,axis=1)
    id0,=np.where(id==0)	#nothing goes in, so column(R) will be 0
    od0,=np.where(od==0)	#nothing comes out, so row(R) will be 0
    #use these colums and rows to check for reachability
    col=list(range(10)); col=np.delete(col,id0)
    row=list(range(10)); row=np.delete(row,od0)
    
    R,D,powr=reachdist2(CIJ,CIJpwr,R,D,n,powr,col,row)

    #'invert' CIJdist to get distances
    D=powr-D+1
    
    #put inf if no path found
    D[D==n+2]=np.inf
    D[:,id0]=np.inf
    D[od0,:]=np.inf

    return R,D

##############################################################################
# MODULARITY
##############################################################################

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
    if not np.size(ci): return ci #list is empty
    _,ci=np.unique(ci,return_inverse=True)
    ci+=1
    nr_indices=int(max(ci))
    ls=[]
    for c in range(nr_indices):
        ls.append([])
    for i,x in enumerate(ci):
        ls[ci[i]-1].append(i)
    return ls

def ls2ci(ls,zeroindexed=False):
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
    if ls is None or np.size(ls)==0: return ()	#list is empty
    nr_indices=sum(map(len,ls))
    ci=np.zeros((nr_indices,),dtype=int)
    z=int(not zeroindexed)
    for i,x in enumerate(ls):
        for j,y in enumerate(ls[i]):
            ci[ls[i][j]]=i+z
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
        objective-function matrix. builtin values 'modularity' uses Q-metric
        as objective function, or 'potts' uses Potts model Hamiltonian.
        Default value 'modularity'.
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
        ci = np.arange(n)+1
    else:
        if len(ci) != n:
            raise BCTParamError('initial ci vector size must equal N')
        _,ci = np.unique(ci, return_inverse=True)
        ci+=1
    Mb = ci.copy()

    if B=='modularity':
        B = W-gamma*np.outer(np.sum(W,axis=1), np.sum(W,axis=0))/s
    elif B=='potts':
        B = W-gamma*np.logical_not(W)
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
            B = (B+B.T)/2

    Hnm = np.zeros((n,n))
    for m in range(1, n+1):
        Hnm[:,m-1] = np.sum(B[:,ci==m], axis=1)   #node to module degree
    H = np.sum(Hnm, axis=1)                     #node degree
    Hm = np.sum(Hnm, axis=0)                    #module degree

    q0 = -np.inf
    #compute modularity
    q = np.sum( B[np.tile(ci, (n,1)) == np.tile(ci, (n,1)).T ])/s

    first_iteration = True

    while q-q0 > 1e-10:
        it=0
        flag = True
        while flag:
            it+=1
            if it>1000:
                raise BCTParamError('Modularity infinite loop style G. '
                    'Please contact the developer.')
            flag = False
            for u in np.random.permutation(n):
                ma = Mb[u]-1
                dQ = Hnm[u,:] - Hnm[u,ma] + B[u,u]  #algorithm condition
                dQ[ma] = 0

                max_dq = np.max(dQ)
                if max_dq>1e-10:
                    flag = True
                    mb = np.argmax(dQ)

                    Hnm[:,mb] += B[:,u]
                    Hnm[:,ma] -= B[:,u] #change node-to-module strengths

                    Hm[mb] += H[u]
                    Hm[ma] -= H[u]  #change module strengths

                    Mb[u] = mb+1

        _,Mb = np.unique(Mb, return_inverse=True)
        Mb += 1

        M0 = ci.copy()
        if first_iteration:
            ci = Mb.copy()
            first_iteration = False
        else:
            for u in range(1, n+1):
                ci[M0==u]=Mb[u-1]    #assign new modules

        n = np.max(Mb)
        b1 = np.zeros((n,n))
        for i in range(1, n+1):
            for j in range(i, n+1):
                #pool weights of nodes in same module
                bm=np.sum(B[np.ix_(Mb==i,Mb==j)])
                b1[i-1,j-1] = bm    
                b1[j-1,i-1] = bm
        B = b1.copy()

        Mb = np.arange(1, n+1)
        Hnm = B.copy()
        H=np.sum(B, axis=0)
        Hm=H.copy()

        q0 = q
        q=np.trace(B)/s         #compute modularity
        
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

    #set diagonal to mean weights
    np.fill_diagonal(W, 0)
    W[list(range(n)), list(range(n))] = (
        np.sum(W, axis=0)/np.sum(np.logical_not(W), axis=0) +
        np.sum(W.T, axis=0)/np.sum(np.logical_not(W.T), axis=0))/2

    #out/in norm squared
    No = np.sum(W**2, axis=1)
    Ni = np.sum(W**2, axis=0)

    #weighted in/out jaccard
    Jo = np.zeros((n,n))    
    Ji = np.zeros((n,n))
    
    for b in range(n):
        for c in range(n):
            Do = np.dot(W[b,:], W[c,:].T)
            Jo[b,c] = Do / (No[b]+No[c]-Do)

            Di = np.dot(W[:,b].T, W[:,c])
            Ji[b,c] = Di / (Ni[b]+Ni[c]-Di)

    #get link similarity
    A,B = np.where( np.logical_and( np.logical_or(W, W.T),
                                    np.triu(np.ones((n,n)), 1)))
    m = len(A)
    Ln = np.zeros((m,2), dtype=np.int32)    #link nodes
    Lw = np.zeros((m,))     #link weights

    for i in range(m):
        Ln[i,:] = (A[i], B[i])
        Lw[i] = (W[A[i], B[i]] + W[B[i], A[i]]) / 2

    ES = np.zeros((m,m), dtype=np.float32)    #link similarity
    for i in range(m):
        for j in range(m):
            if Ln[i,0] == Ln[j,0]:
                a=Ln[i,0]; b=Ln[i,1]; c=Ln[j,1]
            elif Ln[i,0] == Ln[j,1]:
                a=Ln[i,0]; b=Ln[i,1]; c=Ln[j,0]
            elif Ln[i,1] == Ln[j,0]:
                a=Ln[i,1]; b=Ln[i,0]; c=Ln[j,1]
            elif Ln[i,1] == Ln[j,1]:
                a=Ln[i,1]; b=Ln[i,0]; c=Ln[j,0]
            else:
                continue

            ES[i,j] = (W[a,b]*W[a,c]*Ji[b,c] + W[b,a]*W[c,a]*Jo[b,c])/2
    
    np.fill_diagonal(ES, 0)

    #perform hierarchical clustering

    C = np.zeros((m,m), dtype=np.int32)   #community affiliation matrix

    Nc = C.copy()
    Mc = np.zeros((m,m), dtype=np.float32)
    Dc = Mc.copy()           #community nodes, links, density

    U = np.arange(m)        #initial community assignments
    C[0,:] = np.arange(m)

    for i in range(m-1):
        print(('hierarchy %i'%i))

        for j in range(len(U)):     #loop over communities
            ixes = C[i,:] == U[j]   #get link indices

            links = np.sort(Lw[ixes])
            #nodes = np.sort(Ln[ixes,:].flat)

            nodes = np.sort(np.reshape(Ln[ixes,:], 2*np.size(np.where(ixes))))
            
            #get unique nodes
            nodulo = np.append(nodes[0], (nodes[1:])[nodes[1:] != nodes[:-1]])
            #nodulo = ((nodes[1:])[nodes[1:] != nodes[:-1]])

            nc = len(nodulo)
            #nc = len(nodulo)+1
            mc = np.sum(links)
            min_mc = np.sum( links[:nc-1] ) #minimal weight
            dc = (mc - min_mc) / (nc*(nc-1)/2 - min_mc) #community density

            if np.array(dc).shape is not ():
                print(dc)
                print((dc.shape))

            Nc[i,j] = nc
            Mc[i,j] = mc
            Dc[i,j] = dc if not np.isnan(dc) else 0

        C[i+1,:] = C[i,:]       #copy current partition

        if i in (6, 2692, 2693):
            import pdb
            pdb.set_trace()

        u1,u2 = np.where( ES[np.ix_(U,U)]==np.max(ES[np.ix_(U,U)]) )

        if np.size(u1) > 2:
            #pick one 
            wehr = np.where((u1 == u2[0]))

            uc = np.array((u1[0], u2[0]))
            ud = np.array((u2[wehr], u1[wehr]))

            u1 = uc
            u2 = ud

        #get unique links (implementation of sortrows)
        #ugl = np.array((u1,u2))
        ugl = np.sort((u1,u2), axis=1)
        ug_rows = ugl[np.argsort(ugl,axis=0 )[:,0]]
        #implementation of matlab unique(A, 'rows')
        unq_rows = np.vstack({tuple(row) for row in ug_rows})
        V = U[unq_rows]

        for j in range(len(V)):
            if type_clustering=='single':
                x = np.max(ES[V[j,:],:], axis=0)
            elif type_clustering=='complete':
                x = np.min(ES[V[j,:],:], axis=0)

            #assign distances to whole clusters
#            import pdb
#            pdb.set_trace()
            ES[V[j,:],:] = np.array((x,x))
            ES[:,V[j,:]] = np.transpose((x,x))

            # clear diagonal
            ES[V[j,0], V[j,0]] = 0
            ES[V[j,1], V[j,1]] = 0

            #merge communities
            C[i+1, C[i+1,:]==V[j,1]] = V[j,0]
            V[V==V[j,1]] = V[j,0]

        U = np.unique(C[i+1,:])
        if len(U) == 1:
            break

    #Dc[ np.where(np.isnan(Dc)) ]=0 
    i = np.argmax(np.sum(Dc*Mc, axis=1))

    U=np.unique(C[i,:])
    M=np.zeros((len(U),n))
    for j in range(len(U)):
        M[j, np.unique( Ln[C[i,:]==U[j],:])] = 1

    M = M[np.sum(M, axis=1)>2,:]
           
    return M

def modularity_dir(A,gamma=1,kci=None):
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
    n=len(A)							#number of vertices
    ki=np.sum(A,axis=0)					#in degree
    ko=np.sum(A,axis=1)					#out degree
    m=np.sum(ki)						#number of edges
    b=A-gamma*np.outer(ko,ki)/m
    B=b+b.T								#directed modularity matrix

    init_mod=np.arange(n)				#initial one big module
    modules=[]							#output modules list
 
    def recur(module):
        n=len(module)
        modmat=B[module][:,module]

        vals,vecs=linalg.eig(modmat)	#biggest eigendecomposition

        rlvals = np.real(vals)
        max_eigvec=np.squeeze(vecs[:,np.where(rlvals==np.max(rlvals))])
        if max_eigvec.ndim>1:			#if multiple max eigenvalues, pick one
            max_eigvec=max_eigvec[:,0]
        mod_asgn=np.squeeze((max_eigvec>=0)*2-1)	#initial module assignments
        q=np.dot(mod_asgn,np.dot(modmat,mod_asgn))	#modularity change
    
        if q>0:							#change in modularity was positive
            qmax=q
            np.fill_diagonal(modmat, 0)
            it=np.ma.masked_array(np.ones((n,)),False)
            mod_asgn_iter=mod_asgn.copy()
            while np.any(it):			#do some iterative fine tuning
                #this line is linear algebra voodoo
                q_iter=qmax-4*mod_asgn_iter*(np.dot(modmat,mod_asgn_iter))
                qmax=np.max(q_iter*it)
                imax,=np.where(q_iter==qmax)
                mod_asgn_iter[imax]*=-1	#does switching increase modularity?
                it[imax]=np.ma.masked
                if qmax>q:
                    q=qmax
                    mod_asgn=mod_asgn_iter
            if np.abs(np.sum(mod_asgn))==n:	#iteration yielded null module
                modules.append(np.array(module).tolist())
            else:
                mod1=module[np.where(mod_asgn==1)]
                mod2=module[np.where(mod_asgn==-1)]

                recur(mod1)
                recur(mod2)
        else:							#change in modularity was negative or 0
            modules.append(np.array(module).tolist())

    #adjustment to one-based indexing occurs in ls2ci
    if kci is None:
        recur(init_mod)
        ci=ls2ci(modules)
    else:
        ci=kci
    s=np.tile(ci,(n,1))
    q=np.sum(np.logical_not(s-s.T)*B/(2*m))
    return ci,q

def modularity_finetune_dir(W,ci=None,gamma=1,seed=None):
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

    n=len(W)							#number of nodes
    if ci is None:
        ci=np.arange(n)+1
    else:
        _,ci=np.unique(ci,return_inverse=True)
        ci+=1

    s=np.sum(W)							#weight of edges
    knm_o=np.zeros((n,n))				#node-to-module out degree
    knm_i=np.zeros((n,n))				#node-to-module in degree

    for m in range(np.max(ci)):
        knm_o[:,m]=np.sum(W[:,ci==(m+1)],axis=1)	
        knm_i[:,m]=np.sum(W[ci==(m+1),:],axis=0)

    k_o=np.sum(knm_o,axis=1)			#node out-degree
    k_i=np.sum(knm_i,axis=1)			#node in-degree
    km_o=np.sum(knm_o,axis=0)			#module out-degree
    km_i=np.sum(knm_i,axis=0)			#module out-degree

    flag=True
    while flag:
        flag=False
        for u in np.random.permutation(n):	#loop over nodes in random order
            ma=ci[u]-1					#current module of u
            #algorithm condition
            dq_o=((knm_o[u,:]-knm_o[u,ma]+W[u,u])-
                gamma*k_o[u]*(km_i-km_i[ma]+k_i[u])/s)
            dq_i=((knm_i[u,:]-knm_i[u,ma]+W[u,u])-
                gamma*k_i[u]*(km_o-km_o[ma]+k_o[u])/s)
            dq = (dq_o+dq_i)/2
            dq[ma]=0

            max_dq=np.max(dq)			#find maximal modularity increase
            if max_dq>1e-10:			#if maximal increase positive
                mb=np.argmax(dq)		#take only one value
                #print max_dq,mb

                knm_o[:,mb]+=W[u,:].T	#change node-to-module out-degrees
                knm_o[:,ma]-=W[u,:].T
                knm_i[:,mb]+=W[:,u]		#change node-to-module in-degrees
                knm_i[:,ma]-=W[:,u]
                km_o[mb]+=k_o[u]		#change module out-degrees
                km_o[ma]-=k_o[u]
                km_i[mb]+=k_i[u]		#change module in-degrees
                km_i[ma]-=k_i[u]

                ci[u]=mb+1				#reassign module
                flag=True

    _,ci=np.unique(ci,return_inverse=True)
    ci+=1
    m=np.max(ci)						#new number of modules
    w=np.zeros((m,m))					#new weighted matrix

    for u in range(m):
        for v in range(m):
            #pool weights of nodes in same module
            w[u,v]=np.sum(W[np.ix_(ci==u+1,ci==v+1)]) 

    q=np.trace(w)/s-gamma*np.sum(np.dot(w/s,w/s))
    return ci,q

def modularity_finetune_und(W,ci=None,gamma=1,seed=None):
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
    n=len(W)							#number of nodes
    if ci is None:
        ci=np.arange(n)+1
    else:
        _,ci=np.unique(ci,return_inverse=True)
        ci+=1

    s=np.sum(W)							#total weight of edges
    knm=np.zeros((n,n))					#node-to-module degree
    for m in range(np.max(ci)):
        knm[:,m]=np.sum(W[:,ci==(m+1)],axis=1)
    k=np.sum(knm,axis=1)				#node degree
    km=np.sum(knm,axis=0)				#module degree

    flag=True							
    while flag:
        flag=False

        for u in np.random.permutation(n):
        #for u in np.arange(n):
            ma=ci[u]-1
            #time.sleep(1)
            #algorithm condition
            dq=(knm[u,:]-knm[u,ma]+W[u,u])-gamma*k[u]*(km-km[ma]+k[u])/s
            #print np.sum(knm[u,:],knm[u,ma],W[u,u],gamma,k[u],np.sum(km),km[ma],k[u],s
            dq[ma]=0

            max_dq=np.max(dq)			#find maximal modularity increase
            if max_dq>1e-10:			#if maximal increase positive
                mb=np.argmax(dq)		#take only one value

                #print max_dq, mb

                knm[:,mb]+=W[:,u]		#change node-to-module degrees	
                knm[:,ma]-=W[:,u]
                km[mb]+=k[u]			#change module degrees
                km[ma]-=k[u]

                ci[u]=mb+1
                flag=True

    _,ci=np.unique(ci,return_inverse=True)
    ci+=1

    m=np.max(ci)
    w=np.zeros((m,m))
    for u in range(m):
        for v in range(m):
            #pool weights of nodes in same module
            wm=np.sum(W[np.ix_(ci==u+1,ci==v+1)])
            w[u,v]=wm
            w[v,u]=wm

    q=np.trace(w)/s-gamma*np.sum(np.dot(w/s,w/s))
    return ci,q

def modularity_finetune_und_sign(W,qtype='sta',gamma=1,ci=None,seed=None):
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

    n=len(W)							#number of nodes/modules
    if ci is None:
        ci=np.arange(n)+1
    else:
        _,ci=np.unique(ci,return_inverse=True);
        ci+=1

    W0=W*(W>0)							#positive weights matrix
    W1=-W*(W<0)							#negative weights matrix
    s0=np.sum(W0)						#positive sum of weights
    s1=np.sum(W1)						#negative sum of weights
    Knm0=np.zeros((n,n))				#positive node-to-module-degree
    Knm1=np.zeros((n,n))				#negative node-to-module degree

    for m in range(int(np.max(ci))):	#loop over modules
        Knm0[:,m]=np.sum(W0[:,ci==m+1],axis=1)
        Knm1[:,m]=np.sum(W1[:,ci==m+1],axis=1)

    Kn0=np.sum(Knm0,axis=1)				#positive node degree
    Kn1=np.sum(Knm1,axis=1)				#negative node degree
    Km0=np.sum(Knm0,axis=0)				#positive module degree
    Km1=np.sum(Knm1,axis=0)				#negative module degree

    if qtype=='smp': d0=1/s0; d1=1/s1				#dQ=dQ0/s0-dQ1/s1
    elif qtype=='gja': d0=1/(s0+s1); d1=1/(s0+s1)	#dQ=(dQ0-dQ1)/(s0+s1)
    elif qtype=='sta': d0=1/s0; d1=1/(s0+s1)		#dQ=dQ0/s0-dQ1/(s0+s1)
    elif qtype=='pos': d0=1/s0; d1=0				#dQ=dQ0/s0
    elif qtype=='neg': d0=0; d1=1/s1				#dQ=-dQ1/s1
    else: raise KeyError('modularity type unknown')
    
    if not s0:							#adjust for absent positive weights
        s0=1; d0=0
    if not s1:							#adjust for absent negative weights
        s1=1; d1=0

    flag=True							#flag for within hierarchy search
    h=0
    while flag:
        h+=1
        if h>1000:
            raise BCTParamError('Modularity infinite loop style D')
        flag=False
        for u in np.random.permutation(n):	#loop over nodes in random order
            ma=ci[u]-1						#current module of u
            dq0=((Knm0[u,:]+W0[u,u]-Knm0[u,ma])-
                gamma*Kn0[u]*(Km0+Kn0[u]-Km0[ma])/s0)
            dq1=((Knm1[u,:]+W1[u,u]-Knm1[u,ma])-
                gamma*Kn1[u]*(Km1+Kn1[u]-Km1[ma])/s1)
            dq=d0*dq0-d1*dq1			#rescaled changes in modularity
            dq[ma]=0					#no changes for same module

            #print dq,ma,u

            max_dq=np.max(dq)			#maximal increase in modularity
            mb=np.argmax(dq)			#corresponding module
            if max_dq>1e-10:			#if maximal increase is positive
                #print h,max_dq,mb,u
                flag=True
                ci[u]=mb+1				#reassign module

                Knm0[:,mb]+=W0[:,u]
                Knm0[:,ma]-=W0[:,u]
                Knm1[:,mb]+=W1[:,u]
                Knm1[:,ma]-=W1[:,u]
                Km0[mb]+=Kn0[u]
                Km0[ma]-=Kn0[u]
                Km1[mb]+=Kn1[u]
                Km1[ma]-=Kn1[u]

    _,ci=np.unique(ci,return_inverse=True); ci+=1
    m=np.tile(ci,(n,1))
    q0=(W0-np.outer(Kn0,Kn0)/s0)*(m==m.T)
    q1=(W1-np.outer(Kn1,Kn1)/s1)*(m==m.T)
    q=d0*np.sum(q0)-d1*np.sum(q1)

    return ci,q

def modularity_louvain_dir(W,gamma=1,hierarchy=False,seed=None):
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

    n=len(W)							#number of nodes
    s=np.sum(W)							#total weight of edges
    h=0									#hierarchy index
    ci=[]
    ci.append(np.arange(n)+1)			#hierarchical module assignments
    q=[]
    q.append(-1)						#hierarchical modularity index
    n0=n

    while True:	
        if h>300:
            raise BCTParamError('Modularity Infinite Loop Style E.  Please '
                'contact the developer with this error.')
        k_o=np.sum(W,axis=1)			#node in/out degrees
        k_i=np.sum(W,axis=0)
        km_o=k_o.copy()					#module in/out degrees
        km_i=k_i.copy()
        knm_o=W.copy()					#node-to-module in/out degrees
        knm_i=W.copy()

        m=np.arange(n)+1				#initial module assignments

        flag=True						#flag for within hierarchy search
        it=0
        while flag:
            it+=1
            if it>1000:
                raise BCTParamError('Modularity Infinite Loop Style F.  Please '
                    'contact the developer with this error.')
            flag=False
            
            #loop over nodes in random order
            for u in np.random.permutation(n):
                ma=m[u]-1
                #algorithm condition
                dq_o=((knm_o[u,:]-knm_o[u,ma]+W[u,u])-
                    gamma*k_o[u]*(km_i-km_i[ma]+k_i[u])/s)
                dq_i=((knm_i[u,:]-knm_i[u,ma]+W[u,u])-	
                    gamma*k_i[u]*(km_o-km_o[ma]+k_o[u])/s)
                dq = (dq_o+dq_i)/2
                dq[ma]=0

                max_dq=np.max(dq)		#find maximal modularity increase
                if max_dq>1e-10:		#if maximal increase positive
                    mb=np.argmax(dq)	#take only one value

                    knm_o[:,mb]+=W[u,:].T	#change node-to-module degrees
                    knm_o[:,ma]-=W[u,:].T
                    knm_i[:,mb]+=W[:,u]
                    knm_i[:,ma]-=W[:,u]
                    km_o[mb]+=k_o[u]		#change module out-degrees
                    km_o[ma]-=k_o[u]
                    km_i[mb]+=k_i[u]
                    km_i[ma]-=k_i[u]
                    
                    m[u]=mb+1			#reassign module
                    flag=True

        _,m=np.unique(m,return_inverse=True)
        m+=1
        h+=1
        ci.append(np.zeros((n0,)))
        #for i,mi in enumerate(m):		#loop through module assignments
        for i in range(n):
            #ci[h][np.where(ci[h-1]==i)]=mi	#assign new modules
            ci[h][np.where(ci[h-1]==i+1)] = m[i]

        n=np.max(m)						#new number of modules
        W1=np.zeros((n,n))				#new weighted matrix
        for i in range(n):
            for j in range(n):
                #pool weights of nodes in same module
                W1[i,j]=np.sum(W[np.ix_(m==i+1,m==j+1)])

        q.append(0)
        #compute modularity
        q[h]=np.trace(W1)/s-gamma*np.sum(np.dot(W1/s,W1/s))
        if q[h]-q[h-1]<1e-10:			#if modularity does not increase
            break

    ci=np.array(ci, dtype=int)
    if hierarchy:
        ci=ci[1:-1]; q=q[1:-1]
        return ci,q
    else:
        return ci[h-1],q[h-1]

def modularity_louvain_und(W,gamma=1,hierarchy=False,seed=None):
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

    n=len(W)							#number of nodes
    s=np.sum(W)							#weight of edges
    h=0									#hierarchy index
    ci=[]
    ci.append(np.arange(n)+1)			#hierarchical module assignments
    q=[]
    q.append(-1)						#hierarchical modularity values
    n0=n

    #knm = np.zeros((n,n))
    #for j in np.xrange(n0+1):
    #    knm[:,j] = np.sum(w[;, 

    while True:
        if h>300:
            raise BCTParamError('Modularity Infinite Loop Style B.  Please '
            'contact the developer with this error.')
        k=np.sum(W,axis=0)				#node degree
        Km=k.copy()						#module degree
        Knm=W.copy()					#node-to-module degree

        m=np.arange(n)+1				#initial module assignments

        flag=True						#flag for within-hierarchy search
        it=0
        while flag:
            it+=1
            if it>1000:
                raise BCTParamError('Modularity Infinite Loop Style C.  Please '
                'contact the developer with this error.')
            flag=False

            #loop over nodes in random order
            for i in np.random.permutation(n):	
                ma=m[i]-1
                #algorithm condition
                dQ=((Knm[i,:]-Knm[i,ma]+W[i,i])-
                    gamma*k[i]*(Km-Km[ma]+k[i])/s)
                dQ[ma]=0

                max_dq=np.max(dQ)		#find maximal modularity increase
                if max_dq>1e-10:		#if maximal increase positive
                    j=np.argmax(dQ)		#take only one value			
                    #print max_dq,j,dQ[j]

                    Knm[:,j]+=W[:,i]	#change node-to-module degrees
                    Knm[:,ma]-=W[:,i]

                    Km[j]+=k[i]			#change module degrees
                    Km[ma]-=k[i]

                    m[i]=j+1			#reassign module
                    flag=True

        _,m=np.unique(m,return_inverse=True)	#new module assignments
        #print m,h
        m+=1
        h+=1
        ci.append(np.zeros((n0,)))
        #for i,mi in enumerate(m):	#loop through initial module assignments
        for i in range(n):
            #print i, m[i], n0, h, len(m), n
            #ci[h][np.where(ci[h-1]==i+1)]=mi	#assign new modules
            ci[h][np.where(ci[h-1]==i+1)] = m[i]

        n=np.max(m)						#new number of modules
        W1=np.zeros((n,n))				#new weighted matrix
        for i in range(n):
            for j in range(i,n):
                #pool weights of nodes in same module
                wp=np.sum(W[np.ix_(m==i+1,m==j+1)])
                W1[i,j]=wp
                W1[j,i]=wp
        W=W1

        q.append(0)
        #compute modularity
        q[h]=np.trace(W)/s-gamma*np.sum(np.dot(W/s,W/s))
        if q[h]-q[h-1]<1e-10:			#if modularity does not increase
            break

    ci=np.array(ci, dtype=int)
    if hierarchy:
        ci=ci[1:-1]; q=q[1:-1]
        return ci,q
    else:
        return ci[h-1],q[h-1]

def modularity_louvain_und_sign(W,gamma=1,qtype='sta',seed=None):
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

    n=len(W)							#number of nodes

    W0=W*(W>0)							#positive weights matrix
    W1=-W*(W<0)							#negative weights matrix
    s0=np.sum(W0)						#weight of positive links
    s1=np.sum(W1)						#weight of negative links

    if qtype=='smp': d0=1/s0; d1=1/s1			#dQ=dQ0/s0-sQ1/s1
    elif qtype=='gja': d0=1/(s0+s1); d1=d0		#dQ=(dQ0-dQ1)/(s0+s1)
    elif qtype=='sta': d0=1/s0; d1=1/(s0+s1)	#dQ=dQ0/s0-dQ1/(s0+s1)
    elif qtype=='pos': d0=1/s0; d1=0			#dQ=dQ0/s0
    elif qtype=='neg': d0=0; d1=1/s1			#dQ=-dQ1/s1
    else: raise KeyError('modularity type unknown')

    if not s0:							#adjust for absent positive weights
        s0=1; d0=0
    if not s1:							#adjust for absent negative weights
        s1=1; d1=0

    h=1									#hierarchy index
    nh=n								#number of nodes in hierarchy
    ci=[None,np.arange(n)+1]			#hierarchical module assignments
    q=[-1,0]							#hierarchical modularity values
    while q[h]-q[h-1]>1e-10:
        if h>300: 
            raise BCTParamError('Modularity Infinite Loop Style A.  Please '
            'contact the developer with this error.')
        kn0=np.sum(W0,axis=0)			#positive node degree
        kn1=np.sum(W1,axis=0)			#negative node degree
        km0=kn0.copy()					#positive module degree
        km1=kn1.copy()					#negative module degree
        knm0=W0.copy()					#positive node-to-module degree
        knm1=W1.copy()					#negative node-to-module degree

        m=np.arange(nh)+1				#initial module assignments
        flag=True						#flag for within hierarchy search
        it=0
        while flag:
            it+=1
            if it>1000: 
                raise BCTParamError('Infinite Loop was detected and stopped. '
                'This was probably caused by passing in a directed matrix.')
            flag=False
            for u in np.random.permutation(nh):	#loop over nodes in random order
                ma=m[u]-1
                dQ0=((knm0[u,:]+W0[u,u]-knm0[u,ma])-
                    gamma*kn0[u]*(km0+kn0[u]-km0[ma])/s0)	#positive dQ
                dQ1=((knm1[u,:]+W1[u,u]-knm1[u,ma])-
                    gamma*kn1[u]*(km1+kn1[u]-km1[ma])/s1)	#negative dQ

                dQ=d0*dQ0-d1*dQ1		#rescaled changes in modularity
                dQ[ma]=0				#no changes for same module

                max_dQ=np.max(dQ)		#maximal increase in modularity
                if max_dQ>1e-10:		#if maximal increase is positive
                    flag=True
                    mb=np.argmax(dQ)

                    knm0[:,mb]+=W0[:,u]	#change positive node-to-module degrees
                    knm0[:,ma]-=W0[:,u]
                    knm1[:,mb]+=W1[:,u]	#change negative node-to-module degrees
                    knm1[:,ma]-=W1[:,u]
                    km0[mb]+=kn0[u]		#change positive module degrees
                    km0[ma]-=kn0[u]
                    km1[mb]+=kn1[u]		#change negative module degrees
                    km1[ma]-=kn1[u]

                    m[u]=mb+1			#reassign module

        h+=1
        ci.append(np.zeros((n,)))
        _,m=np.unique(m,return_inverse=True)
        m+=1

        for u in range(nh):		#loop through initial module assignments
            ci[h][np.where(ci[h-1]==u+1)]=m[u]	#assign new modules

        nh=np.max(m)					#number of new nodes
        wn0=np.zeros((nh,nh))			#new positive weights matrix
        wn1=np.zeros((nh,nh))
        
        for u in range(nh):
            for v in range(u,nh):
                wn0[u,v]=np.sum(W0[np.ix_(m==u+1,m==v+1)])
                wn1[u,v]=np.sum(W1[np.ix_(m==u+1,m==v+1)])
                wn0[v,u]=wn0[u,v]
                wn1[v,u]=wn1[u,v]

        W0=wn0
        W1=wn1

        q.append(0)
        #compute modularity
        q0=np.trace(W0)-np.sum(np.dot(W0,W0))/s0
        q1=np.trace(W1)-np.sum(np.dot(W1,W1))/s1
        q[h]=d0*q0-d1*q1

    _,ci_ret=np.unique(ci[-1],return_inverse=True)
    ci_ret+=1

    return ci_ret,q[-1]
                
def modularity_probtune_und_sign(W,qtype='sta',gamma=1,ci=None,p=.45,
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

    n=len(W)
    if ci is None:
        ci=np.arange(n)+1
    else:
        _,ci=np.unique(ci,return_inverse=True)
        ci+=1

    W0=W*(W>0)								#positive weights matrix
    W1=-W*(W<0)								#negative weights matrix
    s0=np.sum(W0)							#positive sum of weights
    s1=np.sum(W1)							#negative sum of weights
    Knm0=np.zeros((n,n))					#positive node-to-module degree
    Knm1=np.zeros((n,n))					#negative node-to-module degree

    for m in range(int(np.max(ci))):		#loop over initial modules
        Knm0[:,m]=np.sum(W0[:,ci==m+1],axis=1)
        Knm1[:,m]=np.sum(W1[:,ci==m+1],axis=1)

    Kn0=np.sum(Knm0,axis=1)					#positive node degree
    Kn1=np.sum(Knm1,axis=1)					#negative node degree
    Km0=np.sum(Knm0,axis=0)					#positive module degree
    Km1=np.sum(Knm1,axis=0)					#negaitve module degree

    if qtype=='smp': d0=1/s0; d1=1/s1				#dQ=dQ0/s0-dQ1/s1
    elif qtype=='gja': d0=1/(s0+s1); d1=1/(s0+s1)	#dQ=(dQ0-dQ1)/(s0+s1)
    elif qtype=='sta': d0=1/s0; d1=1/(s0+s1)		#dQ=dQ0/s0-dQ1/(s0+s1)
    elif qtype=='pos': d0=1/s0; d1=0				#dQ=dQ0/s0
    elif qtype=='neg': d0=0; d1=1/s1				#dQ=-dQ1/s1
    else: raise KeyError('modularity type unknown')
    
    if not s0:							#adjust for absent positive weights
        s0=1; d0=0
    if not s1:							#adjust for absent negative weights
        s1=1; d1=0

    for u in np.random.permutation(n):		#loop over nodes in random order
        ma=ci[u]-1							#current module
        r=np.random.random()<p
        if r:
            mb=np.random.randint(n)		 #select new module randomly
        else:
            dq0=((Knm0[u,:]+W0[u,u]-Knm0[u,ma])-
                gamma*Kn0[u]*(Km0+Kn0[u]-Km0[ma])/s0)
            dq1=((Knm1[u,:]+W1[u,u]-Knm1[u,ma])-
                gamma*Kn1[u]*(Km1+Kn1[u]-Km1[ma])/s1)
            dq=d0*dq0-d1*dq1
            dq[ma]=0

            max_dq=np.max(dq)
            mb=np.argmax(dq)

        if r or max_dq>1e-10:
            ci[u]=mb+1

            Knm0[:,mb]+=W0[:,u]
            Knm0[:,ma]-=W0[:,u]
            Knm1[:,mb]+=W1[:,u]
            Knm1[:,ma]-=W1[:,u]
            Km0[mb]+=Kn0[u]
            Km0[ma]-=Kn0[u]
            Km1[mb]+=Kn1[u]
            Km1[ma]-=Kn1[u]

    _,ci=np.unique(ci,return_inverse=True)
    ci+=1
    m=np.tile(ci,(n,1))
    q0=(W0-np.outer(Kn0,Kn0)/s0)*(m==m.T)
    q1=(W1-np.outer(Kn1,Kn1)/s1)*(m==m.T)
    q=d0*np.sum(q0)-d1*np.sum(q1)

    return ci,q

def modularity_und(A,gamma=1,kci=None):
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
    n=len(A)							#number of vertices
    k=np.sum(A,axis=0)					#degree
    m=np.sum(k)							#number of edges (each undirected edge
                                            #is counted twice)
    B=A-gamma*np.outer(k,k)/m			#initial modularity matrix

    init_mod=np.arange(n)				#initial one big module
    modules=[]							#output modules list

    def recur(module):
        n=len(module)
        modmat=B[module][:,module]
        modmat-=np.diag(np.sum(modmat,axis=0))

        vals,vecs=linalg.eigh(modmat)	#biggest eigendecomposition
        rlvals = np.real(vals)
        max_eigvec=np.squeeze(vecs[:,np.where(rlvals==np.max(rlvals))])
        if max_eigvec.ndim>1:			#if multiple max eigenvalues, pick one
            max_eigvec=max_eigvec[:,0]
        mod_asgn=np.squeeze((max_eigvec>=0)*2-1)	#initial module assignments
        q=np.dot(mod_asgn,np.dot(modmat,mod_asgn))	#modularity change

        if q>0:							#change in modularity was positive
            qmax=q
            np.fill_diagonal(modmat, 0)
            it=np.ma.masked_array(np.ones((n,)),False)
            mod_asgn_iter=mod_asgn.copy()
            while np.any(it):			#do some iterative fine tuning
                #this line is linear algebra voodoo
                q_iter=qmax-4*mod_asgn_iter*(np.dot(modmat,mod_asgn_iter))
                qmax=np.max(q_iter*it)
                imax,=np.where(q_iter==qmax)
                mod_asgn_iter[imax]*=-1	#does switching increase modularity?
                it[imax]=np.ma.masked
                if qmax>q:
                    q=qmax
                    mod_asgn=mod_asgn_iter
            if np.abs(np.sum(mod_asgn))==n:	#iteration yielded null module
                modules.append(np.array(module).tolist())
                return
            else:
                mod1=module[np.where(mod_asgn==1)]
                mod2=module[np.where(mod_asgn==-1)]

                recur(mod1)
                recur(mod2)
        else:							#change in modularity was negative or 0
            modules.append(np.array(module).tolist())
                
    #adjustment to one-based indexing occurs in ls2ci
    if kci is None:
        recur(init_mod)
        ci=ls2ci(modules)
    else:
        ci=kci
    s=np.tile(ci,(n,1))
    q=np.sum(np.logical_not(s-s.T)*B/m)
    return ci,q

def modularity_und_sign(W,ci,qtype='sta'):
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
    n=len(W)
    _,ci=np.unique(ci, return_inverse=True)
    ci+=1

    W0=W*(W>0)								#positive weights matrix
    W1=-W*(W<0)								#negative weights matrix
    s0=np.sum(W0)							#positive sum of weights
    s1=np.sum(W1)							#negative sum of weights
    Knm0=np.zeros((n,n))					#positive node-to-module degree
    Knm1=np.zeros((n,n))					#negative node-to-module degree

    for m in range(int(np.max(ci))):		#loop over initial modules
        Knm0[:,m]=np.sum(W0[:,ci==m+1],axis=1)
        Knm1[:,m]=np.sum(W1[:,ci==m+1],axis=1)

    Kn0=np.sum(Knm0,axis=1)					#positive node degree
    Kn1=np.sum(Knm1,axis=1)					#negative node degree
    Km0=np.sum(Knm0,axis=0)					#positive module degree
    Km1=np.sum(Knm1,axis=0)					#negaitve module degree

    if qtype=='smp': d0=1/s0; d1=1/s1				#dQ=dQ0/s0-dQ1/s1
    elif qtype=='gja': d0=1/(s0+s1); d1=1/(s0+s1)	#dQ=(dQ0-dQ1)/(s0+s1)
    elif qtype=='sta': d0=1/s0; d1=1/(s0+s1)		#dQ=dQ0/s0-dQ1/(s0+s1)
    elif qtype=='pos': d0=1/s0; d1=0				#dQ=dQ0/s0
    elif qtype=='neg': d0=0; d1=1/s1				#dQ=-dQ1/s1
    else: raise KeyError('modularity type unknown')

    if not s0:							#adjust for absent positive weights
        s0=1; d0=0
    if not s1:							#adjust for absent negative weights
        s1=1; d1=0

    m=np.tile(ci,(n,1))

    q0=(W0-np.outer(Kn0,Kn0)/s0)*(m==m.T)
    q1=(W1-np.outer(Kn1,Kn1)/s1)*(m==m.T)
    q=d0*np.sum(q0)-d1*np.sum(q1)

    return ci,q

def partition_distance(cx,cy):
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
    n=np.size(cx)
    _,cx=np.unique(cx,return_inverse=True)
    _,cy=np.unique(cy,return_inverse=True)
    _,cxy=np.unique(cx+cy*1j,return_inverse=True)

    cx += 1
    cy += 1
    cxy += 1
    
    Px=np.histogram(cx,bins=np.max(cx))[0]/n
    Py=np.histogram(cy,bins=np.max(cy))[0]/n
    Pxy=np.histogram(cxy,bins=np.max(cxy))[0]/n

    Hx=-np.sum(Px*np.log(Px))
    Hy=-np.sum(Py*np.log(Py))
    Hxy=-np.sum(Pxy*np.log(Pxy))

    Vin=(2*Hxy-Hx-Hy)/np.log(n)
    Min=2*(Hx+Hy-Hxy)/(Hx+Hy)
    return Vin,Min

##############################################################################
# MOTIFS
##############################################################################

motiflib='motif34lib.mat'

#FIXME there may be some subtle bugs here
def find_motif34(m,n=None):
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
    from scipy import io; import os
    fname=os.path.join(os.path.dirname(__file__),motiflib)
    z=(0,)
    if n==3:
        mot=io.loadmat(fname)
        m3=mot['m3']; id3=mot['id3'].squeeze()
        ix,=np.where(id3==m)
        M=np.zeros((3,3,len(ix)))
        for i,ind in enumerate(ix):
            M[:,:,i]=np.reshape(np.concatenate(
                (z,m3[ind,0:3],z,m3[ind,3:6],z)),(3,3))
    elif n==4:
        mot=io.loadmat(fname)
        m4=mot['m4']; id4=mot['id4'].squeeze()
        ix,=np.where(id4==m)
        M=np.zeros((4,4,len(ix)))
        for i,ind in enumerate(ix):
            M[:,:,i]=np.reshape(np.concatenate(
                (z,m4[ind,0:4],z,m4[ind,4:8],z,m4[ind,8:12],z)),(4,4))
    elif n is None:
        try:
            m=np.array(m)
        except TypeError:
            raise BCTParamError('motif matrix must be an array-like')
        if m.shape[0]==3:
            M,=np.where(motif3struct_bin(m))
        elif m.shape[0]==4:
            M,=np.where(motif4struct_bin(m))
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
    from scipy import io; import os
    
    def motif3generate():
        n=0
        M=np.zeros((54,6),dtype=bool)	#isomorphs
        CL=np.zeros((54,6),dtype=np.uint8)#canonical labels (predecssors of IDs)
        cl=np.zeros((6,),dtype=np.uint8)
        for i in range(2**6):			#loop through all subgraphs
            m='{0:b}'.format(i)
            m=str().zfill(6-len(m))+m
            G=np.array(((0,m[2],m[4]),(m[0],0,m[5]),(m[1],m[3],0)),dtype=int)
            ko=np.sum(G,axis=1)
            ki=np.sum(G,axis=0)
            if np.all(ko+ki):			#if subgraph weakly connected
                u=np.array((ko,ki)).T	
                cl.flat=u[np.lexsort((ki,ko))]
                CL[n,:]=cl				#assign motif label to isomorph
                M[n,:]=np.array((G.T.flat[1:4],G.T.flat[5:8])).flat
                n+=1

        #convert CLs into motif IDs
        _,ID=np.unique(CL.view(CL.dtype.descr*CL.shape[1]),return_inverse=True)
        ID+=1

        #convert IDs into sporns & kotter classification
        id_mika=(1,3,4,6,7,8,11); id_olaf=(-3,-6,-1,-11,-4,-7,-8)
        for mika,olaf in zip(id_mika,id_olaf):
            ID[ID==mika]=olaf
        ID=np.abs(ID)

        ix=np.argsort(ID)
        ID=ID[ix]						#sort IDs
        M=M[ix,:]						#sort isomorphs
        N=np.squeeze(np.sum(M,axis=1))	#number of edges
        Mn=np.array(np.sum(np.tile(np.power(10,np.arange(5,-1,-1)),
            (M.shape[0],1))*M,axis=1),dtype=np.uint32)
        return M,Mn,ID,N
    
    def motif4generate():
        n=0
        M=np.zeros((3834,12),dtype=bool)	#isomorphs
        CL=np.zeros((3834,16),dtype=np.uint8)	#canonical labels
        cl=np.zeros((16,),dtype=np.uint8)
        for i in range (2**12):			#loop through all subgraphs
            m='{0:b}'.format(i)
            m=str().zfill(12-len(m))+m
            G=np.array(((0,m[3],m[6],m[9]),(m[0],0,m[7],m[10]),
                (m[1],m[4],0,m[11]),(m[2],m[5],m[8],0)),dtype=int)
            Gs=G+G.T
            v=Gs[0,:]
            for j in range(2):
                v=np.any(Gs[v!=0,:],axis=0)+v
            if np.all(v):					#if subgraph weakly connected
                G2=np.dot(G,G)!=0
                ko=np.sum(G,axis=1)
                ki=np.sum(G,axis=0)
                ko2=np.sum(G2,axis=1)
                ki2=np.sum(G2,axis=0)

                u=np.array((ki,ko,ki2,ko2)).T	
                cl.flat=u[np.lexsort((ko2,ki2,ko,ki))]
                CL[n,:]=cl				#assign motif label to isomorph
                M[n,:]=np.array((G.T.flat[1:5],G.T.flat[6:10],
                    G.T.flat[11:15])).flat
                n+=1

        #convert CLs into motif IDs
        _,ID=np.unique(CL.view(CL.dtype.descr*CL.shape[1]),return_inverse=True)
        ID+=1

        ix=np.argsort(ID)
        ID=ID[ix]						#sort IDs
        M=M[ix,:]						#sort isomorphs
        N=np.sum(M,axis=1)				#number of edges
        Mn=np.array(np.sum(np.tile(np.power(10,np.arange(11,-1,-1)),
            (M.shape[0],1))*M,axis=1),dtype=np.uint64)
        return M,Mn,ID,N

    dir=os.path.dirname(__file__)
    fname=os.path.join(dir,motiflib)
    if os.path.exists(fname):
        print("motif34lib already exists"); return

    m3,m3n,id3,n3=motif3generate()
    m4,m4n,id4,n4=motif4generate()

    io.savemat(fname,mdict={'m3':m3,'m3n':m3n,'id3':id3,'n3':n3,
        'm4':m4,'m4n':m4n,'id4':id4,'n4':n4})

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
    from scipy import io; import os
    fname=os.path.join(os.path.dirname(__file__),motiflib)
    mot=io.loadmat(fname)
    m3=mot['m3']; id3=mot['id3'].squeeze(); n3=mot['n3'].squeeze()

    n=len(A)							#number of vertices in A
    f=np.zeros((13,))					#motif count for whole graph
    F=np.zeros((13,n))					#motif frequency

    A=binarize(A,copy=True)				#ensure A is binary
    As=np.logical_or(A,A.T)				#symmetrized adjmat

    for u in range(n-2):
        #v1: neighbors of u (>u)
        V1=np.append(np.zeros((u,),dtype=int),As[u,u+1:n+1])
        for v1 in np.where(V1)[0]:
            #v2: neighbors of v1 (>u)
            V2=np.append(np.zeros((u,),dtype=int),As[v1,u+1:n+1])
            V2[V1]=0					#not already in V1
                #and all neighbors of u (>v1)
            V2=np.logical_or(np.append(np.zeros((v1,)),As[u,v1+1:n+1]),V2)
            for v2 in np.where(V2)[0]:
                a=np.array((A[v1,u],A[v2,u],A[u,v1],A[v2,v1],A[u,v2],A[v1,2]))
                #find all contained isomorphs
                ix=(np.dot(m3,a)==n3)
                id=id3[ix]-1
                
                #unique motif occurrences
                idu,jx=np.unique(id,return_index=True)
                jx=np.append((0,),jx+1)

                mu=len(idu)				#number of unique motifs
                f2=np.zeros((mu,))
                for h in range(mu):	#for each unique motif
                    f2[h]=jx[h+1]-jx[h]	#and frequencies

                #then add to a cumulative count
                f[idu]+=f2
                #numpy indexing is teh sucks :(
                F[idu,u]+=f2; F[idu,v1]+=f2; F[idu,v2]+=f2

    return f,F

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
    from scipy import io; import os
    fname=os.path.join(os.path.dirname(__file__),motiflib)
    mot=io.loadmat(fname)
    m3=mot['m3']; id3=mot['id3'].squeeze(); n3=mot['n3'].squeeze()

    n=len(W)
    I=np.zeros((13,n))				#intensity
    Q=np.zeros((13,n))				#coherence
    F=np.zeros((13,n))				#frequency

    A=binarize(W,copy=True)			#create binary adjmat
    As=np.logical_or(A,A.T)			#symmetrized adjmat

    for u in range(n-2):
        #v1: neighbors of u (>u)
        V1=np.append(np.zeros((u,),dtype=int),As[u,u+1:n+1])
        for v1 in np.where(V1)[0]:
            #v2: neighbors of v1 (>u)
            V2=np.append(np.zeros((u,),dtype=int),As[v1,u+1:n+1])	
            V2[V1]=0					#not already in V1
                #and all neighbors of u (>v1)
            V2=np.logical_or(np.append(np.zeros((v1,)),As[u,v1+1:n+1]),V2)
            for v2 in np.where(V2)[0]:
                a=np.array((A[v1,u],A[v2,u],A[u,v1],A[v2,v1],A[u,v2],A[v1,v2]))
                ix=(np.dot(m3,a)==n3)
                m=np.sum(ix)

                w=np.array((W[v1,u],W[v2,u],W[u,v1],W[v2,v1],W[u,v2],W[v1,v2]))
                
                M=m3[ix,:]*np.tile(w,(m,1))
                id=id3[ix]-1
                l=n3[ix]
                x=np.sum(M,axis=1)/l		#arithmetic mean
                M[M==0]=1					#enable geometric mean
                i=np.prod(M,axis=1)**(1/l)	#intensity
                q=i/x						#coherence

                #unique motif occurrences
                idu,jx=np.unique(id,return_index=True)
                jx=np.append((0,),jx+1)

                mu=len(idu)				#number of unique motifs
                i2,q2,f2=np.zeros((3,mu))

                for h in range(mu):
                    i2[h]=np.sum(i[jx[h]+1:jx[h+1]+1])
                    q2[h]=np.sum(q[jx[h]+1:jx[h+1]+1])
                    f2[h]=jx[h+1]-jx[h]

                #then add to cumulative count
                I[idu,u]+=i2; I[idu,v1]+=i2; I[idu,v2]+=i2
                Q[idu,u]+=q2; Q[idu,v1]+=q2; Q[idu,v2]+=q2
                F[idu,u]+=f2; F[idu,v1]+=f2; F[idu,v2]+=f2

    return I,Q,F

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
    from scipy import io; import os
    fname=os.path.join(os.path.dirname(__file__),motiflib)
    mot=io.loadmat(fname)
    m3n=mot['m3n']; id3=mot['id3'].squeeze();

    n=len(A)							#number of vertices in A
    f=np.zeros((13,))					#motif count for whole graph
    F=np.zeros((13,n))					#motif frequency

    A=binarize(A,copy=True)				#ensure A is binary
    As=np.logical_or(A,A.T)				#symmetrized adjmat

    for u in range(n-2):
        #v1: neighbors of u (>u)
        V1=np.append(np.zeros((u,),dtype=int),As[u,u+1:n+1])
        for v1 in np.where(V1)[0]:
            #v2: neighbors of v1 (>u)
            V2=np.append(np.zeros((u,),dtype=int),As[v1,u+1:n+1])
            V2[V1]=0					#not already in V1
                #and all neighbors of u (>v1)
            V2=np.logical_or(np.append(np.zeros((v1,)),As[u,v1+1:n+1]),V2)
            for v2 in np.where(V2)[0]:
                a=np.array((A[v1,u],A[v2,u],A[u,v1],A[v2,v1],A[u,v2],A[v1,v2]))
                s=np.uint32(np.sum(np.power(10,np.arange(5,-1,-1))*a))
                ix=id3[np.squeeze(s==m3n)]-1
                F[ix,u]+=1; F[ix,v1]+=1; F[ix,v2]+=1
                f[ix]+=1

    return f,F

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
    from scipy import io; import os
    fname=os.path.join(os.path.dirname(__file__),motiflib)
    mot=io.loadmat(fname)
    m3=mot['m3'];m3n=mot['m3n'];id3=mot['id3'].squeeze();n3=mot['n3'].squeeze()

    n=len(W)						#number of vertices in W
    I=np.zeros((13,n))				#intensity
    Q=np.zeros((13,n))				#coherence
    F=np.zeros((13,n))				#frequency

    A=binarize(W,copy=True)			#create binary adjmat
    As=np.logical_or(A,A.T)			#symmetrized adjmat

    for u in range(n-2):
        #v1: neighbors of u (>u)
        V1=np.append(np.zeros((u,),dtype=int),As[u,u+1:n+1])
        for v1 in np.where(V1)[0]:
            #v2: neighbors of v1 (>u)
            V2=np.append(np.zeros((u,),dtype=int),As[v1,u+1:n+1])
            V2[V1]=0					#not already in V1
                #and all neighbors of u (>v1)
            V2=np.logical_or(np.append(np.zeros((v1,)),As[u,v1+1:n+1]),V2)
            for v2 in np.where(V2)[0]:
                a=np.array((A[v1,u],A[v2,u],A[u,v1],A[v2,v1],A[u,v2],A[v1,2]))
                s=np.uint32(np.sum(np.power(10,np.arange(5,-1,-1))*a))
                ix=np.squeeze(s==m3n)

                w=np.array((W[v1,u],W[v2,u],W[u,v1],W[v2,v1],W[u,v2],W[v1,v2]))
                
                M=w*m3[ix,:]
                id=id3[ix]-1
                l=n3[ix]
                x=np.sum(M,axis=1)/l	#arithmetic mean
                M[M==0]=1				#enable geometric mean
                i=np.prod(M,axis=1)**(1/l)	#intensity
                q=i/x					#coherence

                #add to cumulative counts
                I[id,u]+=i; I[id,v1]+=i; I[id,v2]+=i
                Q[id,u]+=q; Q[id,v1]+=q; Q[id,v2]+=q
                F[id,u]+=1; F[id,v1]+=1; F[id,v1]+=1

    return I,Q,F

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
    from scipy import io; import os
    fname=os.path.join(os.path.dirname(__file__),motiflib)
    mot=io.loadmat(fname)
    m4=mot['m4']; id4=mot['id4'].squeeze(); n4=mot['n4'].squeeze()

    n=len(A)
    f=np.zeros((199,))
    F=np.zeros((199,n))				#frequency

    A=binarize(A,copy=True)			#ensure A is binary	
    As=np.logical_or(A,A.T)			#symmetrized adjmat
    
    for u in range(n-3):
        #v1: neighbors of u (>u)
        V1=np.append(np.zeros((u,),dtype=int),As[u,u+1:n+1])
        for v1 in np.where(V1)[0]:
            V2=np.append(np.zeros((u,),dtype=int),As[v1,u+1:n+1])
            V2[V1]=0				#not already in V1
            #and all neighbors of u (>v1)
            V2=np.logical_or(np.append(np.zeros((v1,)),As[u,v1+1:n+1]),V2)
            for v2 in np.where(V2)[0]:
                vz=np.max((v1,v2))	#vz: largest rank node
                #v3: all neighbors of v2 (>u)
                V3=np.append(np.zeros((u,),dtype=int),As[v2,u+1:n+1])
                V3[V2]=0			#not already in V1 and V2
                #and all neighbors of v1 (>v2) 
                V3=np.logical_or(np.append(np.zeros((v2,)),As[v1,v2+1:n+1]),V3)
                V3[V1]=0			#not already in V1
                #and all neighbors of u (>vz)
                V3=np.logical_or(np.append(np.zeros((vz,)),As[u,vz+1:n+1]),V3)
                for v3 in np.where(V3)[0]:
                    a=np.array((A[v1,u],A[v2,u],A[v3,u],A[u,v1],A[v2,v1],
                        A[v3,v1],A[u,v2],A[v1,v2],A[v3,v2],A[u,v3],A[v1,v3],
                        A[v2,v3]))
    
                    ix=(np.dot(m4,a)==n4)	#find all contained isomorphs
                    id=id4[ix]-1

                    #unique motif occurrences
                    idu,jx=np.unique(id,return_index=True)
                    jx=np.append((0,),jx)
                    mu=len(idu)				#number of unique motifs
                    f2=np.zeros((mu,))
                    for h in range(mu):
                        f2[h]=jx[h+1]-jx[h]
                    
                    #add to cumulative count
                    f[idu]+=f2
                    F[idu,u]+=f2; F[idu,v1]+=f2; F[idu,v2]+=f2; F[idu,v3]+=f2

    return f,F

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
    from scipy import io; import os
    fname=os.path.join(os.path.dirname(__file__),motiflib)
    mot=io.loadmat(fname)
    m4=mot['m4']; id4=mot['id4'].squeeze(); n4=mot['n4'].squeeze()

    n=len(W)
    I=np.zeros((199,n))				#intensity
    Q=np.zeros((199,n))				#coherence
    F=np.zeros((199,n))				#frequency

    A=binarize(W,copy=True)			#ensure A is binary	
    As=np.logical_or(A,A.T)			#symmetrized adjmat

    for u in range(n-3):
        #v1: neighbors of u (>u)
        V1=np.append(np.zeros((u,),dtype=int),As[u,u+1:n+1])
        for v1 in np.where(V1)[0]:
            V2=np.append(np.zeros((u,),dtype=int),As[v1,u+1:n+1])
            V2[V1]=0				#not already in V1
            #and all neighbors of u (>v1)
            V2=np.logical_or(np.append(np.zeros((v1,)),As[u,v1+1:n+1]),V2)
            for v2 in np.where(V2)[0]:
                vz=np.max((v1,v2))	#vz: largest rank node
                #v3: all neighbors of v2 (>u)
                V3=np.append(np.zeros((u,),dtype=int),As[v2,u+1:n+1])
                V3[V2]=0			#not already in V1 and V2
                #and all neighbors of v1 (>v2) 
                V3=np.logical_or(np.append(np.zeros((v2,)),As[v1,v2+1:n+1]),V3)
                V3[V1]=0			#not already in V1
                #and all neighbors of u (>vz)
                V3=np.logical_or(np.append(np.zeros((vz,)),As[u,vz+1:n+1]),V3)
                for v3 in np.where(V3)[0]:
                    a=np.array((A[v1,u],A[v2,u],A[v3,u],A[u,v1],A[v2,v1],
                        A[v3,v1],A[u,v2],A[v1,v2],A[v3,v2],A[u,v3],A[v1,v3],
                        A[v2,v3]))
                    ix=(np.dot(m4,a)==n4)		#find all contained isomorphs
                
                    w=np.array((W[v1,u],W[v2,u],W[v3,u],W[u,v1],W[v2,v1],
                        W[v3,v1],W[u,v2],W[v1,v2],W[v3,v2],W[u,v3],W[v1,v3],
                        W[v2,v3]))

                    m=np.sum(ix)
                    M=m4[ix,:]*np.tile(w,(m,1))
                    id=id4[ix]-1
                    l=n4[ix]
                    x=np.sum(M,axis=1)/l		#arithmetic mean
                    M[M==0]=1					#enable geometric mean
                    i=np.prod(M,axis=1)**(1/l)	#intensity
                    q=i/x						#coherence

                    #unique motif occurrences
                    idu,jx=np.unique(id,return_index=True)
                    jx=np.append((0,),jx+1)

                    mu=len(idu)				#number of unique motifs
                    i2,q2,f2=np.zeros((3,mu))

                    for h in range(mu):
                        i2[h]=np.sum(i[jx[h]+1:jx[h+1]+1])
                        q2[h]=np.sum(q[jx[h]+1:jx[h+1]+1])
                        f2[h]=jx[h+1]-jx[h]

                    #then add to cumulative count
                    I[idu,u]+=i2; I[idu,v1]+=i2; I[idu,v2]+=i2; I[idu,v3]+=i2
                    Q[idu,u]+=q2; Q[idu,v1]+=q2; Q[idu,v2]+=q2; Q[idu,v3]+=q2
                    F[idu,u]+=f2; F[idu,v1]+=f2; F[idu,v2]+=f2; F[idu,v3]+=f2

    return I,Q,F

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
    from scipy import io; import os
    fname=os.path.join(os.path.dirname(__file__),motiflib)
    mot=io.loadmat(fname)
    m4n=mot['m4n']; id4=mot['id4'].squeeze(); 

    n=len(A)
    f=np.zeros((199,))
    F=np.zeros((199,n))				#frequency

    A=binarize(A,copy=True)			#ensure A is binary	
    As=np.logical_or(A,A.T)			#symmetrized adjmat
    
    for u in range(n-3):
        #v1: neighbors of u (>u)
        V1=np.append(np.zeros((u,),dtype=int),As[u,u+1:n+1])
        for v1 in np.where(V1)[0]:
            V2=np.append(np.zeros((u,),dtype=int),As[v1,u+1:n+1])
            V2[V1]=0				#not already in V1
            #and all neighbors of u (>v1)
            V2=np.logical_or(np.append(np.zeros((v1,)),As[u,v1+1:n+1]),V2)
            for v2 in np.where(V2)[0]:
                vz=np.max((v1,v2))	#vz: largest rank node
                #v3: all neighbors of v2 (>u)
                V3=np.append(np.zeros((u,),dtype=int),As[v2,u+1:n+1])
                V3[V2]=0			#not already in V1 and V2
                #and all neighbors of v1 (>v2) 
                V3=np.logical_or(np.append(np.zeros((v2,)),As[v1,v2+1:n+1]),V3)
                V3[V1]=0			#not already in V1
                #and all neighbors of u (>vz)
                V3=np.logical_or(np.append(np.zeros((vz,)),As[u,vz+1:n+1]),V3)
                for v3 in np.where(V3)[0]:
                    
                    a=np.array((A[v1,u],A[v2,u],A[v3,u],A[u,v1],A[v2,v1],
                        A[v3,v1],A[u,v2],A[v1,v2],A[v3,v2],A[u,v3],A[v1,v3],
                        A[v2,v3]))
                
                    s=np.uint64(np.sum(np.power(10,np.arange(11,-1,-1))*a))
                    ix=id4[np.squeeze(s==m4n)]
                    F[ix,u]+=1; F[ix,v1]+=1; F[ix,v2]+=1; F[ix,v3]+=1
                    f[ix]+=1

    return f,F

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
    from scipy import io; import os
    fname=os.path.join(os.path.dirname(__file__),motiflib)
    mot=io.loadmat(fname)
    m4=mot['m4'];m4n=mot['m4n'];id4=mot['id4'].squeeze();n4=mot['n4'].squeeze()

    n=len(W)
    I=np.zeros((199,n))				#intensity
    Q=np.zeros((199,n))				#coherence
    F=np.zeros((199,n))				#frequency

    A=binarize(W,copy=True)			#ensure A is binary	
    As=np.logical_or(A,A.T)			#symmetrized adjmat

    for u in range(n-3):
        #v1: neighbors of u (>u)
        V1=np.append(np.zeros((u,),dtype=int),As[u,u+1:n+1])
        for v1 in np.where(V1)[0]:
            V2=np.append(np.zeros((u,),dtype=int),As[v1,u+1:n+1])
            V2[V1]=0				#not already in V1
            #and all neighbors of u (>v1)
            V2=np.logical_or(np.append(np.zeros((v1,)),As[u,v1+1:n+1]),V2)
            for v2 in np.where(V2)[0]:
                vz=np.max((v1,v2))	#vz: largest rank node
                #v3: all neighbors of v2 (>u)
                V3=np.append(np.zeros((u,),dtype=int),As[v2,u+1:n+1])
                V3[V2]=0			#not already in V1 and V2
                #and all neighbors of v1 (>v2) 
                V3=np.logical_or(np.append(np.zeros((v2,)),As[v1,v2+1:n+1]),V3)
                V3[V1]=0			#not already in V1
                #and all neighbors of u (>vz)
                V3=np.logical_or(np.append(np.zeros((vz,)),As[u,vz+1:n+1]),V3)
                for v3 in np.where(V3)[0]:
                    a=np.array((A[v1,u],A[v2,u],A[v3,u],A[u,v1],A[v2,v1],
                        A[v3,v1],A[u,v2],A[v1,v2],A[v3,v2],A[u,v3],A[v1,v3],
                        A[v2,v3]))
                    s=np.uint64(np.sum(np.power(10,np.arange(11,-1,-1))*a))
                    #print np.shape(s),np.shape(m4n)
                    ix=np.squeeze(s==m4n)
                
                    w=np.array((W[v1,u],W[v2,u],W[v3,u],W[u,v1],W[v2,v1],
                        W[v3,v1],W[u,v2],W[v1,v2],W[v3,v2],W[u,v3],W[v1,v3],
                        W[v2,v3]))

                    M=w*m4[ix,:]
                    id=id4[ix]-1
                    l=n4[ix]
                    x=np.sum(M,axis=1)/l		#arithmetic mean
                    M[M==0]=1					#enable geometric mean
                    i=np.prod(M,axis=1)**(1/l)	#intensity
                    q=i/x						#coherence

                    #then add to cumulative count
                    I[id,u]+=i; I[id,v1]+=i; I[id,v2]+=i; I[id,v3]+=i
                    Q[id,u]+=q; Q[id,v1]+=q; Q[id,v2]+=q; Q[id,v3]+=q
                    F[id,u]+=1; F[id,v1]+=1; F[id,v2]+=1; F[id,v3]+=1

    return I,Q,F

##############################################################################
# OTHER
##############################################################################

def threshold_absolute(W,thr,copy=True):
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
    if copy: W=W.copy()
    np.fill_diagonal(W, 0)				#clear diagonal
    W[W<thr]=0							#apply threshold
    return W

def threshold_proportional(W,p,copy=True):
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
    if p>1 or p<0:
        raise BCTParamError('Threshold must be in range [0,1]')
    if copy: W=W.copy()
    n=len(W)						# number of nodes
    np.fill_diagonal(W, 0)			# clear diagonal

    if np.allclose(W, W.T):				# if symmetric matrix
        W[np.tril_indices(n)]=0		# ensure symmetry is preserved
        ud=2						# halve number of removed links
    else:
        ud=1

    ind=np.where(W)					# find all links

    I=np.argsort(W[ind])[::-1]		# sort indices by magnitude

    en=int(round((n*n-n)*p/ud))		# number of links to be preserved

    W[(ind[0][I][en:],ind[1][I][en:])]=0	# apply threshold
    #W[np.ix_(ind[0][I][en:], ind[1][I][en:])]=0

    if ud==2:						# if symmetric matrix
        W[:,:]=W+W.T						# reconstruct symmetry

    return W

def weight_conversion(W,wcm,copy=True):
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
    if wcm=='binarize': return binarize(W,copy)
    elif wcm=='normalize': return normalize(W,copy)
    elif wcm=='lengths': return invert(W,copy)
    else: raise NotImplementedError('Unknown weight conversion command.')

def binarize(W,copy=True):
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
    if copy: W=W.copy()
    W[W!=0]=1
    return W

def normalize(W,copy=True):
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
    if copy: W=W.copy()
    W/=np.max(np.abs(W))
    return W

def invert(W,copy=True):
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
    if copy: W=W.copy()
    E=np.where(W)
    W[E]=1./W[E]
    return W

def autofix(W,copy=True):
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
    if copy: W=W.copy()
    
    #zero diagonal
    np.fill_diagonal(W, 0) 
   
    #remove np.inf and np.nan   
    W[np.logical_or(np.where(np.isinf(W)), np.where(np.isnan(W)))] = 0

    #ensure exact binarity
    u = np.unique(W)
    if np.all(np.logical_or( np.abs(u)<1e-8, np.abs(u-1)<1e-8 )):
        W = np.around(W, decimal=5)

    #ensure exact symmetry
    if np.allclose(W, W.T):
        W = np.around(W, decimals=5)

    return W

##############################################################################
# PHYSICAL CONNECTIVITY
##############################################################################

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
    n=len(CIJ)
    k=np.size(np.where(CIJ.flatten()))
    kden=k/(n*n-n)
    return kden,n,k

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
    n=len(CIJ)
    k=np.size(np.where(np.triu(CIJ).flatten()))
    kden=k/((n*n-n)/2)
    return kden,n,k

def rentian_scaling(A,xyz,n):
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
    m=np.size(xyz,axis=0)			#find number of nodes in system

    #rescale coordinates so they are all greater than unity
    xyzn=xyz-np.tile(np.min(xyz,axis=0)-1,(m,1))

    #find the absolute minimum and maximum over all directions
    nmax=np.max(xyzn)
    nmin=np.min(xyzn)

    count=0
    N=np.zeros((n,))
    E=np.zeros((n,))

    #create partitions and count the number of nodes inside the partition (n)
    #and the number of edges traversing the boundary of the partition (e)
    while count<n:
        #define cube endpoints
        randx=np.sort((1+nmax-nmin)*np.random.random((2,)))

        #find nodes in cube
        l1=xyzn[:,0]>randx[0]; l2=xyzn[:,0]<randx[1]
        l3=xyzn[:,1]>randx[0]; l4=xyzn[:,1]<randx[1]
        l5=xyzn[:,2]>randx[0]; l6=xyzn[:,2]<randx[1]

        L,=np.where((l1&l2&l3&l4&l5&l6).flatten())
        if np.size(L):
            #count edges crossing at the boundary of the cube
            E[count]=np.sum(A[np.ix_(L,np.setdiff1d(list(range(m)),L))])
            #count nodes inside of the cube
            N[count]=np.size(L)
            count+=1

    return N,E 

##############################################################################
# REFERENCE
##############################################################################

def latmio_dir_connected(R,iter,D=None):
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
    iter : int
        rewiring parameter. Each edge is rewired approximately iter times.
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
    n=len(R)
    
    ind_rp=np.random.permutation(n)		#random permutation of nodes
    R=R.copy()
    R=R[np.ix_(ind_rp,ind_rp)]

    #create distance to diagonal matrix if not specified by user
    if D is None:
        D=np.zeros((n,n))
        un=np.mod(list(range(1,n)),n)
        um=np.mod(list(range(n-1,0,-1)),n)
        u=np.append((0,),np.where(un<um,un,um))

        for v in range(int(np.ceil(n/2))):
            D[n-v-1,:]=np.append(u[v+1:],u[:v+1])
            D[v,:]=D[n-v-1,:][::-1]

    i,j=np.where(R)
    k=len(i)
    iter*=k

    #maximal number of rewiring attempts per iteration
    max_attempts=np.round(n*k/(n*(n-1)))

    #actual number of successful rewirings
    eff=0

    for it in range(iter):
        att=0
        while att<=max_attempts:			#while not rewired
            rewire=True
            while True:
                e1=np.random.randint(k)
                e2=np.random.randint(k)
                while e1==e2:
                    e2=np.random.randint(k)
                a=i[e1]; b=j[e1]
                c=i[e2]; d=j[e2]

                if a!=c and a!=d and b!=c and b!=d:
                    break

            #rewiring condition
            if not (R[a,d] or R[c,b]):
                #lattice condition
                if (D[a,b]*R[a,b]+D[c,d]*R[c,d]>=D[a,d]*R[a,b]+D[c,b]*R[c,d]):
                    #connectedness condition
                    if not (np.any((R[a,c],R[d,b],R[d,c])) and
                            np.any((R[c,a],R[b,d],R[b,a]))):	
                        P=R[(a,c),:].copy()
                        P[0,b]=0;	P[0,d]=1
                        P[1,d]=0;	P[1,b]=1
                        PN=P.copy()
                        PN[0,a]=1;	PN[1,c]=1
                        while True:
                            P[0,:]=np.any(R[P[0,:]!=0,:],axis=0)
                            P[1,:]=np.any(R[P[1,:]!=0,:],axis=0)
                            P*=np.logical_not(PN)
                            PN+=P
                            if not np.all(np.any(P,axis=1)):
                                rewire=False
                                break
                            elif np.any(PN[0,(b,c)]) and np.any(PN[1,(d,a)]):
                                break
                    #end connectedness testing

                    if rewire:					#reassign edges
                        R[a,d]=R[a,b]; R[a,b]=0
                        R[c,b]=R[c,d]; R[c,d]=0

                        j.setflags(write=True)
                        j[e1]=d; j[e2]=b		#reassign edge indices
                        eff+=1
                        break
            att+=1

    Rlatt=R[np.ix_(ind_rp[::-1],ind_rp[::-1])]	#reverse random permutation

    return Rlatt,R,ind_rp,eff

def latmio_dir(R,iter,D=None):
    '''
    This function "latticizes" a directed network, while preserving the in-
    and out-degree distributions. In weighted networks, the function
    preserves the out-strength but not the in-strength distributions.

    Parameters
    ----------
    R : NxN np.ndarray
        directed binary/weighted connection matrix
    iter : int
        rewiring parameter. Each edge is rewired approximately iter times.
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
    n=len(R)

    ind_rp=np.random.permutation(n)		#randomly reorder matrix
    R=R.copy()
    R=R[np.ix_(ind_rp,ind_rp)]

    #create distance to diagonal matrix if not specified by user
    if D is None:
        D=np.zeros((n,n))
        un=np.mod(list(range(1,n)),n)
        um=np.mod(list(range(n-1,0,-1)),n)
        u=np.append((0,),np.where(un<um,un,um))

        for v in range(int(np.ceil(n/2))):
            D[n-v-1,:]=np.append(u[v+1:],u[:v+1])
            D[v,:]=D[n-v-1,:][::-1]

    i,j=np.where(R)
    k=len(i)
    iter*=k

    #maximal number of rewiring attempts per iteration
    max_attempts=np.round(n*k/(n*(n-1)))

    #actual number of successful rewirings
    eff=0

    for it in range(iter):
        att=0
        while att<=max_attempts:			#while not rewired
            while True:
                e1=np.random.randint(k)
                e2=np.random.randint(k)
                while e1==e2:
                    e2=np.random.randint(k)
                a=i[e1]; b=j[e1]
                c=i[e2]; d=j[e2]

                if a!=c and a!=d and b!=c and b!=d:
                    break

            #rewiring condition
            if not (R[a,d] or R[c,b]):
                #lattice condition
                if (D[a,b]*R[a,b]+D[c,d]*R[c,d]>=D[a,d]*R[a,b]+D[c,b]*R[c,d]):
                    R[a,d]=R[a,b]; R[a,b]=0
                    R[c,b]=R[c,d]; R[c,d]=0

                    j.setflags(write=True)
                    j[e1]=d; j[e2]=b		#reassign edge indices
                    eff+=1
                    break
            att+=1

    Rlatt=R[np.ix_(ind_rp[::-1],ind_rp[::-1])]	#reverse random permutation

    return Rlatt,R,ind_rp,eff

def latmio_und_connected(R,iter,D=None):
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
    iter : int
        rewiring parameter. Each edge is rewired approximately iter times.
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
    if not np.all(R==R.T):
        raise BCTParamError("Input must be undirected")

    if number_of_components(R) > 1:
        raise BCTParamError("Input is not connected")

    n=len(R)

    ind_rp=np.random.permutation(n)		#randomly reorder matrix
    R=R.copy()
    R=R[np.ix_(ind_rp,ind_rp)]

    if D is None:
        D=np.zeros((n,n))
        un=np.mod(list(range(1,n)),n)
        um=np.mod(list(range(n-1,0,-1)),n)
        u=np.append((0,),np.where(un<um,un,um))

        for v in range(int(np.ceil(n/2))):
            D[n-v-1,:]=np.append(u[v+1:],u[:v+1])
            D[v,:]=D[n-v-1,:][::-1]

    i,j=np.where(np.tril(R))
    k=len(i)
    iter*=k

    #maximal number of rewiring attempts per iteration
    max_attempts=np.round(n*k/(n*(n-1)/2))

    #actual number of successful rewirings
    eff=0

    for it in range(iter):
        att=0
        while att<=max_attempts:
            rewire=True
            while True:
                e1=np.random.randint(k)
                e2=np.random.randint(k)
                while e1==e2:
                    e2=np.random.randint(k)
                a=i[e1]; b=j[e1]
                c=i[e2]; d=j[e2]

                if a!=c and a!=d and b!=c and b!=d:
                    break

            if np.random.random()>.5:
                i.setflags(write=True); j.setflags(write=True)
                i[e2]=d; j[e2]=c		#flip edge c-d with 50% probability
                c=i[e2]; d=j[e2]		#to explore all potential rewirings

            #rewiring condition
            if not (R[a,d] or R[c,b]):
                #lattice condition
                if (D[a,b]*R[a,b]+D[c,d]*R[c,d]>=D[a,d]*R[a,b]+D[c,b]*R[c,d]):
                    #connectedness condition
                    if not (R[a,c] or R[b,d]):
                        P=R[(a,d),:].copy()
                        P[0,b]=0; P[1,c]=0
                        PN=P.copy()
                        PN[:,d]=1; PN[:,a]=1
                        while True:
                            P[0,:]=np.any(R[P[0,:]!=0,:],axis=0)
                            P[1,:]=np.any(R[P[1,:]!=0,:],axis=0)
                            P*=np.logical_not(PN)
                            if not np.all(np.any(P,axis=1)):
                                rewire=False
                                break
                            elif np.any(P[:,(b,c)]):
                                break
                            PN+=P
                    #end connectedness testing

                    if rewire:			#reassign edges
                        R[a,d]=R[a,b]; R[a,b]=0
                        R[d,a]=R[b,a]; R[b,a]=0
                        R[c,b]=R[c,d]; R[c,d]=0
                        R[b,c]=R[d,c]; R[d,c]=0

                        j.setflags(write=True)
                        j[e1]=d; j[e2]=b
                        eff+=1
                        break
            att+=1

    Rlatt=R[np.ix_(ind_rp[::-1],ind_rp[::-1])]
    return Rlatt,R,ind_rp,eff

def latmio_und(R,iter,D=None):
    '''
    This function "latticizes" an undirected network, while preserving the 
    degree distribution. The function does not preserve the strength 
    distribution in weighted networks.

    Parameters
    ----------
    R : NxN np.ndarray
        undirected binary/weighted connection matrix
    iter : int
        rewiring parameter. Each edge is rewired approximately iter times.
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
    n=len(R)

    ind_rp=np.random.permutation(n)		#randomly reorder matrix
    R=R.copy()
    R=R[np.ix_(ind_rp,ind_rp)]

    if D is None:
        D=np.zeros((n,n))
        un=np.mod(list(range(1,n)),n)
        um=np.mod(list(range(n-1,0,-1)),n)
        u=np.append((0,),np.where(un<um,un,um))

        for v in range(int(np.ceil(n/2))):
            D[n-v-1,:]=np.append(u[v+1:],u[:v+1])
            D[v,:]=D[n-v-1,:][::-1]

    i,j=np.where(np.tril(R))
    k=len(i)
    iter*=k

    #maximal number of rewiring attempts per iteration
    max_attempts=np.round(n*k/(n*(n-1)/2))

    #actual number of successful rewirings
    eff=0

    for it in range(iter):
        att=0
        while att<=max_attempts:
            while True:
                e1=np.random.randint(k)
                e2=np.random.randint(k)
                while e1==e2:
                    e2=np.random.randint(k)
                a=i[e1]; b=j[e1]
                c=i[e2]; d=j[e2]

                if a!=c and a!=d and b!=c and b!=d:
                    break

            if np.random.random()>.5:
                i.setflags(write=True); j.setflags(write=True)
                i[e2]=d; j[e2]=c		#flip edge c-d with 50% probability
                c=i[e2]; d=j[e2]		#to explore all potential rewirings

            #rewiring condition
            if not (R[a,d] or R[c,b]):
                #lattice condition
                if (D[a,b]*R[a,b]+D[c,d]*R[c,d]>=D[a,d]*R[a,b]+D[c,b]*R[c,d]):
                    R[a,d]=R[a,b]; R[a,b]=0
                    R[d,a]=R[b,a]; R[b,a]=0
                    R[c,b]=R[c,d]; R[c,d]=0
                    R[b,c]=R[d,c]; R[d,c]=0

                    j.setflags(write=True)
                    j[e1]=d; j[e2]=b	
                    eff+=1
                    break
            att+=1

    Rlatt=R[np.ix_(ind_rp[::-1],ind_rp[::-1])]
    return Rlatt,R,ind_rp,eff

def makeevenCIJ(n,k,sz_cl):
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
    #compute number of hierarchical levels and adjust cluster size
    mx_lvl=int(np.floor(np.log2(n)))
    sz_cl-=1

    #make a stupid little template
    t=np.ones((2,2))*2

    #check n against the number of levels
    Nlvl=2**mx_lvl
    if Nlvl!=n:
        print("Warning: n must be a power of 2")
    n=Nlvl

    #create hierarchical template
    for lvl in range(1,mx_lvl):
        s=2**(lvl+1)
        CIJ=np.ones((s,s))
        grp1=list(range(int(s/2)))
        grp2=list(range(int(s/2),s))
        ix1=np.add.outer(np.array(grp1)*s,grp1).flatten()
        ix2=np.add.outer(np.array(grp2)*s,grp2).flatten()
        CIJ.flat[ix1]=t					#numpy indexing is teh sucks :(
        CIJ.flat[ix2]=t
        CIJ+=1
        t=CIJ.copy()

    CIJ-=(np.ones((s,s))+mx_lvl*np.eye(s))

    #assign connection probabilities
    CIJp=(CIJ>=(mx_lvl-sz_cl))

    #determine nr of non-cluster connections left and their possible positions
    rem_k=k-np.size(np.where(CIJp.flatten()))
    if rem_k<0:
        print("Warning: K is too small, output matrix contains clusters only")
        return CIJp
    a,b=np.where(np.logical_not(CIJp+np.eye(n)))

    #assign remK randomly dstributed connections
    rp=np.random.permutation(len(a))
    a=a[rp[:rem_k]]
    b=b[rp[:rem_k]]
    for ai,bi in zip(a,b):
        CIJp[ai,bi]=1

    return np.array(CIJp,dtype=int)

def makefractalCIJ(mx_lvl,E,sz_cl):
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
    #make a stupid little template
    t=np.ones((2,2))*2

    #compute N and cluster size
    n=2**mx_lvl
    sz_cl-=1

    for lvl in range(1,mx_lvl):
        s=2**(lvl+1)
        CIJ=np.ones((s,s))
        grp1=list(range(int(s/2)))
        grp2=list(range(int(s/2),s))
        ix1=np.add.outer(np.array(grp1)*s,grp1).flatten()
        ix2=np.add.outer(np.array(grp2)*s,grp2).flatten()
        CIJ.flat[ix1]=t					#numpy indexing is teh sucks :(
        CIJ.flat[ix2]=t
        CIJ+=1
        t=CIJ.copy()

    CIJ-=(np.ones((s,s))+mx_lvl*np.eye(s))

    #assign connection probabilities
    ee=mx_lvl-CIJ-sz_cl
    ee=(ee>0)*ee
    prob=(1/E**ee)*(np.ones((s,s))-np.eye(s))
    CIJ=(prob>np.random.random((n,n)))

    #count connections
    k=np.sum(CIJ)

    return np.array(CIJ,dtype=int),k

def makerandCIJdegreesfixed(inv,outv):
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
    n=len(inv)
    k=np.sum(inv)
    in_inv=np.zeros((k,))
    out_inv=np.zeros((k,))
    i_in=0; i_out=0	

    for i in range(n):
        in_inv[i_in:i_in+inv[i]]=i
        out_inv[i_out:i_out+outv[i]]=i
        i_in+=inv[i]
        i_out+=outv[i]

    CIJ=np.eye(n)
    edges=np.array((out_inv,in_inv[np.random.permutation(k)]))

    #create CIJ and check for double edges and self connections
    for i in range(k):
        if CIJ[edges[0,i],edges[1,i]]:
            tried=set()
            while True:
                if len(tried)==k:
                    raise BCTParamError('Could not resolve the given '
                        'in and out vectors')	
                switch=np.random.randint(k)
                while switch in tried:
                    switch=np.random.randint(k)
                if not (CIJ[edges[0,i],edges[1,switch]] or
                        CIJ[edges[0,switch],edges[1,i]]):
                    CIJ[edges[0,switch],edges[1,switch]]=0
                    CIJ[edges[0,switch],edges[1,i]]=1
                    if switch<i:
                        CIJ[edges[0,switch],edges[1,switch]]=0
                        CIJ[edges[0,switch],edges[1,i]]=1
                    t=edges[1,i]
                    edges[1,i]=edges[1,switch]
                    edges[1,switch]=t
                    break
                tried.add(switch)
        else:
            CIJ[edges[0,i],edges[1,i]]=1

    CIJ-=np.eye(n)
    return CIJ

def makerandCIJ_dir(n,k):
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
    ix,=np.where(np.logical_not(np.eye(n)).flat)
    rp=np.random.permutation(np.size(ix))

    CIJ=np.zeros((n,n))
    CIJ.flat[ix[rp][:k]]=1
    return CIJ

def makerandCIJ_und(n,k):
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
    ix,=np.where(np.triu(np.logical_not(np.eye(n))).flat)
    rp=np.random.permutation(np.size(ix))
    
    CIJ=np.zeros((n,n))
    CIJ.flat[ix[rp][:k]]=1
    return CIJ

def makeringlatticeCIJ(n,k):
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
    #initialize
    CIJ=np.zeros((n,n))
    CIJ1=np.ones((n,n))
    kk=0
    count=0
    seq=list(range(1,n))
    seq2=list(range(n-1,0,-1))

    #fill in
    while kk<k:
        count+=1
        dCIJ=np.triu(CIJ1,seq[count])-np.triu(CIJ1,seq[count]+1)
        dCIJ2=np.triu(CIJ1,seq2[count])-np.triu(CIJ1,seq2[count]+1)
        dCIJ=dCIJ+dCIJ.T+dCIJ2+dCIJ2.T
        CIJ+=dCIJ
        kk=int(np.sum(CIJ))
    
    #remove excess connections
    overby=kk-k
    if overby:
        i,j=np.where(dCIJ)
        rp=np.random.permutation(np.size(i))
        for ii in range(overby):
            CIJ[i[rp[ii]],j[rp[ii]]]=0

    return CIJ

def maketoeplitzCIJ(n,k,s):
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
    from scipy import linalg,stats
    pf=stats.norm.pdf(list(range(1,n)),.5,s)
    template=linalg.toeplitz(np.append((0,),pf),r=np.append((0,),pf))
    template*=(k/np.sum(template))

    CIJ=np.zeros((n,n))
    itr=0
    while np.sum(CIJ)!=k:
        CIJ=(np.random.random((n,n))<template)
        itr+=1
        if itr>10000:
            raise BCTParamError('Infinite loop was caught generating toeplitz ' 
                'matrix.  This means the matrix could not be resolved with the '
                'specified parameters.')

    return CIJ

def null_model_dir_sign(W,bin_swaps=5,wei_freq=.1):
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
    W=W.copy()
    n=len(W)
    np.fill_diagonal(W, 0)						#clear diagonal
    Ap=(W>0)									#positive adjmat
    if np.size(np.where(Ap.flat))<(n*(n-1)):	#if Ap not fully connected
        Ap_r,_=randmio_dir(Ap,bin_swaps)			#randomized Ap
    else:
        Ap_r=Ap.copy()

    An=np.logical_not(Ap) 						#negative adjmat
    np.fill_diagonal(An, 0)
    An_r=np.logical_not(Ap_r) 					#randomized An
    np.fill_diagonal(An_r, 0)

    W0=np.zeros((n,n))
    for s in (1,-1):
        if s==1:
            Acur=Ap; A_rcur=Ap_r
        else:
            Acur=An; A_rcur=An_r

        Si=np.sum(W*Acur,axis=0)				#positive in-strength
        So=np.sum(W*Acur,axis=1)				#positive out-strength
        Wv=np.sort(W[Acur].flat)				#sorted weights vector
        i,j=np.where(A_rcur)
        Lij,=np.where(A_rcur.flat)				#weights indices

        P=np.outer(So,Si)
        
        if wei_freq==0:							#get indices of Lij that sort P
            Oind=np.argsort(P.flat[Lij])		#assign corresponding sorted
            W0.flat[Lij[Oind]]=s*Wv				#weight at this index
        else:
            wsize=np.size(Wv)
            wei_period=np.round(1/wei_freq)		#convert frequency to period
            lq=np.arange(wsize,0,-wei_period,dtype=int)	
            for m in lq:				#iteratively explore at this period
                Oind=np.argsort(P.flat[Lij])	#get indices of Lij that sort P
                R=np.random.permutation(m)[:np.min((m,wei_period))]
                for q,r in enumerate(R):
                    o=Oind[r]	#choose random index of sorted expected weight
                    W0.flat[Lij[o]]=s*Wv[r]		#assign corresponding weight

                    #readjust expected weighted probability for i[o],j[o]
                    f=1-Wv[r]/So[i[o]]	
                    P[i[o],:]*=f
                    f=1-Wv[r]/So[j[o]]
                    P[j[o],:]*=f

                    #readjust in-strength of i[o]
                    So[i[o]]-=Wv[r]
                    #readjust out-strength of j[o]
                    Si[j[o]]-=Wv[r]

                O=Oind[R]
                #remove current indices from further consideration
                Lij=np.delete(Lij,O)
                i=np.delete(i,O)
                j=np.delete(j,O)
                Wv=np.delete(Wv,O)

    rpos_in=np.corrcoef(np.sum(W*(W>0),axis=0),np.sum(W0*(W0>0),axis=0))
    rpos_ou=np.corrcoef(np.sum(W*(W>0),axis=1),np.sum(W0*(W0>0),axis=1))
    rneg_in=np.corrcoef(np.sum(-W*(W<0),axis=0),np.sum(-W0*(W0<0),axis=0))
    rneg_ou=np.corrcoef(np.sum(-W*(W<0),axis=1),np.sum(-W0*(W0<0),axis=1))
    return W0,(rpos_in[0,1],rpos_ou[0,1],rneg_in[0,1],rneg_ou[0,1])

def null_model_und_sign(W,bin_swaps=5,wei_freq=.1):
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
    if not np.all(W==W.T):
        raise BCTParamError("Input must be undirected")
    W=W.copy()
    n=len(W)
    np.fill_diagonal(W, 0)						#clear diagonal
    Ap=(W>0)									#positive adjmat
    if np.size(np.where(Ap.flat))<(n*(n-1)):	#if Ap not fully connected
        Ap_r,_=randmio_und(Ap,bin_swaps)		#randomized Ap
    else:
        Ap_r=Ap.copy()

    An=np.logical_not(Ap) 						#negative adjmat
    np.fill_diagonal(An, 0)
    An_r=np.logical_not(Ap_r) 					#randomized An
    np.fill_diagonal(An_r, 0)

    W0=np.zeros((n,n))
    for s in (1,-1):
        if s==1:
            Acur=Ap; A_rcur=Ap_r
        else:
            Acur=An; A_rcur=An_r

        S=np.sum(W*Acur,axis=0)					#strengths
        Wv=np.sort(W[np.where(np.triu(Acur))])	#sorted weights vector
        i,j=np.where(np.triu(A_rcur))
        Lij,=np.where(np.triu(A_rcur).flat)		#weights indices

        P=np.outer(S,S)

        if wei_freq==0:							#get indices of Lij that sort P
            Oind=np.argsort(P.flat[Lij])		#assign corresponding sorted
            W0.flat[Lij[Oind]]=s*Wv				#weight at this index
        else:
            wsize=np.size(Wv)
            wei_period=np.round(1/wei_freq)		#convert frequency to period
            lq=np.arange(wsize,0,-wei_period,dtype=int)	
            for m in lq:				#iteratively explore at this period
                Oind=np.argsort(P.flat[Lij])	#get indices of Lij that sort P
                R=np.random.permutation(m)[:np.min((m,wei_period))]
                for q,r in enumerate(R):
                    o=Oind[r]	#choose random index of sorted expected weight
                    W0.flat[Lij[o]]=s*Wv[r]		#assign corresponding weight

                    #readjust expected weighted probability for i[o],j[o]
                    f=1-Wv[r]/S[i[o]]	
                    P[i[o],:]*=f
                    P[:,i[o]]*=f
                    f=1-Wv[r]/S[j[o]]
                    P[j[o],:]*=f
                    P[:,j[o]]*=f

                    #readjust strength of i[o]
                    S[i[o]]-=Wv[r]
                    #readjust strength of j[o]
                    S[j[o]]-=Wv[r]

                O=Oind[R]
                #remove current indices from further consideration
                Lij=np.delete(Lij,O)
                i=np.delete(i,O)
                j=np.delete(j,O)
                Wv=np.delete(Wv,R)

    W0=W0+W0.T

    rpos_in=np.corrcoef(np.sum(W*(W>0),axis=0),np.sum(W0*(W0>0),axis=0))
    rpos_ou=np.corrcoef(np.sum(W*(W>0),axis=1),np.sum(W0*(W0>0),axis=1))
    rneg_in=np.corrcoef(np.sum(-W*(W<0),axis=0),np.sum(-W0*(W0<0),axis=0))
    rneg_ou=np.corrcoef(np.sum(-W*(W<0),axis=1),np.sum(-W0*(W0<0),axis=1))
    return W0,(rpos_in[0,1],rpos_ou[0,1],rneg_in[0,1],rneg_ou[0,1])
            
def randmio_dir_connected(R,iter):
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
    iter : int
        rewiring parameter. Each edge is rewired approximately iter times.
    
    Returns
    -------
    R : NxN np.ndarray
        randomized network
    eff : int
        number of actual rewirings carried out
    '''
    R=R.copy()
    n=len(R)
    i,j=np.where(R)
    k=len(i)
    iter*=k

    max_attempts=np.round(n*k/(n*(n-1)))
    eff=0

    for it in range(iter):
        att=0
        while att<=max_attempts:			#while not rewired
            rewire=True
            while True:
                e1=np.random.randint(k)
                e2=np.random.randint(k)
                while e1==e2:
                    e2=np.random.randint(k)
                a=i[e1]; b=j[e1]
                c=i[e2]; d=j[e2]

                if a!=c and a!=d and b!=c and b!=d:
                    break					#all 4 vertices must be different

            #rewiring condition
            if not (R[a,d] or R[c,b]):
                #connectedness condition
                if not (np.any((R[a,c],R[d,b],R[d,c])) and
                        np.any((R[c,a],R[b,d],R[b,a]))):
                    P=R[(a,c),:].copy()
                    P[0,b]=0; P[0,d]=1
                    P[1,d]=0; P[1,b]=1
                    PN=P.copy()
                    PN[0,a]=1; PN[1,c]=1
                    while True:
                        P[0,:]=np.any(R[P[0,:]!=0,:],axis=0)
                        P[1,:]=np.any(R[P[1,:]!=0,:],axis=0)
                        P*=np.logical_not(PN)
                        PN+=P
                        if not np.all(np.any(P,axis=1)):
                            rewire=False
                            break
                        elif np.any(PN[0,(b,c)]) and np.any(PN[1,(d,a)]):
                            break	
                #end connectedness testing

                if rewire:						#reassign edges
                    R[a,d]=R[a,b]; R[a,b]=0
                    R[c,b]=R[c,d]; R[c,d]=0

                    j.setflags(write=True)
                    j[e1]=d						#reassign edge indices
                    j[e2]=b
                    eff+=1
                    break
            att+=1
                
    return R,eff

def randmio_dir(R,iter):
    '''
    This function randomizes a directed network, while preserving the in-
    and out-degree distributions. In weighted networks, the function
    preserves the out-strength but not the in-strength distributions.

    Parameters
    ----------
    W : NxN np.ndarray
        directed binary/weighted connection matrix
    iter : int
        rewiring parameter. Each edge is rewired approximately iter times.
    
    Returns
    -------
    R : NxN np.ndarray
        randomized network
    eff : int
        number of actual rewirings carried out
    '''
    R=R.copy()
    n=len(R)
    i,j=np.where(R)
    k=len(i)
    iter*=k

    max_attempts=np.round(n*k/(n*(n-1)))
    eff=0

    for it in range(iter):
        att=0
        while att<=max_attempts:			#while not rewired
            while True:
                e1=np.random.randint(k)
                e2=np.random.randint(k)
                while e1==e2:
                    e2=np.random.randint(k)
                a=i[e1]; b=j[e1]
                c=i[e2]; d=j[e2]

                if a!=c and a!=d and b!=c and b!=d:
                    break					#all 4 vertices must be different

            #rewiring condition
            if not (R[a,d] or R[c,b]):
                R[a,d]=R[a,b]; R[a,b]=0
                R[c,b]=R[c,d]; R[c,d]=0

                i.setflags(write=True); j.setflags(write=True)
                i[e1]=d; j[e2]=b			#reassign edge indices
                eff+=1
                break
            att+=1
                
    return R,eff

def randmio_und_connected(R,iter):
    '''
    This function randomizes an undirected network, while preserving the 
    degree distribution. The function does not preserve the strength 
    distribution in weighted networks. The function also ensures that the 
    randomized network maintains connectedness, the ability for every node 
    to reach every other node in the network. The input network for this 
    function must be connected.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected binary/weighted connection matrix
    iter : int
        rewiring parameter. Each edge is rewired approximately iter times.
    
    Returns
    -------
    R : NxN np.ndarray
        randomized network
    eff : int
        number of actual rewirings carried out
    '''
    if not np.all(R==R.T):
        raise BCTParamError("Input must be undirected")

    if number_of_components(R) > 1:
        raise BCTParamError("Input is not connected")

    R=R.copy()
    n=len(R)
    i,j=np.where(np.tril(R))
    k=len(i)
    iter*=k

    #maximum number of rewiring attempts per iteration
    max_attempts=np.round(n*k/(n*(n-1)))
    #actual number of successful rewirings
    eff=0
    
    for it in range(iter):
        att=0
        while att<=max_attempts:			#while not rewired
            rewire=True
            while True:
                e1=np.random.randint(k)
                e2=np.random.randint(k)
                while e1==e2:
                    e2=np.random.randint(k)
                a=i[e1]; b=j[e1]
                c=i[e2]; d=j[e2]

                if a!=c and a!=d and b!=c and b!=d:
                    break					#all 4 vertices must be different

            if np.random.random()>.5:

                i.setflags(write=True); j.setflags(write=True)
                i[e2]=d; j[e2]=c			#flip edge c-d with 50% probability
                c=i[e2]; d=j[e2]			#to explore all potential rewirings

            #rewiring condition
            if not (R[a,d] or R[c,b]):
                #connectedness condition
                if not (R[a,c] or R[b,d]):
                    P=R[(a,d),:].copy()
                    P[0,b]=0; P[1,c]=0
                    PN=P.copy()
                    PN[:,d]=1; PN[:,a]=1
                    while True:
                        PN[0,:]=np.any(R[P[0,:]!=0,:],axis=0)
                        PN[1,:]=np.any(R[P[1,:]!=0,:],axis=0)
                        P*=np.logical_not(PN)
                        if not np.all(np.any(P,axis=1)):
                            rewire=0
                            break
                        elif np.any(P[:,(b,c)]):
                            break
                        PN+=P
                #end connectedness testing

                if rewire:
                    R[a,d]=R[a,b]; R[a,b]=0
                    R[d,a]=R[b,a]; R[b,a]=0
                    R[c,b]=R[c,d]; R[c,d]=0
                    R[b,c]=R[d,c]; R[d,c]=0

                    j.setflags(write=True)
                    j[e1]=d; j[e2]=b			#reassign edge indices
                    eff+=1
                    break
            att+=1

    return R,eff

def randmio_und(R,iter):
    '''
    This function randomizes an undirected network, while preserving the 
    degree distribution. The function does not preserve the strength 
    distribution in weighted networks.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected binary/weighted connection matrix
    iter : int
        rewiring parameter. Each edge is rewired approximately iter times.
    
    Returns
    -------
    R : NxN np.ndarray
        randomized network
    eff : int
        number of actual rewirings carried out
    '''
    if not np.all(R==R.T):
        raise BCTParamError("Input must be undirected")
    R=R.copy()
    n=len(R)
    i,j=np.where(np.tril(R))
    k=len(i)
    iter*=k

    #maximum number of rewiring attempts per iteration
    max_attempts=np.round(n*k/(n*(n-1)))
    #actual number of successful rewirings
    eff=0
    
    for it in range(iter):
        att=0
        while att<=max_attempts:			#while not rewired
            while True:
                e1,e2=np.random.randint(k,size=(2,))
                while e1==e2: e2=np.random.randint(k)
                a=i[e1]; b=j[e1]
                c=i[e2]; d=j[e2]

                if a!=c and a!=d and b!=c and b!=d:
                    break					#all 4 vertices must be different

            if np.random.random()>.5:
                i.setflags(write=True); j.setflags(write=True)
                i[e2]=d; j[e2]=c		#flip edge c-d with 50% probability
                c=i[e2]; d=j[e2]		#to explore all potential rewirings

            #rewiring condition
            if not (R[a,d] or R[c,b]):
                R[a,d]=R[a,b]; R[a,b]=0
                R[d,a]=R[b,a]; R[b,a]=0
                R[c,b]=R[c,d]; R[c,d]=0
                R[b,c]=R[d,c]; R[d,c]=0

                j.setflags(write=True)
                j[e1]=d; j[e2]=b			#reassign edge indices
                eff+=1
                break
            att+=1

    return R,eff

def randmio_und_signed(R,iter):
    '''
    This function randomizes an undirected weighted network with positive
    and negative weights, while simultaneously preserving the degree 
    distribution of positive and negative weights. The function does not 
    preserve the strength distribution in weighted networks.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected binary/weighted connection matrix
    iter : int
        rewiring parameter. Each edge is rewired approximately iter times.
    
    Returns
    -------
    R : NxN np.ndarray
        randomized network
    '''
    if not np.all(R==R.T):
        raise BCTParamError("Input must be undirected")
    R=R.copy()
    i,j=np.where(np.tril(R))
    i_p,j_p=np.where(np.tril(R)>0)
    i_m,j_m=np.where(np.tril(R)<0)
    k=len(i)
    k_p=len(i_p)
    k_m=len(i_m)
    iter*=k

    if not (k_p and k_m):
        return randmio_und(R,iter)[0]

    for it in range(iter):					#while not rewired
        while True:
            while True:
                #choose two edges to rewire but make sure they are either
                #both positive or both negative
                do_pos=np.random.random()>.5	#randomly rewires pos or neg
                if do_pos: kcur=k_p; icur=i_p; jcur=j_p
                else: kcur=k_m; icur=i_m; jcur=j_m

                e1=np.random.randint(kcur)
                e2=np.random.randint(kcur)
                while e1==e2:
                    e2=np.random.randint(kcur)
                a=icur[e1]; b=jcur[e1]
                c=icur[e2]; d=jcur[e2]
                if a!=c and a!=d and b!=c and b!=d:
                    break					#all 4 vertices must be different

            if np.random.random()>.5:
                icur.setflags(write=True); jcur.setflags(write=True)
                icur[e2]=d; jcur[e2]=c		#flip edge c-d with 50% probability
                c=icur[e2]; d=jcur[e2]		#to explore all potential rewirings
            #rewiring condition
            if not (R[a,d] or R[c,b]):
                R[a,d]=R[a,b]; R[a,b]=0
                R[d,a]=R[b,a]; R[b,a]=0
                R[c,b]=R[c,d]; R[c,d]=0
                R[b,c]=R[d,c]; R[d,c]=0
                jcur.setflags(write=True)
                jcur[e1]=d					#reassign edge indices
                jcur[e2]=b
                break

    return R

def randomize_graph_partial_und(A,B,maxswap):
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
    A=A.copy()
    i,j=np.where(np.triu(A,1))
    i.setflags(write=True); j.setflags(write=True)
    m=len(i)

    nswap=0
    while nswap < maxswap: 
        while True:
            e1,e2=np.random.randint(m,size=(2,));
            while e1==e2: e2=np.random.randint(m)
            a=i[e1]; b=j[e1]
            c=i[e2]; d=j[e2]
        
            if a!=c and a!=d and b!=c and b!=d:
                break					#all 4 vertices must be different

        if np.random.random()>.5:
            i[e2]=d; j[e2]=c			#flip edge c-d with 50% probability
            c=i[e2]; d=j[e2]			#to explore all potential rewirings
            
        #rewiring condition
        if not (A[a,d] or A[c,b] or B[a,d] or B[c,b]): #avoid specified ixes
            A[a,d]=A[a,b]; A[a,b]=0
            A[d,a]=A[b,a]; A[b,a]=0
            A[c,b]=A[c,d]; A[c,d]=0
            A[b,c]=A[d,c]; A[d,c]=0

            j[e1]=d; j[e2]=b			#reassign edge indices
            nswap+=1
    return A

def randomizer_bin_und(R,alpha):
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
    R=binarize(R,copy=True)					#binarize
    if not np.all(R==R.T):
        raise BCTParamError('randomizer_bin_und only takes undirected matrices')

    ax=len(R)
    nr_poss_edges=(np.dot(ax,ax)-ax)/2			#find maximum possible edges

    savediag=np.diag(R)
    np.fill_diagonal(R, np.inf)				#replace diagonal with high value
    
    #if there are more edges than non-edges, invert the matrix to reduce
    #computation time.  "invert" means swap meaning of 0 and 1, not matrix
    #inversion

    i,j=np.where(np.triu(R,1))
    k=len(i)
    if k>nr_poss_edges/2:
        swap=True
        R=np.logical_not(R)
        np.fill_diagonal(R, np.inf)
        i,j=np.where(np.triu(R,1))
        k=len(i)
    else: swap=False

    #exclude fully connected nodes
    fullnodes=np.where((np.sum(np.triu(R,1),axis=0)+
        np.sum(np.triu(R,1),axis=1).T)==(ax-1))
    if np.size(fullnodes):
        R[fullnode,:]=0; R[:,fullnode]=0
        np.fill_diagonal(R, np.inf)
        i,j=np.where(np.triu(R,1))
        k=len(i)

    if k==0 or k>=(nr_poss_edges-1):
        raise BCTParamError("No possible randomization")
    
    for it in range(k):
        if np.random.random()>alpha:
            continue						#rewire alpha% of edges

        a=i[it]; b=j[it]					#it is the chosen edge from a<->b
        
        alliholes,=np.where(R[:,a]==0)		#find where each end can connect
        alljholes,=np.where(R[:,b]==0)

        #we can only use edges with connection to neither node
        i_intersect=np.intersect1d(alliholes,alljholes)
        #find which of these nodes are connected
        ii,jj=np.where(R[np.ix_(i_intersect,i_intersect)])

        #if there is an edge to switch
        if np.size(ii):
            #choose one randomly
            nummates=np.size(ii)
            mate=np.random.randint(nummates)

            #randomly orient the second edge
            if np.random.random()>.5:
                c=i_intersect[ii[mate]];	d=i_intersect[jj[mate]]
            else:
                d=i_intersect[ii[mate]];	c=i_intersect[jj[mate]]

            #swap the edges
            R[a,b]=0; R[c,d]=0
            R[b,a]=0; R[d,c]=0
            R[a,c]=1; R[b,d]=1
            R[c,a]=1; R[d,b]=1

            #update the edge index (this is inefficient)
            for m in range(k):
                if i[m]==d and j[m]==c:
                    i.setflags(write=True); j.setflags(write=True)
                    i[it]=c; j[m]=b
                elif i[m]==c and j[m]==d:
                    i.setflags(write=True); j.setflags(write=True)
                    j[it]=c; i[m]=b

    #restore fullnodes
    if np.size(fullnodes):
        R[fullnodes,:]=1; R[:,fullnodes]=1
    
    #restore inversion
    if swap:
        R=np.logical_not(R)

    #restore diagonal
    np.fill_diagonal(R, 0)
    R+=savediag

    return np.array(R,dtype=int)

##############################################################################
# SIMILARITY 
##############################################################################

def edge_nei_overlap_bd(CIJ):
    '''	
    This function determines the neighbors of two nodes that are linked by 
    an edge, and then computes their overlap.  Connection matrix must be
    binary and directed.  Entries of 'EC' that are 'inf' indicate that no
    edge is present.  Entries of 'EC' that are 0 denote "local bridges",
    i.e. edges that link completely non-overlapping neighborhoods.  Low
    values of EC indicate edges that are "weak ties".

    If CIJ is weighted, the weights are ignored. Neighbors of a node can be
    linked by incoming, outgoing, or reciprocal connections.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        directed binary/weighted connection matrix

    Returns
    -------
    EC : NxN np.ndarray
        edge neighborhood overlap matrix
    ec : Kx1 np.ndarray
        edge neighborhood overlap per edge vector
    degij : NxN np.ndarray
        degrees of node pairs connected by each edge
    '''

    ik,jk=np.where(CIJ)
    lel=len(CIJ[ik,jk])
    n=len(CIJ)

    _,_,deg=degrees_dir(CIJ)

    ec=np.zeros((lel,))
    degij=np.zeros((2,lel))
    for e in range(lel):
        neiik=np.setdiff1d(np.union1d(
            np.where(CIJ[ik[e],:]),np.where(CIJ[:,ik[e]])),(ik[e],jk[e]))
        neijk=np.setdiff1d(np.union1d(
            np.where(CIJ[jk[e],:]),np.where(CIJ[:,jk[e]])),(ik[e],jk[e]))
        ec[e]=len(np.intersect1d(neiik,neijk))/len(np.union1d(neiik,neijk))
        degij[:,e]=(deg[ik[e]],deg[jk[e]])

    EC=np.tile(np.inf,(n,n))
    EC[ik,jk]=ec
    return EC,ec,degij

def edge_nei_overlap_bu(CIJ):
    '''
    This function determines the neighbors of two nodes that are linked by 
    an edge, and then computes their overlap.  Connection matrix must be
    binary and directed.  Entries of 'EC' that are 'inf' indicate that no
    edge is present.  Entries of 'EC' that are 0 denote "local bridges", i.e.
    edges that link completely non-overlapping neighborhoods.  Low values
    of EC indicate edges that are "weak ties".

    If CIJ is weighted, the weights are ignored.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        undirected binary/weighted connection matrix

    Returns
    -------
    EC : NxN np.ndarray
        edge neighborhood overlap matrix
    ec : Kx1 np.ndarray
        edge neighborhood overlap per edge vector
    degij : NxN np.ndarray
        degrees of node pairs connected by each edge
    '''
    ik,jk=np.where(CIJ)
    lel=len(CIJ[ik,jk])
    n=len(CIJ)

    deg=degrees_und(CIJ)

    ec=np.zeros((lel,))
    degij=np.zeros((2,lel))
    for e in range(lel):
        neiik=np.setdiff1d(np.union1d(
            np.where(CIJ[ik[e],:]),np.where(CIJ[:,ik[e]])),(ik[e],jk[e]))
        neijk=np.setdiff1d(np.union1d(
            np.where(CIJ[jk[e],:]),np.where(CIJ[:,jk[e]])),(ik[e],jk[e]))
        ec[e]=len(np.intersect1d(neiik,neijk))/len(np.union1d(neiik,neijk))
        degij[:,e]=(deg[ik[e]],deg[jk[e]])

    EC=np.tile(np.inf,(n,n))
    EC[ik,jk]=ec
    return EC,ec,degij

def gtom(adj,nr_steps):
    '''
    The m-th step generalized topological overlap measure (GTOM) quantifies
    the extent to which a pair of nodes have similar m-th step neighbors.
    Mth-step neighbors are nodes that are reachable by a path of at most
    length m.

    This function computes the the M x M generalized topological overlap
    measure (GTOM) matrix for number of steps, numSteps. 
    
    Parameters
    ----------
    adj : NxN np.ndarray
        connection matrix
    nr_steps : int
        number of steps

    Returns
    -------
    gt : NxN np.ndarray
        GTOM matrix

    Notes
    -----
    When numSteps is equal to 1, GTOM is identical to the topological
    overlap measure (TOM) from reference [2]. In that case the 'gt' matrix
    records, for each pair of nodes, the fraction of neighbors the two
    nodes share in common, where "neighbors" are one step removed. As
    'numSteps' is increased, neighbors that are furter out are considered.
    Elements of 'gt' are bounded between 0 and 1.  The 'gt' matrix can be
    converted from a similarity to a distance matrix by taking 1-gt.
    '''
    bm=binarize(bm,copy=True)
    bm_aux=bm.copy()
    nr_nodes=len(adj)

    if nr_steps>nr_nodes:
        print("Warning: nr_steps exceeded nr_nodes. Setting nr_steps=nr_nodes")
    if nr_steps==0:
        return bm
    else:
        for steps in range(2,nr_steps):
            for i in range(nr_nodes):
                #neighbors of node i
                ng_col,=np.where(bm_aux[i,:]==1)
                #neighbors of neighbors of node i
                nng_row,nng_col=np.where(bm_aux[ng_col,:]==1)
                new_ng=np.setdiff1d(nng_col,(i,))

                #neighbors of neighbors of i become considered neighbors of i
                bm_aux[i,new_ng]=1
                bm_aux[new_ng,i]=1

        #numerator of GTOM formula
        numerator_mat=np.dot(bm_aux,bm_aux)+bm+np.eye(nr_nodes)

        #vector of node degrees
        bms=np.sum(bm_aux,axis=0)
        bms_r=np.tile(bms,(nr_nodes,1))
        
        denominator_mat=-bm+np.where(bms_r>bms_r.T,bms_r,bms_r.T)+1
        return numerator_mat/denominator_mat

def matching_ind(CIJ):
    '''
    For any two nodes u and v, the matching index computes the amount of
    overlap in the connection patterns of u and v. Self-connections and
    u-v connections are ignored. The matching index is a symmetric 
    quantity, similar to a correlation or a dot product.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        adjacency matrix

    Returns
    -------
    Min : NxN np.ndarray
        matching index for incoming connections
    Mout : NxN np.ndarray
        matching index for outgoing connections
    Mall : NxN np.ndarray
        matching index for all connections

    Notes
    -----
    Does not use self- or cross connections for comparison.
    Does not use connections that are not present in BOTH u and v.
    All output matrices are calculated for upper triangular only.
    '''
    n=len(CIJ)

    Min=np.zeros((n,n))
    Mout=np.zeros((n,n))
    Mall=np.zeros((n,n))

    #compare incoming connections
    for i in range(n-1):
        for j in range(i+1,n):
            c1i=CIJ[:,i]
            c2i=CIJ[:,j]
            usei=np.logical_or(c1i,c2i)
            usei[i]=0; usei[j]=0
            nconi=np.sum(c1i[usei])+np.sum(c2i[usei])
            if not nconi:
                Min[i,j]=0
            else:
                Min[i,j]=2*np.sum(np.logical_and(c1i[usei],c2i[usei]))/nconi

            c1o=CIJ[i,:]
            c2o=CIJ[j,:]
            useo=np.logical_or(c1o,c2o)
            useo[i]=0; useo[j]=0
            ncono=np.sum(c1o[useo])+np.sum(c2o[useo])
            if not ncono:
                Mout[i,j]=0
            else:
                Mout[i,j]=2*np.sum(np.logical_and(c1o[useo],c2o[useo]))/ncono

            c1a=np.ravel((c1i,c1o))
            c2a=np.ravel((c2i,c2o))
            usea=np.logical_or(c1a,c2a)
            usea[i]=0; usea[i+n]=0
            usea[j]=0; usea[j+n]=0
            ncona=np.sum(c1a[usea])+np.sum(c2a[usea])
            if not ncona:
                Mall[i,j]=0
            else:
                Mall[i,j]=2*np.sum(np.logical_and(c1a[usea],c2a[usea]))/ncona

    Min=Min+Min.T; Mout=Mout+Mout.T; Mall=Mall+Mall.T

    return Min,Mout,Mall 

def matching_ind_und(CIJ0):
    '''
    M0 = MATCHING_IND_UND(CIJ) computes matching index for undirected
    graph specified by adjacency matrix CIJ. Matching index is a measure of
    similarity between two nodes' connectivity profiles (excluding their
    mutual connection, should it exist).

    Parameters
    ----------
    CIJ : NxN np.ndarray
        undirected adjacency matrix

    Returns
    -------
    M0 : NxN np.ndarray
        matching index matrix
    '''
    K=np.sum(CIJ0,axis=0) 
    n=len(CIJ0)
    R=(K!=0)
    N=np.sum(R)
    xR,=np.where(R==0)
    CIJ=np.delete(np.delete(CIJ0,xR,axis=0),xR,axis=1)
    I=np.logical_not(np.eye(N))
    M=np.zeros((N,N))

    for i in range(N):
        c1=CIJ[i,:]
        use=np.logical_or(c1,CIJ)
        use[:,i]=0
        use*=I

        ncon1=c1*use
        ncon2=c1*CIJ
        ncon=np.sum(ncon1+ncon2,axis=1)
        print(ncon)

        M[:,i] = 2*np.sum(np.logical_and(ncon1,ncon2),axis=1)/ncon

    M*=I
    M[np.isnan(M)]=0
    M0=np.zeros((n,n))
    yR,=np.where(R)
    M0[np.ix_(yR,yR)]=M
    return M0

def dice_pairwise_und(a1,a2):
    '''
    Calculates pairwise dice similarity for each vertex between two 
    matrices. Treats the matrices as binary and undirected.

    Paramaters
    ----------
    A1 : NxN np.ndarray
        Matrix 1
    A2 : NxN np.ndarray
        Matrix 2

    Returns
    -------
    D : Nx1 np.ndarray
        dice similarity vector
    '''
    a1=binarize(a1,copy=True)
    a2=binarize(a2,copy=True)			#ensure matrices are binary

    n=len(a1)
    np.fill_diagonal(a1, 0)
    np.fill_diagonal(a2, 0)				#set diagonals to 0

    d=np.zeros((n,))					#dice similarity

    #calculate the common neighbors for each vertex
    for i in range(n):
        d[i]=2*(np.sum(np.logical_and(a1[:,i],a2[:,i]))/
            (np.sum(a1[:,i])+np.sum(a2[:,i])))

    return d

def corr_flat_und(a1,a2):
    '''
    Returns the correlation coefficient between two flattened adjacency
    matrices.  Only the upper triangular part is used to avoid double counting
    undirected matrices.  Similarity metric for weighted matrices.

    Parameters
    ----------
    A1 : NxN np.ndarray
        undirected matrix 1
    A2 : NxN np.ndarray
        undirected matrix 2

    Returns
    -------
    r : float
        Correlation coefficient describing edgewise similarity of a1 and a2
    '''
    n=len(a1)
    if len(a2)!=n:
        raise BCTParamError("Cannot calculate flattened correlation on "
            "matrices of different size")
    triu_ix=np.where(np.triu(np.ones((n,n)),1))
    return np.corrcoef(a1[triu_ix].flat,a2[triu_ix].flat)[0][1]

def corr_flat_dir(a1,a2):
    '''
    Returns the correlation coefficient between two flattened adjacency
    matrices.  Similarity metric for weighted matrices.

    Parameters
    ----------
    A1 : NxN np.ndarray
        directed matrix 1
    A2 : NxN np.ndarray
        directed matrix 2

    Returns
    -------
    r : float
        Correlation coefficient describing edgewise similarity of a1 and a2
    '''
    n=len(a1)
    if len(a2)!=n:
        raise BCTParamError("Cannot calculate flattened correlation on "
            "matrices of different size")
    ix=np.logical_not(np.eye(n))
    return np.corrcoef(a1[ix].flat,a2[ix].flat)[0][1]

##############################################################################
# VISUALIZATION
##############################################################################

def adjacency_plot_und(A,coor,tube=False):
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
    nr_edges = (n*n-1)//2

    #starts = np.zeros((nr_edges,3))
    #vecs = np.zeros((nr_edges,3))
    #adjdat = np.zeros((nr_edges,))

    ixes, = np.where(np.triu(np.ones((n,n)),1).flat)

    #i=0
    #for r2 in xrange(n):
    #	for r1 in xrange(r2):
    #		starts[i,:] = coor[r1,:]
    #		vecs[i,:] = coor[r2,:] - coor[r1,:]
    #		adjdat[i,:]
    #		i+=1

    adjdat = A.flat[ixes]

    A_r = np.tile(coor,(n,1,1))
    starts = np.reshape(A_r,(n*n,3))[ixes,:]

    vecs = np.reshape(A_r - np.transpose(A_r,(1,0,2)),(n*n,3))[ixes,:]

    #plotting
    fig=mlab.figure()
    
    nodesource = mlab.pipeline.scalar_scatter(
        coor[:,0],coor[:,1],coor[:,2],figure=fig)

    nodes = mlab.pipeline.glyph(nodesource, scale_mode='none',
        scale_factor=3., mode='sphere', figure=fig)
    nodes.glyph.color_mode='color_by_scalar'

    vectorsrc = mlab.pipeline.vector_scatter(
        starts[:,0],starts[:,1],starts[:,2],vecs[:,0],vecs[:,1],vecs[:,2],
        figure=fig)
    vectorsrc.mlab_source.dataset.point_data.scalars=adjdat

    thres=mlab.pipeline.threshold(vectorsrc,
        low=0.0001,up=np.max(A),figure=fig)
        
    vectors=mlab.pipeline.vectors(thres, colormap='YlOrRd', 
        scale_mode='vector', figure=fig)
    vectors.glyph.glyph.clamping=False
    vectors.glyph.glyph.color_mode='color_by_scalar'
    vectors.glyph.color_mode='color_by_scalar'
    vectors.glyph.glyph_source.glyph_position='head'
    vectors.actor.property.opacity=.7
    if tube:
        vectors.glyph.glyph_source.glyph_source = (vectors.glyph.glyph_source.
            glyph_dict['cylinder_source'])
        vectors.glyph.glyph_source.glyph_source.radius = 0.015
    else:
        vectors.glyph.glyph_source.glyph_source.glyph_type='dash'

    return fig

def align_matrices(m1,m2,dfun='sqrdiff',verbose=False,H=1e6,Texp=1,
    T0=1e-3,Hbrk=10):
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
    n=len(m1)
    if n<2:
        raise BCTParamError("align_matrix will infinite loop on a singleton "
            "or null matrix.")

    #define maxcost (greatest possible difference) and lowcost
    if dfun in ('absdiff','absdff'):
        maxcost=np.sum(np.abs(np.sort(m1.flat)-np.sort(m2.flat)[::-1]))
        lowcost=np.sum(np.abs(m1-m2))/maxcost
    elif dfun in ('sqrdiff','sqrdff'):
        maxcost=np.sum((np.sort(m1.flat)-np.sort(m2.flat)[::-1])**2)
        lowcost=np.sum((m1-m2)**2)/maxcost
    elif dfun=='cosang':
        maxcost=np.pi/2
        lowcost=np.arccos(np.dot(m1.flat,m2.flat)/
            np.sqrt(np.dot(m1.flat,m1.flat)*np.dot(m2.flat,m2.flat)))/maxcost
    else:
        raise BCTParamError('dfun must be absdiff or sqrdiff or cosang')

    mincost=lowcost
    anew=np.arange(n)
    amin=np.arange(n)
    h=0; hcnt=0

    #adjust annealing parameters from user provided coefficients
    #H determines the maximal number of steps (user-provided)
    #Texp determines the steepness of the temperature gradient
    Texp=1-Texp/H
    #T0 sets the initial temperature and scales the energy term (user provided)
    #Hbrk sets a break point for the stimulation
    Hbrk=H/Hbrk
    
    while h<H:
        h+=1; hcnt+=1
        #terminate if no new mincost has been found for some time
        if hcnt>Hbrk:
            break
        #current temperature
        T=T0*(Texp**h)

        #choose two positions at random and flip them
        atmp=anew.copy()
        r1,r2=np.random.randint(n,size=(2,))
        while r1==r2:
            r2=np.random.randint(n)
        atmp[r1]=anew[r2]
        atmp[r2]=anew[r1]
        m2atmp=m2[np.ix_(atmp,atmp)]
        if dfun in ('absdiff','absdff'):
            costnew=np.sum(np.abs(m1-m2atmp))/maxcost
        elif dfun in ('sqrdiff','sqrdff'):
            costnew=np.sum((m1-m2atmp)**2)/maxcost
        elif dfun=='cosang':
            costnew=np.arccos(np.dot(m1.flat,m2atmp.flat)/np.sqrt(
                np.dot(m1.flat,m1.flat)*np.dot(m2.flat,m2.flat)))/maxcost

        if costnew<lowcost or np.random.random()<np.exp(-(costnew-lowcost)/T):
            anew=atmp
            lowcost=costnew
            #is this the absolute best?
            if lowcost<mincost:
                amin=anew
                mincost=lowcost
                if verbose:
                    print(('step %i ... current lowest cost = %f' % (h,mincost)))
                hcnt=0
            #if the cost is 0 we're done
            if mincost==0:
                break
    if verbose:
        print(('step %i ... final lowest cost = %f' % (h,mincost)))

    M_reordered=m2[np.ix_(amin,amin)]
    M_indices=amin
    cost=mincost
    return M_reordered,M_indices,cost

def backbone_wu(CIJ,avgdeg):
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
    n=len(CIJ)
    if not np.all(CIJ==CIJ.T):
        raise BCTParamError('backbone_wu can only be computed for undirected '
            'matrices.  If your matrix is has noise, correct it with np.around')
    CIJtree=np.zeros((n,n))

    #find strongest edge (if multiple edges are tied, use only first one)
    i,j=np.where(np.max(CIJ)==CIJ)
    im=[i[0],i[1]]	#what?  why take two values?  doesnt that mess up multiples?
    jm=[j[0],j[1]]

    #copy into tree graph
    CIJtree[im,jm]=CIJ[im,jm]
    in_=im
    out=np.setdiff1d(list(range(n)),in_)

    #repeat n-2 times
    for ix in range(n-2):
        CIJ_io=CIJ[np.ix_(in_,out)]
        i,j=np.where(np.max(CIJ_io)==CIJ_io)
        #i,j=np.where(np.max(CIJ[in_,out])==CIJ[in_,out])
        print((i,j))
        im=in_[i[0]]
        jm=out[j[0]]

        #copy into tree graph
        CIJtree[im,jm]=CIJ[im,jm]
        CIJtree[jm,im]=CIJ[jm,im]
        in_=np.append(in_,jm)
        out=np.setdiff1d(list(range(n)),in_)

    #now add connections back with the total number of added connections
    #determined by the desired avgdeg

    CIJnotintree=CIJ*np.logical_not(CIJtree)
    ix,=np.where(CIJnotintree.flat)
    a=np.sort(CIJnotintree.flat[ix])[::-1]
    cutoff=avgdeg*n-2*(n-1)-1
    #if the avgdeg req is already satisfied, skip this
    if cutoff>=np.size(a):
        CIJclus=CIJtree.copy()
    else:
        thr=a[cutoff]
        CIJclus=CIJtree+CIJnotintree*(CIJnotintree>=thr)

    return CIJtree,CIJclus

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
    c=c.copy()
    nr_c=np.max(c)
    ixes=np.argsort(c)
    c=c[ixes]

    bounds=[]

    for i in range(nr_c):
        ind=np.where(c==i+1)
        if np.size(ind):
            mn=np.min(ind)-.5
            mx=np.max(ind)+.5
            bounds.extend([mn,mx])

    bounds=np.unique(bounds)
    return bounds,ixes

def reorderMAT(m,H=5000,cost='line'):
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
    from scipy import linalg,stats
    m=m.copy()
    n=len(m)
    np.fill_diagonal(m, 0)

    #generate cost function
    if cost=='line':
        profile=stats.norm.pdf(list(range(1,n+1)),0,n/2)[::-1]
    elif cost=='circ':
        profile=stats.norm.pdf(list(range(1,n+1)),n/2,n/4)[::-1]
    else:
        raise BCTParamError('dfun must be line or circ')
    costf=linalg.toeplitz(profile,r=profile)

    lowcost=np.sum(costf*m)

    #keep track of starting configuration
    m_start=m.copy()
    starta=np.arange(n)
    #reorder
    for h in range(H):
        a=np.arange(n)
        #choose two positions and flip them
        r1,r2=np.random.randint(n,size=(2,))
        a[r1]=r2
        a[r2]=r1
        costnew=np.sum((m[np.ix_(a,a)])*costf)
        #if this reduced the overall cost
        if costnew<lowcost:
            m=m[np.ix_(a,a)]
            r2_swap=starta[r2]
            r1_swap=starta[r1]
            starta[r1]=r2_swap
            starta[r2]=r1_swap
            lowcost=costnew

    M_reordered=m_start[np.ix_(starta,starta)]
    m_indices=starta
    cost=lowcost
    return M_reordered,m_indices,cost

def reorder_matrix(m1,cost='line',verbose=False,H=1e4,Texp=10,T0=1e-3,Hbrk=10):
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
    from scipy import linalg,stats
    n=len(m1)
    if n<2:
        raise BCTParamError("align_matrix will infinite loop on a singleton "
            "or null matrix.")

    #generate cost function
    if cost=='line':
        profile=stats.norm.pdf(list(range(1,n+1)),loc=0,scale=n/2)[::-1]
    elif cost=='circ':
        profile=stats.norm.pdf(list(range(1,n+1)),loc=n/2,scale=n/4)[::-1]
    else:
        raise BCTParamError('cost must be line or circ')

    costf=linalg.toeplitz(profile,r=profile)*np.logical_not(np.eye(n))
    costf/=np.sum(costf)

    #establish maxcost, lowcost, mincost
    maxcost=np.sum(np.sort(costf.flat)*np.sort(m1.flat))
    lowcost=np.sum(m1*costf)/maxcost
    mincost=lowcost

    #initialize
    anew=np.arange(n)
    amin=np.arange(n)
    h=0; hcnt=0

    #adjust annealing parameters
    #H determines the maximal number of steps (user specified)
    #Texp determines the steepness of the temperature gradient
    Texp=1-Texp/H	
    #T0 sets the initial temperature and scales the energy term (user provided)
    #Hbrk sets a break point for the stimulation
    Hbrk=H/Hbrk

    while h<H:
        h+=1; hcnt+=1
        #terminate if no new mincost has been found for some time
        if hcnt>Hbrk:
            break
        T=T0*Texp**h
        atmp=anew.copy()
        r1,r2=np.random.randint(n,size=(2,))
        while r1==r2:
            r2=np.random.randint(n)
        atmp[r1]=anew[r2]
        atmp[r2]=anew[r1]	
        costnew=np.sum((m1[np.ix_(atmp,atmp)])*costf)/maxcost
        #annealing
        if costnew<lowcost or np.random.random()<np.exp(-(costnew-lowcost)/T):
            anew=atmp
            lowcost=costnew
            #is this a new absolute best?
            if lowcost<mincost:
                amin=anew
                mincost=lowcost
                if verbose:
                    print(('step %i ... current lowest cost = %f' % (h,mincost)))
                hcnt=0
    
    if verbose:
        print(('step %i ... final lowest cost = %f' % (h,mincost)))

    M_reordered=m1[np.ix_(amin,amin)]
    M_indices=amin
    cost=mincost
    return M_reordered,M_indices,cost

def reorder_mod(A,ci):
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
    #TODO update function with 2015 changes

    from scipy import stats
    _,max_module_size=stats.mode(ci)
    u,ci=np.unique(ci,return_inverse=True)	#make consecutive
    n=np.size(ci)							#number of nodes
    m=np.size(u)							#number of modules

    nm=np.zeros((m,))						#number of nodes in modules
    knm=np.zeros((n,m))						#degree to other modules

    for i in range(m):
        nm[i]=np.size(np.where(ci==i))
        knm[:,i]=np.sum(A[:,ci==i],axis=1)

    am=np.zeros((m,m))						#relative intermodular connectivity
    for i in range(m):
        am[i,:]=np.sum(knm[ci==i,:],axis=0)
    am/=np.outer(nm,nm)

    #1. Arrange densely connected modules together
    i,j=np.where(np.tril(am,-1)+1)		#symmetrized intermodular connectivity
    s=(np.tril(am,-1)+1)[i,j]
    ord=np.argsort(s)[::-1]		#sort by high relative connectivity
    i=i[ord]; j=j[ord]
    i+=1; j+=1							#fix off by 1 error so np.where doesnt
    om=np.array((i[0],j[0]))			#catch module 0
    i[0]=0; j[0]=0
    while len(om)<m:						#while not all modules ordered
        ui,=np.where(np.logical_and(i,np.logical_or(j==om[0],j==om[-1])))
        uj,=np.where(np.logical_and(j,np.logical_or(i==om[0],i==om[-1])))

        if np.size(ui): ui=ui[0]
        if np.size(uj): uj=uj[0]

        if ui==uj:
            i[ui]=0; j[uj]=0;
        if not np.size(ui): ui=np.inf
        if not np.size(uj): uj=np.inf
        if ui<uj:
            old=j[ui]; new=i[ui]
        if uj<ui:
            old=i[uj]; new=j[uj]
        if old==om[0]:
            om=np.append((new,),om)
        if old==om[-1]:
            om=np.append(om,(new,))

        i[i==old]=0; j[j==old]=0

    print(om)

    #2. Reorder nodes within modules
    on=np.zeros((n,),dtype=int)
    for y,x in enumerate(om):
        ind,=np.where(ci==x-1)				#indices
        pos,=np.where(om==x)				#position
        #NOT DONE! OE NOES

        mod_imp=np.array((om,np.sign(np.arange(m)-pos),
            np.abs(np.arange(m)-pos),am[x-1,om-1])).T
        print((np.shape((mod_imp[:,3][::-1],mod_imp[:,2]))))
        ix=np.lexsort((mod_imp[:,3][::-1],mod_imp[:,2]))
        mod_imp=mod_imp[ix]
        #at this point mod_imp agrees with the matlab version
        signs=mod_imp[:,1]
        mod_imp=np.abs(mod_imp[:,0]*mod_imp[:,1])
        mod_imp=np.append(mod_imp[1:],x)
        mod_imp=np.array(mod_imp-1,dtype=int)
        print((mod_imp,signs))
        #at this point mod_imp is the absolute value of that in the matlab
        #version.  this limitation comes from sortrows ability to deal with
        #negative indices, which we would have to do manually.

        #instead, i punt on its importance; i only bother to order by the
        #principal dimension.  some within-module orderings
        #may potentially be a little bit out of order.   

        #ksmi=knm[ind,:].T[mod_imp[::-1]]
        #reverse mod_imp to sort by the first column first and so on
        #print ksmi
        #for i,sin in enumerate(signs):
        #	if sin==-1:
        #		ksmi[i,:]=ksmi[i,:][::-1]
        #print ksmi
        #print np.shape(ksmi)

        # ^ this is unworkable and wrong, lexsort alone cannot handle the
        #negative indices problem of sortrows.  you would pretty much need
        #to rewrite sortrows to do lexsort plus negative indices; the algorithm
        #cant be further simplified.

        ord=np.lexsort(knm[np.ix_(ind,mod_imp[::-1])])
        #ord=np.lexsort(knm[ind,:].T[mod_imp[::-1]])
        if signs[mod_imp[0]]==-1:
            ord=ord[::-1]
            #reverse just the principal level and punt on the other levels.
            #this will basically be fine for most purposes and probably won't
            #ever show a difference for weighted graphs.
        on[ind[ord]]=y*int(max_module_size)+np.arange(nm[x-1],dtype=int)
        
    on=np.argsort(on)
    ar=A[np.ix_(on,on)]

    return on,ar

def writetoPAJ(CIJ,fname,directed):
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
    n=np.size(CIJ,axis=0)
    with open(fname,'w') as fd:
        fd.write('*vertices %i \r' % n)
        for i in range(1,n+1): fd.write('%i "%i" \r' % (i,i))
        if directed: fd.write('*arcs \r')
        else: fd.write('*edges \r')
        for i in range(n):
            for j in range(n):
                if CIJ[i,j]!=0:
                    fd.write('%i %i %.6f \r' % (i+1,j+1,CIJ[i,j]))

#############################################################################
# MISCELLANEOUS UTILITIES
#############################################################################
def _cuberoot(x):
    '''
    Correctly handle the cube root for negative weights, instead of uselessly
    crashing as in python or returning the wrong root as in matlab
    '''
    return np.sign(x)*np.abs(x)**(1/3)

def dummyvar(cis, return_sparse=False):
    '''
    This is an efficient implementation of matlab's "dummyvar" command
    using sparse matrices.

    input: partitions, NxM array-like containing M partitions of N nodes
        into <=N distinct communities

    output: dummyvar, an NxR matrix containing R column variables (indicator
        variables) with N entries, where R is the total number of communities
        summed across each of the M partitions.

        i.e.
        r = sum((max(len(unique(partitions[i]))) for i in range(m)))
    '''
    #num_rows is not affected by partition indexes
    n=np.size(cis,axis=0)
    m=np.size(cis,axis=1)
    r=np.sum((np.max(len(np.unique(cis[:,i])))) for i in range(m))
    nnz=np.prod(cis.shape)

    ix=np.argsort(cis,axis=0)
    #s_cis=np.sort(cis,axis=0)
    #FIXME use the sorted indices to sort by row efficiently
    s_cis=cis[ix][:,list(range(m)),list(range(m))]

    mask=np.hstack((((True,),)*m,(s_cis[:-1,:]!=s_cis[1:,:]).T))
    indptr,=np.where(mask.flat)
    indptr=np.append(indptr,nnz)

    import scipy.sparse as sp
    dv=sp.csc_matrix((np.repeat((1,),nnz),ix.T.flat,indptr),shape=(n,r))
    return dv.toarray()
