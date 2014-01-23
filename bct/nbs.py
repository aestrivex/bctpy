from __future__ import division
import numpy as np

from bct import BCTParamError

def nbs_bct(x,y,alpha=.05,k=1000,tail='both'):
	'''
      PVAL = NBS(X,Y,THRESH) 
	  Performs the NBS for populations X and Y for a t-statistic threshold of
	  alpha.

      inputs: x,y,	matrices representing the two populations being compared.
					x and y are of size NxNxP, where N is the number of nodes 
					in the network and P is the number of subjects within the 
					population.  P need not be equal for both X and Y.  
					X[i,j,k] stores the connectivity value	corresponding to 
					the edge between i and j for the kth member of the
					population.

		IS IT DIRECTED?  CAN X AND Y HAVE DIFFERENT SAMPLE SIZES?

			alpha,	the type-I significance threshold (default .05)
			    k,	the number of permutations to be generated to estimate the
					empirical null distribution (default 1000)
			 tail,	enables specification of the type of alternative hypothesis
					to test.
						'left': mean of population X < mean of population Y
						'right': mean of population Y < mean of population X
						'both': means are unequal (default)
	  outputs:
			  pval,	a vector of corrected p-values for each component of the
					network that is identified.  If at least one p-value is
					less than alpha, then the omnibus null hypothesis can be
					rejected at alpha significance. The null hypothesis is that
					the value of connectivity at each edge comes from
					distributions of equal mean between the two populations.
			  adj,	an adjacency matrix identifying the edges comprising each
					component.  Edges are assigned indexed values.
			  null,	A vector of k samples from the null distribution of maximal
					component size

      ALGORITHM DESCRIPTION 
      The NBS is a nonparametric statistical test used to isolate the 
      components of an N x N undirected connectivity matrix that differ 
      significantly between two distinct populations. Each element of the 
      connectivity matrix stores a connectivity value and each member of 
      the two populations possesses a distinct connectivity matrix. A 
      component of a connectivity matrix is defined as a set of 
      interconnected edges. 
 
      The NBS is essentially a procedure to control the family-wise error 
      rate, in the weak sense, when the null hypothesis is tested 
      independently at each of the N(N-1)/2 edges comprising the undirected
      connectivity matrix. The NBS can provide greater statistical power 
      than conventional procedures for controlling the family-wise error 
      rate, such as the false discovery rate, if the set of edges at which
      the null hypothesis is rejected constitues a large component or
      components.
      The NBS comprises fours steps:
      1. Perform a two-sample T-test at each edge indepedently to test the
         hypothesis that the value of connectivity between the two
         populations come from distributions with equal means. 
      2. Threshold the T-statistic available at each edge to form a set of
         suprathreshold edges. 
      3. Identify any components in the adjacency matrix defined by the set
         of suprathreshold edges. These are referred to as observed 
         components. Compute the size of each observed component 
         identified; that is, the number of edges it comprises. 
      4. Repeat K times steps 1-3, each time randomly permuting members of
         the two populations and storing the size of the largest component 
         identified for each permuation. This yields an empirical estimate
         of the null distribution of maximal component size. A corrected 
         p-value for each observed component is then calculated using this
         null distribution.
 
      [1] Zalesky A, Fornito A, Bullmore ET (2010) Network-based statistic:
          Identifying differences in brain networks. NeuroImage.
          10.1016/j.neuroimage.2010.06.041
	'''
