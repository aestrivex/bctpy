import numpy as np
import bct
from scipy import io

def _load_sample():
	return bct.threshold_proportional(np.load('mats/sample_data.npy'), .4)

def test_modularity_und():
	x = _load_sample()
	ci,q = bct.modularity_und(x)
	print q
	assert np.allclose(q, 0.24253272, atol=0.0025)
	#matlab and bctpy appear to return different results due to the cross-
	#package numerical instability of eigendecompositions

def test_modularity_louvain_und():
	#this algorithm depends on a random seed so we will run 100 times and make
	#sure the modularities are relatively close each time
	#
	#performance is very similar to matlab
	x = _load_sample()
	fails = 0
	for i in xrange(100):
		ci,q = bct.modularity_louvain_und(x)
		try:
			assert np.allclose(q, .25, atol=0.01)
		except AssertionError:
			if fails>=5: raise
			else: fails+=1

def test_modularity_finetune_und():
	x = _load_sample()
	fails = 0
	for i in xrange(100):
		ci,q = bct.modularity_finetune_und(x)
		try:
			assert np.allclose(q, .25, atol=0.03)
		except AssertionError:
			if fails>=5: raise
			else: fails+=1

def test_modularity_finetune_und_actually_finetune():
	x = _load_sample()
	oci,oq = bct.modularity_und(x)
	for i in xrange(100):
		ci,q = bct.modularity_finetune_und(x,ci=oci)
		assert np.allclose(q, .25, atol=0.002)

	#modularity_finetune_und appears to be very stable when given a stable ci
	#in thousands of test runs on the sample data, only two states appeared; 
	#both improved the optimal modularity. a basic increase -- modules that
	#always benefited from switching -- always occurred. on top of that, a
	#slightly larger increase dependent on order occurred in both matlab and
	#bctpy around ~0.6% of the time. Due to numerical instability arising
	#from something different between matlab and scipy, these values were not 
	#the same across languages, but both languages showed bistable transitions.
	#they were extremely stable. The values were about .0015 apart.

	#also the matlab and python versions of modularity_und return slightly
	#different modular structure, but the instability is present despite this
	#(i.e. it is unstable both when the modular structure is identical and not)

def _load_signed_sample():
	return np.load('sample_signed.npy')

def test_modularity_louvain_und_sign():
	# performance is not that close to matlab
	x = _load_signed_sample()
