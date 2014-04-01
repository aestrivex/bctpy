import numpy as np
import bct

def _load_sample():
	return bct.threshold_proportional(np.load('mats/sample_data.npy'), .4)

def test_modularity_und():
	x = _load_sample()
	ci,q = bct.modularity_und(x)
	assert np.allclose(q, 0.24253272, atol=0.05)
	# not sure why matlab and bctpy return different modular structures --
	# they are close but nonidentical

def test_modularity_louvain_und():
	x = _load_sample()
	#this algorithm depends on a random seed so we will run 100 times and make
	#sure the modularities are relatively close each time
	for i in xrange(100):
		ci,q = bct.modularity_louvain_und(x)
		assert np.allclose(q, .25578, atol=0.1)
