from load_samples import *
import numpy as np
import bct
from scipy import io

def test_modularity_und():
	x = load_sample(thres=.4)
	_,q = bct.modularity_und(x)
	assert np.allclose(q, 0.24253272, atol=0.0025)
	#matlab and bctpy appear to return different results due to the cross-
	#package numerical instability of eigendecompositions

def test_modularity_louvain_und():
	x = load_sample(thres=.4)

	seed = 38429004
	_,q = bct.modularity_louvain_und(x, seed=seed)
	assert np.allclose(q, 0.25892588)

	fails = 0
	for i in xrange(100):
		ci,q = bct.modularity_louvain_und(x)
		try:
			assert np.allclose(q, .25, atol=0.01)
		except AssertionError:
			if fails>=5: raise
			else: fails+=1

	seed = 94885236
	_,q = bct.modularity_finetune_und(x, seed=seed)
	assert np.allclose(q, .25879794)

def test_modularity_finetune_und():
	x = load_sample(thres=.4)

	seed = 94885236
	_,q = bct.modularity_finetune_und(x, seed=seed)
	assert np.allclose(q, .25879794)

	fails = 0
	for i in xrange(100):
		_,q = bct.modularity_finetune_und(x)
		try:
			assert np.allclose(q, .25, atol=0.03)
		except AssertionError:
			if fails>=5: raise
			else: fails+=1

	seed = 71040925
	ci,oq = bct.modularity_louvain_und(x,seed=seed)
	_,q = bct.modularity_finetune_und(x,ci=ci,seed=seed)
	assert np.allclose(q, .25892588)
	assert q >= oq

	ci,oq = bct.modularity_und(x)
	for i in xrange(100):
		_,q = bct.modularity_finetune_und(x,ci=ci)
		assert np.allclose(q, .25, atol=0.002)
		assert q >= oq

	#modularity_finetune_und appears to be very stable when given a stable ci
	#in thousands of test runs on the sample data (using the deterministic
	#modularity maximization algorithm), only two states appeared; 
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

def test_modularity_louvain_und_sign_seed():
	#performance is same as matlab if randomness is quashed
	x = load_signed_sample()
	seed = 90772777
	_,q = bct.modularity_louvain_und_sign(x, seed=seed)
	assert np.allclose(q, .48336787)

def test_modularity_finetune_und_sign_actually_finetune():
	x = load_signed_sample()
	seed = 34908314
	ci,oq = bct.modularity_louvain_und_sign(x, seed=seed)
	_,q = bct.modularity_finetune_und_sign(x, seed=seed, ci=ci)
	assert np.allclose(q, .48034182)
	assert q >= oq

	seed = 88215881
	np.random.seed(seed)
	randomized_sample = np.random.random(size=(len(x),len(x)))
	randomized_sample = randomized_sample + randomized_sample.T
	x[np.where(bct.threshold_proportional(randomized_sample, .2))] = 0

	ci,oq = bct.modularity_louvain_und_sign(x, seed=seed)
	#assert np.allclose(oq, .50225885)
	assert np.allclose(oq, .48013250)
	for i in xrange(100):
		_,q = bct.modularity_finetune_und_sign(x, ci=ci)
		assert q >= oq

def test_modularity_probtune_und_sign():
	x = load_signed_sample()
	seed = 59468096
	ci,q = bct.modularity_probtune_und_sign(x, seed=seed)
	assert np.allclose(q, .13322379)

	seed = 1742447
	ci,_ = bct.modularity_louvain_und_sign(x, seed=seed)
	_,oq = bct.modularity_finetune_und_sign(x, seed=seed, ci=ci)
	
	for i in np.arange(.05, .5, .02):
		fails=0
		for j in xrange(100):
			_,q = bct.modularity_probtune_und_sign(x, ci=ci, p=i)
			try:
				assert q < oq
			except AssertionError:
				if fails > 5: raise
				else: fails+=1

def test_modularity_dir():
	pass

def test_modularity_louvain_dir():
	pass

def test_modularity_finetune_dir():
	pass
