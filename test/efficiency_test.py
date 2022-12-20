from .load_samples import load_sample, load_directed_sample
import numpy as np
import bct

def test_diffusion_efficiency():
    x = load_sample(thres=.23)
    gde, ed = bct.diffusion_efficiency(x)
    print(gde, np.sum(ed)) 
    assert np.allclose(gde, .0069472)
    assert np.allclose(np.sum(ed), 131.34, atol=.01)

def test_resource_efficiency():
    x = load_sample(thres=.39)
    x = bct.binarize(x)

    eres, prob = bct.resource_efficiency_bin(x, .35)

    assert np.allclose(np.sum(eres), 323.5398, atol=.0001)
    assert np.allclose(np.sum(prob), 138.0000, atol=.0001)

def test_rout_efficiency():
    x = load_directed_sample(thres=1)
    GErout, Erout, Eloc = bct.rout_efficiency(x, 'inv')

    assert np.allclose(np.sum(Erout), 9515.25, atol=.01)
    assert np.allclose(GErout, 1.0655, atol=.0001)
    assert np.allclose(np.sum(Eloc), 2906.574, atol=.001)
    
