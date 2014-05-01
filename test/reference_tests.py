import numpy as np
import bct
from scipy import io

def _load_sample():
	return bct.threshold_proportional(np.load('mats/sample_data.npy'), .35)

