import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
import sys

from common import *

current_script_path = os.path.dirname(os.path.abspath(__file__))
fast_tensor_leverage_path = os.path.dirname(current_script_path)
sys.path.append(fast_tensor_leverage_path)

import cppimport.import_hook
from cpp_ext.tt_module import TTSampler
from cpp_ext.tt_module import DenseTensor_float, DenseTensor_double

class PyDenseTensor:
    def __init__(self, data):
        self.data = data
        self.shape = list(data.shape)
        self.N = len(self.shape)
        self.data_norm = la.norm(data)
        if np.issubdtype(data.dtype, np.float32):
            self.ten = DenseTensor_float(data, 10000) 
        elif np.issubdtype(data.dtype, np.float64):
            self.ten = DenseTensor_double(data, 10000) 

    def execute_sampled_spmm(self, samples, design, j, result):
        self.ten.execute_downsampled_mttkrp(
                samples,
                design,
                j,
                result)
