import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from common import *

import cppimport.import_hook
from cpp_ext.tt_module import DenseTensor_float, DenseTensor_double 

class PyDenseTensor:
    def __init__(self, data):
        self.data = data
        self.data_norm = la.norm(data)
        if np.issubdtype(data.dtype, np.float32):
            self.ten = DenseTensor_float(data, 10000) 
        elif np.issubdtype(data.dtype, np.float64):
            self.ten = DenseTensor_double(data, 10000) 