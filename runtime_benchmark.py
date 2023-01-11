import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
import json
import os

import cppimport.import_hook
from cpp_ext.als_module import Sampler

# Benchmarks the time to construct the sampler
# and samples from the Khatri-Rao product of
# randomly initialized matrices

sample_count = 100000

if __name__=='__main__':
    print("Hello world!")
