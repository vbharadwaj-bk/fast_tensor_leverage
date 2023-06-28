import numpy as np
from dense_tensor import *

import os

def get_amrex_single_plot_tensor(filepath, field='charge'):
    print("Loading yt...")
    import yt
    from yt.frontends.boxlib.data_structures import AMReXDataset
    print("yt loaded!")

    ds = AMReXDataset(plot_path)
    ad0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    charge_numpy_array = ad0['charge'].to_ndarray()

    print(f"Loaded AMREX plot {filepath}...") 
    tensor = PyDenseTensor(charge_numpy_array)
    print("Initialized dense tensor...")

    return tensor