import os
import scipy.io
import numpy as np

# Point this at one file from Healthy
HEALTHY_FILE = r"data\raw\fraunhofer_lbf\Healthy\Healthy_2023_11_08_114335.mat"
try:
    mat = scipy.io.loadmat(HEALTHY_FILE)
    print("Keys:", [k for k in mat.keys() if not k.startswith('_')])
    for k, v in mat.items():
        if not k.startswith('_'):
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v)} = {v}")
except Exception as e:
    print(f"scipy failed: {e}")
    print("Trying h5py...")
    import h5py
    with h5py.File(HEALTHY_FILE, 'r') as f:
        def show(name, obj):
            if hasattr(obj, 'shape'):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(show)