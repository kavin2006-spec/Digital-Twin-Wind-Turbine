import scipy.io
import numpy as np

filepath = r"data\raw\fraunhofer_lbf\Healthy\Healthy_2023_11_08_114335.mat"

mat = scipy.io.loadmat(filepath, simplify_cells=True)
signal = mat["brng_f_x"].flatten().astype(np.float32)
fs = 74000

print("Testing different window sizes and approaches:\n")

# Test 1: raw kurtosis at different window sizes
for win_sec in [0.01, 0.1, 0.5, 1.0]:
    n = int(fs * win_sec)
    wins = [signal[i*n:(i+1)*n] for i in range(min(10, len(signal)//n))]
    kurts = []
    for w in wins:
        m, s = w.mean(), w.std()
        if s > 0:
            kurts.append(float(np.mean(((w-m)/s)**4)))
    if kurts:
        print(f"  Raw {win_sec:.2f}s window: mean={np.mean(kurts):.2f} std={np.std(kurts):.2f}")

# Test 2: envelope kurtosis (abs of signal, then kurtosis)
print()
for win_sec in [0.01, 0.1, 1.0]:
    n = int(fs * win_sec)
    wins = [np.abs(signal[i*n:(i+1)*n]) for i in range(min(10, len(signal)//n))]
    kurts = []
    for w in wins:
        m, s = w.mean(), w.std()
        if s > 0:
            kurts.append(float(np.mean(((w-m)/s)**4)))
    if kurts:
        print(f"  Envelope {win_sec:.2f}s window: mean={np.mean(kurts):.2f} std={np.std(kurts):.2f}")

# Test 3: squared signal kurtosis (energy envelope)  
print()
for win_sec in [0.1, 1.0]:
    n = int(fs * win_sec)
    wins = [signal[i*n:(i+1)*n]**2 for i in range(min(10, len(signal)//n))]
    kurts = []
    for w in wins:
        m, s = w.mean(), w.std()
        if s > 0:
            kurts.append(float(np.mean(((w-m)/s)**4)))
    if kurts:
        print(f"  Squared {win_sec:.2f}s window: mean={np.mean(kurts):.2f} std={np.std(kurts):.2f}")

