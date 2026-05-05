import scipy.io
import numpy as np

# Change this to your actual healthy file path
filepath = r"data\raw\fraunhofer_lbf\Healthy\Healthy_2023_11_08_114335.mat"

mat = scipy.io.loadmat(filepath, simplify_cells=True)
signal = mat["brng_f_x"].flatten().astype(np.float32)

print(f"Signal length: {len(signal):,}")
print(f"Signal mean:   {signal.mean():.6f}  (should be near 0)")
print(f"Signal std:    {signal.std():.6f}")
print(f"Signal min:    {signal.min():.6f}")
print(f"Signal max:    {signal.max():.6f}")
print(f"First 5 values: {signal[:5]}")

# Compute kurtosis on first 74000 samples (1 second)
win = signal[:74000]
mean = win.mean()
std  = win.std()
kurt = np.mean(((win - mean) / std) ** 4)
print(f"\nFirst 1-second window kurtosis: {kurt:.3f}  (healthy should be ~3)")

# Try DC-removed
win_dc = win - win.mean()
std_dc = win_dc.std()
kurt_dc = np.mean(((win_dc) / std_dc) ** 4) if std_dc > 0 else 0
print(f"DC-removed kurtosis:            {kurt_dc:.3f}")