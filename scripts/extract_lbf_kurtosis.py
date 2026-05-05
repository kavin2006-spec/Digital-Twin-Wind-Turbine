"""
scripts/extract_lbf_kurtosis.py
================================
Extracts vibration kurtosis from Fraunhofer LBF .mat files.

Produces a CSV with one row per 1-second window:
  timestamp, kurtosis, crest_factor, rms, peak, label, fault_type

This CSV is then used to:
  1. Validate our VibrationSimulator kurtosis values
  2. Replace simulated kurtosis with real kurtosis in the VAE feature set
  3. Characterise the bearing fault kurtosis distribution

Signal used: brng_f_x (front bearing, x-axis)
  - 74kHz sampling rate
  - 74,000 samples per 1-second window
  - One kurtosis value per window -> 1Hz feature cadence

Run:
  python scripts/extract_lbf_kurtosis.py

Output:
  data/processed/lbf_kurtosis.csv

💡 Industrial insight: Front bearing (brng_f) is the correct sensor for
detecting main bearing faults in this turbine. The front bearing is on
the rotor shaft side -- it carries the full thrust and radial load from
the wind. Bearing faults typically manifest first here. SKF's WinCM
platform uses the same selection criteria for sensor prioritisation.
"""

from __future__ import annotations
import os
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT   = Path("data/raw/fraunhofer_lbf")
OUTPUT_CSV  = Path("data/processed/lbf_kurtosis.csv")
WINDOW_SEC  = 1.0       # 1-second windows -> 1Hz feature cadence
SENSOR_KEY  = "brng_f_x"   # Front bearing, x-axis (highest fault sensitivity)
BACKUP_KEYS = ["brng_f_y", "brng_f_z", "brng_r_x"]  # Fallbacks

# Fault type mapping from folder name
FAULT_MAP = {
    "Healthy":                  "healthy",
    "InnerRace":                "bearing_inner_race",
    "InnerRace_MassImbalance":  "bearing_inner_race_imbalance",
    "OutterRace":               "bearing_outer_race",
    "RollerElement":            "bearing_roller_element",
    "Imbalance_High":           "mass_imbalance_high",
    "Imbalance_Mid":            "mass_imbalance_mid",
    "Imbalance_Low":            "mass_imbalance_low",
    "Aerodynamic_5_Degrees":    "aerodynamic_imbalance_5deg",
    "Aerodynamic_12_Degrees":   "aerodynamic_imbalance_12deg",
    "Aerodynamic_15_Degrees":   "aerodynamic_imbalance_15deg",
}


def compute_kurtosis(signal: np.ndarray) -> float:
    """
    Statistical kurtosis of a signal window.
    kurtosis = E[(x-mu)^4] / sigma^4
    Healthy bearing: ~3.0 (Gaussian)
    Bearing fault:   >> 3.0 (impulsive spikes)
    """
    mean = signal.mean()
    std  = signal.std()
    if std < 1e-10:
        return 0.0
    return float(np.mean(((signal - mean) / std) ** 4))


def compute_crest_factor(signal: np.ndarray) -> float:
    """Peak / RMS -- another bearing fault indicator."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-10:
        return 0.0
    return float(np.max(np.abs(signal)) / rms)


def extract_from_file(
    filepath:   str,
    fault_type: str,
    label:      int,    # 0=healthy, 1=fault
) -> list[dict]:
    """
    Extract 1-second kurtosis windows from one .mat file.

    Returns list of dicts, one per window.
    """
    try:
        mat = scipy.io.loadmat(filepath, simplify_cells=True)
    except Exception as e:
        print(f"  Failed to load {filepath}: {e}")
        return []

    # Get sampling rate from file
    fs = float(mat.get("Sample_rate", 74000))
    if hasattr(fs, '__len__'):
        fs = float(fs.flat[0])

    # Get bearing signal -- try primary then fallbacks
    signal = None
    used_key = None
    for key in [SENSOR_KEY] + BACKUP_KEYS:
        if key in mat:
            raw = mat[key]
            if hasattr(raw, 'flatten'):
                signal = raw.flatten().astype(np.float32)
                used_key = key
                break

    if signal is None:
        print(f"  No bearing signal found in {filepath}")
        return []

    # Get temperature if available
    tmp_brng = mat.get("tmp_brng_f", None)
    if tmp_brng is not None and hasattr(tmp_brng, 'flatten'):
        tmp_brng = tmp_brng.flatten()
    else:
        tmp_brng = None

    tmp_amb = mat.get("tmp_amb", None)
    if tmp_amb is not None and hasattr(tmp_amb, 'flatten'):
        tmp_amb = tmp_amb.flatten()
    else:
        tmp_amb = None

    # Extract windows
    window_samples = int(fs * WINDOW_SEC)
    n_windows      = len(signal) // window_samples
    rows           = []

    for i in range(n_windows):
        start = i * window_samples
        end   = start + window_samples
        win   = signal[start:end]

        kurtosis     = compute_kurtosis(win)
        crest_factor = compute_crest_factor(win)
        rms          = float(np.sqrt(np.mean(win ** 2)))
        peak         = float(np.max(np.abs(win)))

        # Downsample temperature to match window index
        # tmp arrays are at ~1.48kHz -- find closest sample
        brng_temp = None
        amb_temp  = None
        if tmp_brng is not None and len(tmp_brng) > 0:
            tmp_idx  = min(
                int(i * len(tmp_brng) / n_windows),
                len(tmp_brng) - 1
            )
            brng_temp = float(tmp_brng[tmp_idx])
        if tmp_amb is not None and len(tmp_amb) > 0:
            tmp_idx  = min(
                int(i * len(tmp_amb) / n_windows),
                len(tmp_amb) - 1
            )
            amb_temp = float(tmp_amb[tmp_idx])

        rows.append({
            "window_idx":    i,
            "fault_type":    fault_type,
            "label":         label,      # 0=healthy, 1=fault
            "kurtosis":      round(kurtosis,     4),
            "crest_factor":  round(crest_factor, 4),
            "rms":           round(rms,          6),
            "peak":          round(peak,         6),
            "brng_temp_c":   round(brng_temp, 2) if brng_temp else None,
            "amb_temp_c":    round(amb_temp,  2) if amb_temp  else None,
            "sensor":        used_key,
            "source_file":   Path(filepath).name,
            "fs_hz":         fs,
        })

    return rows


def run_extraction() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Fraunhofer LBF Kurtosis Extraction")
    print(f"  Source: {DATA_ROOT}")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"{'='*55}\n")

    all_rows = []

    # Process each condition folder
    for folder_name, fault_type in FAULT_MAP.items():
        folder_path = DATA_ROOT / folder_name
        if not folder_path.exists():
            continue

        # Handle bearing subfolders
        mat_files = list(folder_path.glob("*.mat"))
        sub_folders = [d for d in folder_path.iterdir() if d.is_dir()]

        if sub_folders:
            # e.g. Bearing/ has InnerRace/, OutterRace/ subfolders
            for sub in sub_folders:
                sub_fault = FAULT_MAP.get(sub.name, sub.name.lower())
                sub_files = list(sub.glob("*.mat"))
                if not sub_files:
                    continue
                print(f"  {sub.name}: {len(sub_files)} files")
                for filepath in sub_files:
                    rows = extract_from_file(str(filepath), sub_fault, label=1)
                    all_rows.extend(rows)
        else:
            label = 0 if folder_name == "Healthy" else 1
            if not mat_files:
                continue
            print(f"  {folder_name}: {len(mat_files)} files")
            for filepath in mat_files:
                rows = extract_from_file(str(filepath), fault_type, label=label)
                all_rows.extend(rows)

    if not all_rows:
        print("ERROR: No data extracted. Check DATA_ROOT path.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    # Summary
    print(f"\n{'='*55}")
    print(f"  Extraction complete")
    print(f"  Total windows: {len(df):,}")
    print(f"\n  Per fault type:")
    summary = df.groupby("fault_type").agg(
        windows=("kurtosis", "count"),
        kurtosis_mean=("kurtosis", "mean"),
        kurtosis_std=("kurtosis", "std"),
        kurtosis_p95=("kurtosis", lambda x: x.quantile(0.95)),
    ).round(3)
    print(summary.to_string())

    # Key comparison: healthy vs bearing faults
    print(f"\n  Kurtosis healthy vs fault:")
    healthy = df[df["label"] == 0]["kurtosis"]
    faults  = df[df["label"] == 1]["kurtosis"]
    print(f"    Healthy: mean={healthy.mean():.3f}  "
          f"std={healthy.std():.3f}  "
          f"p95={healthy.quantile(0.95):.3f}")
    print(f"    Fault:   mean={faults.mean():.3f}  "
          f"std={faults.std():.3f}  "
          f"p95={faults.quantile(0.95):.3f}")
    print(f"    Separation ratio: {faults.mean()/max(healthy.mean(),0.001):.2f}x")

    print(f"\n  Saved to: {OUTPUT_CSV}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_extraction()
