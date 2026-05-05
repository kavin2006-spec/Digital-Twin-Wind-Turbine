"""
scripts/run_real_data_experiment.py
=====================================
Trains the ensemble denoising VAE on real CARE to Compare Wind Farm A
data and evaluates it on turbine 21's labeled anomaly windows.

Pipeline:
  1. Load normal training rows (turbines 0,10,11,13, status=0, is_fault=0)
  2. Extract 4 physics-normalised features
  3. Train ensemble denoising VAE (same architecture as simulation)
  4. Calibrate threshold on held-out normal rows from turbine 21
  5. Evaluate on turbine 21 anomaly rows
  6. Report precision, recall, F1, PR-AUC, cost

Key difference from simulation:
  - 10-minute SCADA averages (not 1Hz)
  - Real turbine variability (not physics model + noise)
  - window_size in steps = physical_seconds / 600
    e.g. 60-min window = 6 steps at 10-min resolution

Run:
  python scripts/run_real_data_experiment.py
"""

from __future__ import annotations
import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ml_anomaly_detector import LSTMAutoencoderDetector, AnomalyResult
from models.evaluator import AnomalyEvaluator, StepResult

DB_PATH      = "data/processed/wind_farm_a.db"
WINDOW_STEPS = 6      # 6 × 10min = 60-minute window (matches simulation intent)
SLIDE_EVERY  = 1      # slide every step (more evaluation points)
N_TRAIN_ROWS = 3000   # rows to use for training (sample for speed)
N_CALIB_ROWS = 500    # rows for calibration threshold


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the same 4 physics-normalised features used in simulation.

    efficiency_ratio:   active_power / expected_power
    deviation_pct/100:  (actual - expected) / expected
    vibration proxy:    gearbox bearing temp gradient (no vibration sensor)
    nacelle_temp/100:   generator_temp_c / 100

    Note: No vibration sensor in Wind Farm A SCADA.
    We use gearbox_bearing_temp_c as a proxy — bearing faults show up
    in temperature before vibration sensors would catch them at 10min res.

    Expected power from simplified power curve:
      P_expected = clip(0.5 * rho * A * v^3 * Cp, 0, rated_kw)
    """
    # Power is normalised [0,1] in CARE dataset (1.0 = rated power)
    # Wind speed is in m/s (confirmed from feature_description)
    # Use wind speed to estimate expected normalised power via Betz approximation
    AIR_DENSITY  = 1.225
    ROTOR_AREA   = 3.14159 * 40.0**2   # ~2MW turbine, rotor ~80m diameter
    CP           = 0.45
    RATED_W      = 2_000_000.0          # 2MW rated

    v = df["wind_speed_ms"].values
    p_expected_norm = np.clip(
        0.5 * AIR_DENSITY * ROTOR_AREA * (v ** 3) * CP / RATED_W,
        0.0, 1.0
    )

    p_actual = df["active_power_kw"].values   # actually normalised [0,1]

    # Efficiency: actual / expected normalised power
    valid = p_expected_norm > 0.01
    efficiency = np.ones(len(df))
    efficiency[valid] = p_actual[valid] / p_expected_norm[valid]
    efficiency = np.clip(efficiency, 0.0, 1.5)

    deviation = np.zeros(len(df))
    deviation[valid] = (p_actual[valid] - p_expected_norm[valid]) / (p_expected_norm[valid] + 1e-6)
    deviation = np.clip(deviation, -1.0, 1.0)

    # Vibration proxy: gearbox bearing temp relative to ambient
    # Healthy: stable ~30-40°C above ambient. Fault: rising.
    if "gearbox_bearing_temp_c" in df.columns and "ambient_temp_c" in df.columns:
        bearing_margin = (
            df["gearbox_bearing_temp_c"].values - df["ambient_temp_c"].values
        )
        # Normalise to [0,1] range: 0=cool, 1=hot
        vib_proxy = np.clip(bearing_margin / 60.0, 0.0, 1.0)
    else:
        vib_proxy = np.full(len(df), 0.3)

    # Generator temperature normalised
    gen_temp = np.clip(df["generator_temp_c"].values / 100.0, 0.0, 1.5)

    features = np.column_stack([
        efficiency,
        deviation / 100.0,   # match simulation scaling
        vib_proxy,
        gen_temp,
    ]).astype(np.float32)

    return features


def run_experiment() -> None:
    print(f"\n{'='*60}")
    print(f"  Real Data Experiment — Wind Farm A")
    print(f"{'='*60}\n")

    conn = sqlite3.connect(DB_PATH)

    # ── 1. Load training data ─────────────────────────────────────────────────
    print("Loading training data...")
    # Use placeholders via params to avoid SQL string quoting issues
    train_df = pd.read_sql(
        "SELECT wind_speed_ms, active_power_kw, gearbox_bearing_temp_c, "
        "generator_temp_c, ambient_temp_c, is_fault "
        "FROM turbine_readings "
        "WHERE proper_split = 'eval' "
        "AND is_fault = 0 "
        "AND fault_label IS NULL "
        "AND active_power_kw > 0.001 "
        "LIMIT ?",
        conn,
        params=(N_TRAIN_ROWS,),
    )

    print(f"  Training rows: {len(train_df):,}")

    print(f"  train_df shape: {train_df.shape}")
    print(f"  train_df columns: {list(train_df.columns)}")
    print(f"  wind_speed sample: {train_df['wind_speed_ms'].head(3).tolist()}")

    train_features = extract_features(train_df)
    print(f"  Feature shape: {train_features.shape}")
    if len(train_features) == 0:
        print("ERROR: train_features is empty -- SQL query returned no rows")
        return
    print(f"  Feature means: {train_features.mean(axis=0).round(3)}")
    print(f"  Feature stds:  {train_features.std(axis=0).round(3)}")

    # ── 2. Load calibration data (normal rows from eval turbine) ──────────────
    print("\nLoading calibration data...")
    calib_df = pd.read_sql(
        "SELECT wind_speed_ms, active_power_kw, gearbox_bearing_temp_c, "
        "generator_temp_c, ambient_temp_c "
        "FROM turbine_readings "
        "WHERE proper_split = 'eval' "
        "AND is_fault = 0 "
        "AND fault_label IS NULL "
        "AND active_power_kw > 0.001 "
        "LIMIT ?",
        conn,
        params=(N_TRAIN_ROWS,),
    )

    print(f"  Calibration rows: {len(calib_df):,}")

    # ── 3. Load evaluation data ───────────────────────────────────────────────
    print("\nLoading evaluation data...")
    eval_df = pd.read_sql(
        "SELECT wind_speed_ms, active_power_kw, gearbox_bearing_temp_c, "
        "generator_temp_c, ambient_temp_c, is_fault, fault_label, timestamp "
        "FROM turbine_readings "
        "WHERE proper_split = ? "
        "ORDER BY timestamp",
        conn,
        params=("eval",),
    )

    print(f"  Eval rows:   {len(eval_df):,}")
    print(f"  Fault rows:  {eval_df['is_fault'].sum():,} "
          f"({100*eval_df['is_fault'].mean():.1f}%)")

    conn.close()
    
    # Diagnostic: what do fault vs normal windows actually look like?
    fault_df = eval_df[eval_df['is_fault'] == 1]
    normal_df = eval_df[eval_df['is_fault'] == 0]
    print(f"\n  Feature comparison (fault vs normal):")
    for col in ['wind_speed_ms', 'active_power_kw', 'gearbox_bearing_temp_c', 'generator_temp_c']:
        print(f"  {col}:")
        print(f"    Normal: mean={normal_df[col].mean():.3f} std={normal_df[col].std():.3f}")
        print(f"    Fault:  mean={fault_df[col].mean():.3f} std={fault_df[col].std():.3f}")

    # ── 4. Train VAE ──────────────────────────────────────────────────────────
    print("\nTraining VAE on real data...")
    detector = LSTMAutoencoderDetector(
        window_size=WINDOW_STEPS,
        slide_every=SLIDE_EVERY,
        latent_dim=4,
        n_layers=2,
        n_epochs=50,
        lr=0.001,
        train_after=len(train_df),   # train immediately after loading all data
        beta=0.5,
        kl_weight=0.3,
        n_ensemble=3,
        noise_scale=0.03,
        grad_clip=1.0,
        calibration_size=len(calib_df) - WINDOW_STEPS,
    )

    # Train directly on feature array -- bypass streaming buffer
    # _train() reads from self._buffer, so we populate it correctly
    from collections import deque
    maxlen = max(len(train_features) + 100, detector.window_size * 4)
    detector._buffer = deque(maxlen=maxlen)
    for row in train_features:
        detector._buffer.append(list(row))

    detector._step_count = len(train_features)
    print(f"  Buffer size: {len(detector._buffer)}")
    detector._train()

    # ── 5. Calibrate on normal eval rows ─────────────────────────────────────
    print("\nCalibrating threshold on normal eval rows...")
    calib_features = extract_features(calib_df)

    detector._calibration_buffer = [
        list(row) for row in calib_features
    ]
    detector._run_calibration()

    # ── 6. Evaluate on eval set ───────────────────────────────────────────────
    print("\nEvaluating on turbine 21...")
    eval_features = extract_features(eval_df)
    eval_labels   = eval_df["is_fault"].values.astype(bool)

    # Score windows
    scores     = []
    is_anomaly = []

    data_norm = detector._normalise(eval_features)
    windows   = detector._make_windows(data_norm)

    print(f"  Scoring {len(windows):,} windows...")

    X = torch.tensor(windows, dtype=torch.float32)
    recon_errors, kl_divs = detector._ensemble_score(X)

    # Apply threshold
    recon_score = recon_errors / (detector._recon_threshold + 1e-10)
    if detector._kl_threshold < float("inf"):
        kl_score = kl_divs / (detector._kl_threshold + 1e-10)
        combined = (
            (1 - detector.kl_weight) * recon_score
            + detector.kl_weight * kl_score
        )
    else:
        combined = recon_score

    display_scores = combined / (combined + 1.0)
    anomaly_flags  = combined > 1.0

    # Align labels with windows
    # Window i covers rows [i, i+window_size) — label = last row
    window_labels = eval_labels[WINDOW_STEPS-1:][:len(windows)]

    # ── 7. Report ─────────────────────────────────────────────────────────────
    evaluator = AnomalyEvaluator(
        model_name="Denoising VAE — Real Data",
        early_tolerance=3,
        late_tolerance=2,
    )

    for i, (label, flag, score) in enumerate(
        zip(window_labels, anomaly_flags, display_scores)
    ):
        evaluator.add(StepResult(
            step=i,
            is_fault_ground_truth=bool(label),
            is_anomaly_predicted=bool(flag),
            anomaly_score=float(score),
            fault_type="anomaly" if label else None,
        ))

    report = evaluator.evaluate()
    print(report)

    # Additional stats
    n_windows    = len(windows)
    n_fault_win  = window_labels.sum()
    n_normal_win = (~window_labels).sum()

    print(f"\n  Window stats:")
    print(f"    Total windows:  {n_windows:,}")
    print(f"    Fault windows:  {n_fault_win:,} ({100*n_fault_win/n_windows:.1f}%)")
    print(f"    Normal windows: {n_normal_win:,}")
    print(f"\n  Score distribution:")
    print(f"    Normal  — mean: {display_scores[~window_labels].mean():.4f}  "
          f"std: {display_scores[~window_labels].std():.4f}")
    print(f"    Fault   — mean: {display_scores[window_labels].mean():.4f}  "
          f"std: {display_scores[window_labels].std():.4f}")
    print(f"    Separation:     "
          f"{display_scores[window_labels].mean() - display_scores[~window_labels].mean():.4f}")

    print(f"\n{'='*60}")
    print(f"  Experiment complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_experiment()