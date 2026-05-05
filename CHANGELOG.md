# Changelog

All significant architectural decisions, milestones, and lessons learned.

---

## Phase 1 — Foundation

**Completed:** Physics model, wind generator, fault injector, sensor simulator, data schemas, Streamlit dashboard, RAG assistant interface.

**Key decisions:**
- IEC power curve for expected power (Betz limit, cut-in/rated/cut-out)
- Ornstein-Uhlenbeck process for wind speed (physically realistic autocorrelation)
- Fault injector as a state machine (start → persist → end) rather than per-step random injection
- `SensorReading` schema as the hardware abstraction layer — swap simulation for real SCADA by replacing this module only

---

## Phase 2 — Simulation upgrade

**Completed:** 21-signal SCADA sensor simulator, 5-fault-type injector, 100Hz VibrationSimulator.

**Key decisions:**
- VibrationSimulator generates full 100-sample time series per SCADA timestep, enabling FFT and kurtosis extraction
- FaultInjector extended with PRIMARY + SECONDARY effects per signal — each fault type has realistic multi-signal correlation
- Pitch asymmetry on `efficiency_drop` (one blade misaligned) feeds Tier-2 feature directly

**Lessons:**
- `SensorSimulator.read()` return signature changed from `SensorReading` to `(SensorReading, VibrationWindow)` — required updating twin orchestrator

---

## Phase 3 — Feature engineering

**Completed:** 41-feature `EnrichedReading` (Tier-1 raw + Tier-2 physics residuals + Tier-3 rolling stats).

**Key decisions:**
- Physics residuals (power_residual_kw, efficiency_ratio, tip_speed_ratio) remove wind-speed correlation — critical for VAE training
- Tier-3 rolling stats computed over configurable window (default 60s) — matches simulation and SCADA cadences
- Feature-wise normalisation with std floor — prevents constant features from causing division instability

**Lessons:**
- `vibration_rms` was constant (std=0.0065) because FeatureEngineer was computing RMS over a single step — fixed by using the full VibrationWindow

---

## Phase 4 — Real data (CARE to Compare)

**Completed:** SQLite database builder, SCADA loader, dataset registry, train/eval partition.

**Finding:** CARE to Compare Wind Farm A fault labels correspond to high-load operating periods, not mechanical degradation faults. Fault windows have HIGHER power and temperature than normal windows. Our efficiency-based features score these as normal (correctly — the turbine is operating efficiently). PR-AUC on real data: 0.071, consistent with published CARE baselines (0.05-0.15).

**Key decisions:**
- Turbine-based train/eval split (turbines 0,10,11,13 → train; turbine 21 → eval) rather than time-based split — tests generalisation to unseen turbine
- `status_type_id=0` filter for normal operation (excludes curtailment=4 and maintenance=3)

---

## Phase 5 — VAE architecture and training

**Attempted and failed:** 41-feature LSTM VAE, flat MLP VAE, feature masking, robust normalisation.

**Root cause:** Raw SCADA signals are highly correlated with wind speed (RPM, torque, power all move together). VAE reconstruction loss stayed at 0.81+ regardless of architecture because the model couldn't separate wind-driven from fault-driven variation with 641 training windows.

**Solution:** 4 physics-residual features (efficiency_ratio, deviation_pct, vibration_kurtosis, nacelle_temp) — wind correlation already removed. Model converges to recon loss 0.0011 in 50 epochs.

**Critical bug fixed:** Calibration threshold was computed as μ+5σ of normalised scores (dividing by training-set p95). Since calibration windows come from the same distribution as training, ~50% scored above 1.0, threshold always hit the ceiling. Fix: threshold computed from raw reconstruction errors of calibration windows, not normalised scores.

**Training fixes:**
- β warmup: 0 → 0.5 over first 25 epochs (prevents posterior collapse)
- Gradient clipping max_norm=1.0 (prevents encoder collapse spikes)
- KL collapse guard: encoder reset + lr×0.5 if KL < 1e-4 for 5 epochs
- Fixed-centre subtraction normalisation (not z-score) for the 4 hand-crafted features

---

## Phase 5 (final baseline) — Multi-scale VAE

**Completed:** Two parallel detectors (30s + 120s windows) with AND combination logic.

**Monte Carlo results (50 runs × 3600 steps):**

| Metric | Mean | Std | CV | Status |
|---|---|---|---|---|
| Precision | 0.588 | 0.078 | 0.132 | ✅ |
| Recall | 0.796 | 0.119 | 0.149 | ✅ |
| F1 | 0.669 | 0.075 | 0.112 | ✅ |
| PR-AUC | 0.400 | 0.085 | 0.213 | ⚠️ |

**Key decisions:**
- AND logic (both detectors must flag): reduces FP by 41% vs OR logic
- Max score for display (preserves PR curve shape)
- Isolated simulator instances for fault characterisation (fixed RNG contamination bug that degraded Monte Carlo CV by 2x)
- `train_after_long=400` for long window detector (needs more data for stable threshold)

**Lessons:**
- OR combination logic inflated false positives dramatically — threshold instability in one detector contaminates combined score
- RNG contamination from shared simulators is subtle but measurable — always use isolated instances for auxiliary computation

---

## Phase 6 — Semi-supervised fault characterisation

**Completed:** Fault characterisation pass after calibration — scores each fault type independently with isolated simulators.

**Finding:** Threshold manipulation from characterisation scores degraded performance (F1 0.578 → 0.545). The calibration threshold (μ+5σ) was already well-placed. Semi-supervised value is diagnostic only: confirms which fault types are detectable and which need better features.

**Detectability report (final model):**
- `efficiency_drop`: mu=0.041 [detectable — 18x above threshold]
- `overheating`: mu=0.021 [detectable — 9x above threshold]
- `vibration_spike`: mu=1.209 [detectable — very strong signal]
- `gearbox_fault`: mu=0.007 [detectable — just above threshold]
- `bearing_fault`: mu=0.0017 [HARD — below threshold, needs 100Hz vibration sensor]

---

## Phase 7 — Fraunhofer LBF vibration dataset

**Explored:** Bearing fault kurtosis from 74kHz accelerometer data.

**Finding:** At 74kHz, healthy bearing kurtosis (mean=44.8 at 1s windows) is HIGHER than fault bearing kurtosis (mean=2.8). This is the opposite of theoretical expectation. Root cause: at high sampling rates, healthy turbine mechanical impacts (wind gusts, blade passing) produce higher kurtosis than the smooth rolling-contact pattern of an inner race fault. Kurtosis at 0.01s windows (740 samples) approaches the theoretical value (healthy ~3.1).

**Conclusion:** Raw kurtosis from 74kHz signals cannot replace simulated kurtosis directly. Proper bearing fault detection at this sampling rate requires order-tracked envelope analysis (filter around defect frequency harmonics + tachometer), which is out of scope for this phase.

**Value:** Validated that our vibration simulator produces physically correct kurtosis values at 100Hz (healthy ~3.0 at 0.01s windows).

---

## Phase 8 — 3D Visualization

**Completed:** Three.js windmill with FastAPI backend serving pre-computed Monte Carlo results.

**Architecture:**
- FastAPI pre-computes all 50 runs at startup (~10-15 min), stores in memory
- Browser fetches run summaries on load, full time-series on run selection
- Run buttons colored by real F1 score
- Status ring: green=TN, red=TP, orange=FN, yellow=FP
- Blade color reflects efficiency ratio
- Wind particles speed reflects wind speed

---

## Known limitations

1. **Bearing fault undetectable** — needs 100Hz vibration + envelope analysis
2. **Vibration spike recall CV=0.752** — 8% power loss is near noise floor
3. **CARE to Compare incompatibility** — fault labels don't match physics-based fault signatures
4. **KL collapse persists** — VAE behaves as denoising AE; KL signal contributes but is weak
5. **PR-AUC CV=0.213** — above production-ready threshold; driven by score distribution variance
