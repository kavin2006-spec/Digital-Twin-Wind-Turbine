# Wind Turbine Digital Twin with ML Anomaly Detection

A physics-informed digital twin for wind turbine condition monitoring, combining a multi-scale denoising VAE ensemble with a 3D browser visualization. Built from scratch — no industrial APIs, no labeled training data.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What this is

A full-stack anomaly detection system for wind turbines that:

- **Simulates** a physically realistic NREL 5MW turbine (IEC power curve, Ornstein-Uhlenbeck wind, 5 fault types, 21 SCADA signals, 100Hz vibration)
- **Detects** anomalies using an ensemble of denoising Variational Autoencoders trained on physics-residual features
- **Evaluates** model stability via 50-run Monte Carlo with cost-sensitive metrics
- **Visualises** simulation results in an interactive 3D browser interface (Three.js + FastAPI)

This is not an industrial SCADA system. It is a research and learning platform demonstrating how digital twin principles apply to condition monitoring.

---

## Results (50-run Monte Carlo, 3600 steps/run)

| Model                            | Precision | Recall    | F1        | PR-AUC    | FP mean |
| -------------------------------- | --------- | --------- | --------- | --------- | ------- |
| Isolation Forest                 | 0.320     | 0.376     | 0.238     | 0.320     | —       |
| LSTM Autoencoder                 | 0.446     | 0.484     | 0.464     | 0.360     | —       |
| Temporal VAE (4-feat, 5σ)        | 0.476     | 0.760     | 0.578     | 0.427     | 320     |
| **Multi-scale VAE (30s+120s) ★** | **0.588** | **0.796** | **0.669** | **0.400** | **188** |

★ = recommended model. Stability: Precision CV=0.132 ✅, Recall CV=0.149 ✅, F1 CV=0.112 ✅

**Fault type detection (best model):**

- `efficiency_drop` (35% power loss): Recall 0.700 ✅
- `overheating` (20% power loss): Recall 0.744 ✅
- `vibration_spike` (8% power loss): Recall 0.594 ⚠️
- `bearing_fault`: Undetectable without 100Hz vibration sensor ⛔

---

## Architecture

```
Wind Generator (Ornstein-Uhlenbeck)
    ↓
Sensor Simulator (21 Tier-1 SCADA signals + 100Hz VibrationWindow)
    ↓
FaultInjector (state machine: 5 fault types)
    ↓
FeatureEngineer (41 features: Tier-1 raw + Tier-2 physics residuals + Tier-3 rolling stats)
    ↓
Multi-scale Denoising VAE
    ├── Short detector (30s window) — catches sudden faults
    └── Long detector  (120s window) — catches gradual degradation
    ↓ AND combination
AnomalyResult (score, is_anomaly, reconstruction_error, kl_divergence)
    ↓
FastAPI server → Three.js 3D visualization
```

**Key design decisions:**

- Physics-residual features (efficiency_ratio, deviation_pct) remove wind-speed correlation before feeding the VAE — this is the single most important design choice
- Denoising VAE: corrupt input with Gaussian noise, reconstruct clean — forces encoder to learn robust normal-state representations
- Ensemble of 3 models with different seeds — reduces variance by ~40% vs single model
- Calibration threshold: μ+5σ of normal calibration windows (raw reconstruction error, not normalised score) — eliminates score-ceiling bug that plagued earlier versions
- AND combination for multi-scale: both windows must agree before flagging — reduces false positives 41% vs OR logic

---

## Project structure

```
wind_turbine_twin/
├── api/
│   ├── server.py              # FastAPI backend (pre-computes Monte Carlo, serves JSON)
│   └── static/
│       └── windmill_viz.html  # Three.js 3D visualization
├── config/
│   └── turbine_config.yaml    # All hyperparameters
├── data_pipeline/
│   ├── schemas.py             # SensorReading, VibrationWindow, EnrichedReading, TwinSnapshot
│   ├── feature_engineer.py    # Tier-2 physics residuals + Tier-3 rolling stats
│   ├── pipeline.py            # Validation + CSV export
│   ├── scada_loader.py        # CARE to Compare Wind Farm A loader
│   └── dataset_registry.py    # SQLite query interface
├── models/
│   ├── physics_model.py       # IEC power curve
│   ├── ml_anomaly_detector.py # Multi-scale denoising VAE ensemble
│   ├── evaluator.py           # Confusion matrix, PR-AUC, cost-sensitive metrics
│   ├── visualizer.py          # Matplotlib plots
│   └── feature_config.py      # 41-feature registry with metadata
├── simulation/
│   ├── wind_generator.py      # Ornstein-Uhlenbeck wind process
│   ├── fault_injector.py      # State machine fault injection (5 types)
│   ├── sensor_simulator.py    # 21 Tier-1 SCADA signals
│   └── vibration_simulator.py # 100Hz physics-based vibration
├── twin_engine/
│   ├── twin.py                # Orchestrator — wires all modules
│   └── comparator.py          # Deviation metrics
├── scripts/
│   ├── build_database.py      # CARE to Compare → SQLite
│   ├── prepare_dataset.py     # Train/eval partition
│   ├── run_real_data_experiment.py
│   └── extract_lbf_kurtosis.py # Fraunhofer LBF kurtosis extraction
├── experiments/
│   └── compare.py             # Model comparison table
├── main.py                    # CLI: simulate | evaluate | dashboard
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Install dependencies
pip install torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Run a single simulation (3600 steps)
python main.py simulate 3600

# 3. Run Monte Carlo evaluation (50 runs)
python main.py evaluate 50 3600

# 4. Launch 3D visualization
pip install fastapi uvicorn
python api/server.py
# Open http://localhost:8000
```

---

## Datasets used

### Simulation

Physics-based NREL 5MW reference turbine. No external data required for core functionality.

### CARE to Compare (real SCADA data)

- Source: https://zenodo.org/records/10958775
- Wind Farm A: 5 turbines, 1.2M rows, 10-minute averages, 86 features
- Used for: real-data experiment, feature schema validation
- Note: fault labels in this dataset correspond to high-load operating periods, not mechanical degradation — limits supervised evaluation

### Fraunhofer LBF Vibration Dataset

- Source: https://zenodo.org/records/11820598
- 74kHz bearing accelerometer data with labeled fault conditions
- Used for: vibration simulator validation, kurtosis characterisation
- Download: `Healthy.zip` + `Bearing.zip` (~6GB)

---

## Configuration

All hyperparameters in `config/turbine_config.yaml`:

```yaml
ml:
  contamination: 0.05
  random_state: 42
  window_size: 60
  window_size_short: 30
  window_size_long: 120
  multi_scale_weight: 0.5 # 0=all short, 1=all long
  slide_every: 10
  latent_dim: 4
  hidden_size: 64
  n_epochs: 50
  train_after: 200
  epsilon: 0.01
  max_clean_iterations: 3
  beta: 0.5
  kl_weight: 0.3
  noise_scale: 0.03
  n_ensemble: 3
  calibration_size: 200
  lr: 0.001
  grad_clip: 1.0
  train_after_long: 400
  fault_start_after_step: 600
anomaly:
  enabled: true
  fault_probability: 0.002
  fault_duration_seconds: 60
  efficiency_drop_fraction: 0.35
  fault_start_after_step: 600
simulation:
  duration_seconds: 3600
  sensor_noise_stddev: 0.02
  timestep_seconds: 1
  wind_mean_speed_ms: 9.0
  wind_turbulence_intensity: 0.12
  vibration_sampling_rate_hz: 100.0
  vibration_base_noise: 0.05
  vibration_seed: 77
  ambient_temp_c: 15.0
turbine:
  air_density_kg_m3: 1.225
  cut_in_speed_ms: 3.0
  cut_out_speed_ms: 25.0
  hub_height_m: 90.0
  name: NREL 5MW Reference Turbine
  power_coefficient_cp: 0.45
  rated_power_kw: 5000.0
  rated_speed_ms: 11.4
  rotor_diameter_m: 126.0
```

---

## How anomaly detection works

**Training (unsupervised):**

1. VAE trains on first N steps of normal operation
2. Learns to reconstruct 4 physics-residual features: `efficiency_ratio`, `deviation_pct`, `vibration_kurtosis`, `nacelle_temp`
3. Denoising: input is corrupted with Gaussian noise, target is clean — forces robust encoding

**Calibration:**

1. Post-training, known-normal steps feed calibration buffer
2. Threshold = μ + 5σ of reconstruction errors on those windows
3. Fault characterisation: isolated simulators score each fault type — reports which are detectable

**Inference:**

1. Short window (30s) and long window (120s) score independently
2. Combined score = max(short_score, long_score)
3. Anomaly flagged only if BOTH detectors exceed threshold (AND logic)

---

## What's next (roadmap)

### Phase 7 — Real vibration kurtosis (Fraunhofer LBF)

Replace simulated `vibration_kurtosis` with kurtosis extracted from 74kHz bearing accelerometer data. Bearing faults currently undetectable — this directly fixes that.

### Phase 8 — Transformer attention layer

Replace LSTM encoder with Transformer encoder. <br>
Expected: +10-15% F1 based on published benchmarks (VAE+LSTM+Transformer ensemble achieves F1=0.856 on CARE).

### Phase 9 — Health Index trending

Replace binary anomaly flag with a continuous degradation score that trends over hours/days. Early warning system rather than fault detection.

### Phase 10 — Predictive maintenance scheduler

Use anomaly score trends to predict remaining useful life and schedule maintenance windows. Builds directly on this codebase.

---

## Comparison with published work

| Paper / System                          | Method                                   | F1    | Data            |
| --------------------------------------- | ---------------------------------------- | ----- | --------------- |
| This project                            | Multi-scale Denoising VAE (unsupervised) | 0.669 | Simulated       |
| Hybrid Autoencoder (arxiv 2510.15010)   | VAE+LSTM+Transformer ensemble            | 0.856 | CARE (labeled)  |
| 1D-CNN (Nature Scientific Reports 2026) | Supervised 1D-CNN                        | 0.850 | SCADA (labeled) |
| LSTM-AE (SCADA)                         | LSTM Autoencoder                         | ~0.60 | Real SCADA      |
| Isolation Forest                        | Statistical baseline                     | 0.238 | Simulated       |

The approach is fully unsupervised (no fault labels used during training), making direct comparison with supervised methods unfair. Within the unsupervised category our results are competitive.

---

## License

MIT
