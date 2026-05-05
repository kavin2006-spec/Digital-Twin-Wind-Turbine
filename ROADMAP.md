# Roadmap

What's built, what's next, and why in that order.

---

## Current state (v1.0)

Multi-scale Denoising VAE ensemble with 3D visualization. Fully functional simulation platform with real SCADA data ingestion. F1=0.669, three stability metrics production-ready.

---

## Near-term (build on this codebase)

### v1.1 — Real vibration kurtosis (Fraunhofer LBF)

**What:** Replace simulated `vibration_kurtosis` with kurtosis extracted from 74kHz bearing accelerometer data using order-tracked envelope analysis.

**Why:** `bearing_fault` is currently undetectable. This is the single highest-impact feature improvement available.

**How:**
1. Implement bandpass filter around BPFO/BPFI frequencies using tachometer RPM
2. Envelope the filtered signal, compute kurtosis
3. Replace `vibration_kurtosis / 10.0` feature in `_extract_features()`
4. Retrain VAE — expect bearing_fault to become detectable

**Data needed:** Fraunhofer LBF (already downloaded) + `tach` channel for RPM

**Expected impact:** bearing_fault recall 0.0 → 0.6+ based on kurtosis separation at envelope level

---

### v1.2 — Transformer attention encoder

**What:** Replace LSTM encoder with a Transformer encoder (multi-head self-attention over the window).

**Why:** Published benchmarks show VAE+LSTM+Transformer ensemble achieves F1=0.856 on CARE vs our F1=0.669. The Transformer captures long-range temporal dependencies within the window that LSTM misses.

**How:**
```python
class TransformerVAEEncoder(nn.Module):
    def __init__(self, n_features, latent_dim, n_heads=4, n_layers=2):
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_features, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.fc_mu     = nn.Linear(n_features, latent_dim)
        self.fc_logvar = nn.Linear(n_features, latent_dim)
```

**Expected impact:** F1 0.669 → 0.72+ based on published results

---

### v1.3 — Health Index trending

**What:** Continuous degradation score that trends over hours rather than a binary flag per step.

**Why:** Early warning is more valuable than fault detection. A bearing degrading over 3 days is more actionable than a sudden anomaly flag.

**How:**
- Exponential moving average of anomaly scores over configurable window (e.g. 6 hours)
- Health Index (HI) = 1 - EMA(score), normalized to [0,1]
- Alert thresholds: HI < 0.7 = warning, HI < 0.5 = critical
- Trend slope as additional feature

**Reference:** LSTM-AE on real SCADA detects early degradation weeks before failure using HI trending.

---

## Medium-term (new capabilities)

### v2.0 — Predictive maintenance scheduler

**What:** Use HI trends to predict Remaining Useful Life (RUL) and schedule maintenance windows.

**How:**
- Fit degradation model (linear, exponential, or Wiener process) to HI trajectory
- Predict time to HI=0.3 (critical threshold)
- Maintenance scheduler: balance turbine downtime cost vs fault risk cost
- Output: recommended maintenance date with confidence interval

**This builds directly on v1.3** — needs HI trending first.

---

### v2.1 — Multi-turbine fleet monitoring

**What:** Run one twin per turbine in a fleet, cross-turbine anomaly correlation.

**How:**
- Extend FastAPI to handle N turbines
- Fleet-level dashboard: which turbines are degrading simultaneously (weather event vs mechanical fault)
- Cross-turbine normalisation: efficiency_ratio relative to fleet median

---

### v2.2 — Real-time streaming

**What:** Replace post-hoc Monte Carlo replay with live data streaming.

**How:**
- WebSocket endpoint in FastAPI
- Twin runs continuously, pushes updates every N seconds
- Browser receives and renders in real-time
- Connect to real SCADA via OPC-UA adapter (same interface as simulation)

---

## Research phase (needs more data / compute)

### v3.0 — Disentangled VAE

**What:** Separate latent dimensions per fault type — one latent direction for efficiency loss, one for vibration, one for thermal.

**Why:** Standard VAE mixes all fault signatures in the same latent space. Disentangled VAE makes fault type diagnosis possible from the latent vector directly.

**Constraint:** Needs ~5000 labeled examples per fault type. Currently have simulated labels but no real labeled vibration data at scale.

---

### v3.1 — Diffusion model anomaly detection

**What:** Replace VAE with a denoising diffusion probabilistic model. Score anomalies by reconstruction probability under the diffusion process.

**Why:** Diffusion models learn richer data distributions than VAEs and don't suffer from posterior collapse. Recent results on time-series anomaly detection show significant improvement over VAE baselines.

**Constraint:** Much slower inference — may not suit real-time applications.

---

## Graphics roadmap (3D visualization)

### v1.1 vis — Realistic turbine mesh
Replace box/cylinder geometry with a proper GLTF turbine model. Three.js GLTFLoader — drop-in replacement for current geometry.

### v1.2 vis — Multi-turbine fleet view
Show 5 turbines simultaneously, each with independent health coloring. Click to drill down.

### v1.3 vis — VR interface
Three.js XR support — view the wind farm in WebXR (browser-based VR). Walk around the turbines, inspect fault indicators up close.
