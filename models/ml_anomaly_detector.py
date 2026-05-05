"""
models/ml_anomaly_detector.py
==============================
Ensemble Denoising Temporal VAE anomaly detector.

Architecture: 4-feature LSTM-based Denoising VAE.

Why 4 features and not 41:
  Raw SCADA signals (RPM, torque, power) are highly correlated with
  wind speed. Training a VAE on them requires thousands of windows to
  separate wind-driven variation from fault-driven variation.
  
  These 4 features are physics-normalised residuals — wind speed is
  already removed from them:
    efficiency_ratio    = actual / expected power  (wind removed)
    deviation_pct       = % deviation from expected (wind removed)
    vibration_kurtosis  = statistical fault indicator (wind independent)
    nacelle_temp_c      = thermal load indicator

  The VAE only needs to learn "normal efficiency ~ 1.0, normal kurtosis
  ~ 3.0" — a simple distribution learnable from 141 windows.

  Feature expansion (Tier 2 residuals) is the correct next step once
  Monte Carlo stability is confirmed on this baseline.

Denoising VAE:
  Input to encoder:   X + gaussian_noise (corrupted)
  Reconstruction target: X (clean)
  Forces the encoder to learn robust normal-state representations
  rather than memorising the training data.

Ensemble:
  n_ensemble models trained with different random seeds.
  Anomaly score = average across all models.
  Reduces single-model variance — key for stable Monte Carlo results.

Training safeguards:
  - Beta warmup: 0 -> beta over first n_epochs//2 epochs
    Prevents posterior collapse (encoder gives up on KL term)
  - Gradient clipping: max_norm=1.0
    Prevents encoder collapse under large gradient spikes
  - KL collapse guard: if KL < threshold for 5 epochs, reset encoder
    Catches collapse that warmup alone doesn't prevent

Calibration:
  After training, feed known-normal steps into calibration buffer.
  Threshold = mu + 3*sigma of normal scores (robust, not 95th percentile)
  Requires calibration_size windows for stability.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_pipeline.schemas import SensorReading, TwinSnapshot


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    is_anomaly:           bool
    score:                float
    z_score:              float
    method:               str
    reconstruction_error: float = 0.0
    kl_divergence:        float = 0.0


# ── Model ─────────────────────────────────────────────────────────────────────

class TemporalVAEEncoder(nn.Module):
    """LSTM encoder: sequence -> (mu, logvar)."""

    def __init__(self, n_features: int, latent_dim: int, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=64,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (hidden, _) = self.lstm(x)
        h = hidden[-1]
        return self.fc_mu(h), self.fc_logvar(h)

    def reset_weights(self) -> None:
        for layer in self.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class TemporalVAEDecoder(nn.Module):
    """LSTM decoder: z -> reconstructed sequence."""

    def __init__(
        self, n_features: int, latent_dim: int,
        window_size: int, n_layers: int = 2
    ):
        super().__init__()
        self.window_size = window_size
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=64,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.output_layer = nn.Linear(64, n_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        repeated   = z.unsqueeze(1).repeat(1, self.window_size, 1)
        decoded, _ = self.lstm(repeated)
        return self.output_layer(decoded)


class TemporalVAE(nn.Module):
    """Full Denoising Temporal VAE."""

    def __init__(
        self, n_features: int, latent_dim: int,
        window_size: int, n_layers: int = 2
    ):
        super().__init__()
        self.encoder = TemporalVAEEncoder(n_features, latent_dim, n_layers)
        self.decoder = TemporalVAEDecoder(
            n_features, latent_dim, window_size, n_layers
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu   # deterministic at inference

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar    = self.encoder(x)
        z             = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


def vae_loss(
    reconstructed: torch.Tensor,
    original:      torch.Tensor,
    mu:            torch.Tensor,
    logvar:        torch.Tensor,
    beta:          float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss = MSE reconstruction + beta * KL divergence."""
    recon_loss = F.mse_loss(reconstructed, original, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ── Detector ──────────────────────────────────────────────────────────────────

class LSTMAutoencoderDetector:
    """
    Ensemble Denoising Temporal VAE anomaly detector.

    4 physics-normalised features:
      efficiency_ratio    — actual/expected power (wind removed)
      deviation_pct       — % deviation from expected (wind removed)
      vibration_kurtosis  — bearing fault indicator (kurtosis >> 3 = fault)
      nacelle_temp_c      — thermal load (normalised by /100)

    💡 Industrial insight: Using physics residuals instead of raw signals
    is called "signal conditioning" in ABB's Asset Health Center platform.
    They found anomaly models trained on raw SCADA needed 10x more data
    than models trained on physics residuals, because residuals remove
    wind-driven variance that dominates raw signal distributions.
    """

    N_FEATURES = 4
    FEATURE_NAMES = [
        "efficiency_ratio",
        "deviation_pct_normalised",
        "vibration_kurtosis_normalised",
        "nacelle_temp_normalised",
    ]

    def __init__(
        self,
        window_size:           int   = 60,
        slide_every:           int   = 10,
        latent_dim:            int   = 4,
        n_layers:              int   = 2,
        n_epochs:              int   = 50,
        lr:                    float = 0.001,
        train_after:           int   = 200,
        contamination:         float = 0.05,
        epsilon:               float = 0.01,
        max_clean_iterations:  int   = 3,
        beta:                  float = 0.5,
        kl_weight:             float = 0.3,
        n_ensemble:            int   = 3,
        noise_scale:           float = 0.03,
        grad_clip:             float = 1.0,
        kl_collapse_threshold: float = 1e-4,
        calibration_size:      int   = 200,
    ):
        self.window_size           = window_size
        self.slide_every           = slide_every
        self.latent_dim            = latent_dim
        self.n_layers              = n_layers
        self.n_epochs              = n_epochs
        self.lr                    = lr
        self.train_after           = train_after
        self.contamination         = contamination
        self.epsilon               = epsilon
        self.max_clean_iterations  = max_clean_iterations
        self.beta                  = beta
        self.kl_weight             = kl_weight
        self.n_ensemble            = n_ensemble
        self.noise_scale           = noise_scale
        self.grad_clip             = grad_clip
        self.kl_collapse_threshold = kl_collapse_threshold

        self._models: list[TemporalVAE] = []
        self._model:  Optional[TemporalVAE] = None

        self._buffer: deque[list[float]] = deque(
            maxlen=max(train_after, window_size * 4)
        )
        self._step_count  = 0
        self._is_trained  = False
        self._score_threshold: float = 0.75

        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std:  Optional[np.ndarray] = None
        self._recon_p95:    float = 1.0
        self._kl_p95:       float = 1.0
        self._recon_mean:   float = 0.0
        self._recon_std:    float = 1.0
        self._kl_mean:      float = 0.0

        self._last_result:               Optional[AnomalyResult] = None
        self._consecutive_anomaly_count: int = 0
        self._dwell_threshold:           int = 1

        # Raw recon/KL thresholds set by calibration (not normalised scores)
        self._recon_threshold: float = float("inf")   # flags nothing until calibrated
        self._kl_threshold:    float = float("inf")

        # Semi-supervised: fault characterisation scores
        # Set by characterise_faults() after calibration
        self._fault_recon_mean:   Optional[float] = None
        self._fault_recon_std:    Optional[float] = None
        self._characterisation_done: bool = False

        self._calibration_size:   int  = calibration_size
        self._calibration_done:   bool = False
        self._calibration_buffer: list[list[float]] = []

    # ── Public interface ──────────────────────────────────────────────────────

    def update(self, snapshot: TwinSnapshot, sensor: SensorReading) -> None:
        features = self._extract_features(snapshot, sensor)
        self._buffer.append(features)
        self._step_count += 1

        if not self._is_trained and self._step_count >= self.train_after:
            print("\n[VAEDetector] Starting ensemble denoising VAE training...")
            self._train()

    def predict(
        self, snapshot: TwinSnapshot, sensor: SensorReading
    ) -> AnomalyResult:
        if not self._is_trained:
            return self._warmup_predict(snapshot)
        if (
            self._step_count % self.slide_every == 0
            or self._last_result is None
        ):
            self._last_result = self._compute_anomaly_score()
        return self._last_result

    def calibrate(
        self, snapshot: TwinSnapshot, sensor: SensorReading
    ) -> bool:
        """Feed one known-normal step into calibration buffer."""
        if self._calibration_done or not self._is_trained:
            return self._calibration_done
        features = self._extract_features(snapshot, sensor)
        self._calibration_buffer.append(features)
        if len(self._calibration_buffer) >= self._calibration_size:
            self._run_calibration()
            return True
        return False

    def set_score_threshold(self, threshold: float) -> None:
        self._score_threshold = threshold

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def training_progress(self) -> float:
        return min(1.0, self._step_count / self.train_after)

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_features(
        self, snapshot: TwinSnapshot, sensor: SensorReading
    ) -> list[float]:
        """
        4 physics-normalised features.

        vibration_kurtosis is now real (from EnrichedReading.vibration_kurtosis
        computed by FeatureEngineer from the 100Hz VibrationWindow).
        Healthy kurtosis ~ 3.0. Bearing fault >> 3.0.
        Normalised by /10 to put it in the same range as other features.

        Falls back to sensor.nacelle_temp_c if enriched not available.
        """
        vib_kurtosis = 3.0   # healthy baseline default
        if snapshot.enriched is not None:
            vib_kurtosis = snapshot.enriched.vibration_kurtosis

        return [
            snapshot.efficiency_ratio,
            snapshot.deviation_pct / 100.0,
            vib_kurtosis / 10.0,
            sensor.nacelle_temp_c / 100.0,
        ]

    # ── Training ──────────────────────────────────────────────────────────────

    def _train(self) -> None:
        """
        Train n_ensemble denoising VAE models.

        Denoising: corrupt input X with gaussian noise, reconstruct clean X.
        Forces encoder to learn robust representations of normal state
        rather than memorising exact training values.

        Beta warmup: start beta=0 (pure reconstruction), ramp to target.
        This lets reconstruction stabilise before KL pressure is applied,
        preventing posterior collapse where encoder outputs N(0,1) for
        everything and decoder ignores the latent space.
        """
        data = np.array(list(self._buffer), dtype=np.float32)
        # Fixed-range normalisation -- no z-score needed
        # (features are already in [0,1] range by design)
        self._feature_mean = np.zeros(self.N_FEATURES, dtype=np.float32)
        self._feature_std  = np.ones(self.N_FEATURES,  dtype=np.float32)
        data_norm          = self._normalise(data)
        all_windows        = self._make_windows(data_norm)

        if len(all_windows) < 10:
            print("[VAEDetector] Not enough windows to train")
            return

        X            = torch.tensor(all_windows, dtype=torch.float32)
        self._models = []

        for i in range(self.n_ensemble):
            seed = 42 + i * 17
            torch.manual_seed(seed)
            np.random.seed(seed)

            print(
                f"\n[VAEDetector] Training model {i+1}/{self.n_ensemble} "
                f"(seed={seed})..."
            )

            model     = TemporalVAE(
                n_features=self.N_FEATURES,
                latent_dim=self.latent_dim,
                window_size=self.window_size,
                n_layers=self.n_layers,
            )
            optimizer        = torch.optim.Adam(model.parameters(), lr=self.lr)
            warmup_epochs    = self.n_epochs // 2
            kl_low_count     = 0
            collapse_retried = False

            model.train()
            for epoch in range(self.n_epochs):
                current_beta = (
                    self.beta * (epoch / max(warmup_epochs, 1))
                    if epoch < warmup_epochs else self.beta
                )

                optimizer.zero_grad()

                # Denoising: corrupt input, reconstruct clean
                noise       = torch.randn_like(X) * self.noise_scale
                X_corrupted = X + noise

                reconstructed, mu, logvar = model(X_corrupted)
                loss, recon_l, kl_l = vae_loss(
                    reconstructed, X, mu, logvar, current_beta
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.grad_clip
                )
                optimizer.step()

                # KL collapse guard
                kl_val       = kl_l.item()
                kl_low_count = (
                    kl_low_count + 1
                    if kl_val < self.kl_collapse_threshold else 0
                )

                if kl_low_count >= 5 and not collapse_retried:
                    print(
                        f"  [!] KL collapse at epoch {epoch+1} "
                        f"(KL={kl_val:.6f}) -- reinitialising encoder"
                    )
                    model.encoder.reset_weights()
                    optimizer        = torch.optim.Adam(
                        model.parameters(), lr=self.lr * 0.5
                    )
                    kl_low_count     = 0
                    collapse_retried = True

                if (epoch + 1) % 10 == 0:
                    print(
                        f"  Epoch {epoch+1:3d}/{self.n_epochs} "
                        f"-- Loss: {loss.item():.4f} "
                        f"(recon: {recon_l.item():.4f}, "
                        f"kl: {kl_l.item():.5f}, "
                        f"beta={current_beta:.3f})"
                    )

            self._models.append(model)

        # Compute p95 normalisation from training windows
        all_recon, all_kl = self._ensemble_score(X)
        self._recon_p95   = float(np.percentile(all_recon, 95))
        self._kl_p95      = float(np.percentile(all_kl,    95))
        self._recon_mean  = float(all_recon.mean())
        self._recon_std   = float(all_recon.std())
        self._kl_mean     = float(all_kl.mean())
        self._model       = self._models[0]
        self._is_trained  = True

        kl_alive = self._kl_p95 > self.kl_collapse_threshold
        print(f"\n[VAEDetector] Ensemble training complete.")
        print(f"  Models:    {len(self._models)}")
        print(f"  Features:  {self.N_FEATURES} ({', '.join(self.FEATURE_NAMES)})")
        print(f"  Windows:   {len(all_windows)}")
        print(f"  Latent:    {self.latent_dim}")
        print(f"  Recon p95: {self._recon_p95:.4f}")
        print(f"  KL p95:    {self._kl_p95:.6f}  "
              f"{'alive' if kl_alive else 'COLLAPSED'}")

    # ── Calibration ───────────────────────────────────────────────────────────

    def _run_calibration(self) -> None:
        """
        Set anomaly threshold from known-normal calibration windows.

        Core insight: the threshold must be set on RAW reconstruction
        errors from calibration windows, not on p95-normalised scores.

        Previous approach was broken:
          normalised_score = recon / recon_p95_training
          recon_p95_training = 95th percentile of TRAINING recon errors
          Calibration windows come from same distribution as training
          -> ~50% of calibration windows have recon > recon_p95_training
          -> normalised score clips to 1.0 for half of normal data
          -> threshold always hits the 0.95 ceiling

        Correct approach:
          1. Score calibration windows -> raw recon errors
          2. Threshold = mu + 3*sigma of those raw recon errors
          3. At inference: flag if raw recon > threshold
          This uses calibration distribution as reference, not training p95.

        💡 Industrial insight: This is called "threshold learning from
        nominal data" in condition monitoring literature. ISO 13381-1
        (machinery prognostics standard) specifies that alarm thresholds
        must be derived from known-good operational data, not from the
        model's own training statistics.
        """
        if len(self._calibration_buffer) < 10:
            print("[VAEDetector] Calibration buffer too small -- keeping threshold")
            self._calibration_done = True
            return

        data      = np.array(self._calibration_buffer, dtype=np.float32)
        data_norm = self._normalise(data)
        windows   = self._make_windows(data_norm)

        if len(windows) < 5:
            self._calibration_done = True
            return

        X = torch.tensor(windows, dtype=torch.float32)
        recon_errors, kl_divs = self._ensemble_score(X)

        # Set threshold on RAW recon errors from calibration
        recon_mean = float(recon_errors.mean())
        recon_std  = float(recon_errors.std())

        if recon_std < 1e-8:
            recon_threshold = recon_mean * 2.0
            method = "2x_mean"
        else:
            recon_threshold = recon_mean + 3.0 * recon_std
            method = "mu+3sigma_raw"

        # Similarly for KL if alive
        if self._kl_p95 > self.kl_collapse_threshold:
            kl_mean = float(kl_divs.mean())
            kl_std  = float(kl_divs.std())
            kl_threshold = kl_mean + 3.0 * kl_std
        else:
            kl_threshold = float("inf")

        # Store raw thresholds -- used directly at inference
        self._recon_threshold = recon_threshold
        self._kl_threshold    = kl_threshold
        self._calibration_done = True

        print(f"\n[VAEDetector] Calibration complete ({method}).")
        print(f"  Windows:         {len(windows)}")
        print(f"  Recon mu:        {recon_mean:.6f}  sigma: {recon_std:.6f}")
        print(f"  Recon threshold: {recon_threshold:.6f}")
        if kl_threshold != float("inf"):
            print(f"  KL threshold:    {kl_threshold:.6f}")

    # ── Semi-supervised fault characterisation ───────────────────────────────

    def characterise_faults_by_type(
        self, fault_features_by_type: dict[str, list[list[float]]]
    ) -> None:
        """
        Score each fault type separately and find optimal threshold.

        Characterises each fault type independently to avoid mixing
        easy-to-detect faults (efficiency_drop: large recon error)
        with hard-to-detect ones (vibration_spike: tiny recon error).

        Strategy: use the fault type with the LOWEST mean recon error
        (hardest to detect) above the normal threshold as the anchor.
        This maximises recall on the hardest fault type without
        inflating the threshold beyond what's needed.

        Args:
            fault_features_by_type: dict mapping fault_type -> feature vectors
        """
        if not self._is_trained or not self._calibration_done:
            print("[VAEDetector] Characterisation skipped -- not trained/calibrated")
            return

        print(f"[VAEDetector] Characterising {len(fault_features_by_type)} fault types...")

        type_stats: dict[str, tuple[float, float]] = {}

        for fault_type, features in fault_features_by_type.items():
            if len(features) < self.window_size:
                continue
            data      = np.array(features, dtype=np.float32)
            data_norm = self._normalise(data)
            windows   = self._make_windows(data_norm)
            if len(windows) < 5:
                continue
            X = torch.tensor(windows, dtype=torch.float32)
            recon_errors, _ = self._ensemble_score(X)
            mu  = float(recon_errors.mean())
            std = float(recon_errors.std())
            type_stats[fault_type] = (mu, std)
            detectable = "detectable" if mu > self._recon_threshold else "HARD"
            print(f"  {fault_type:20s}: mu={mu:.6f} std={std:.6f} [{detectable}]")

        if not type_stats:
            print("[VAEDetector] No valid fault windows -- keeping threshold")
            self._characterisation_done = True
            return

        # Use the fault type whose lower bound (mu - 1*std) is closest
        # to but still above the normal threshold.
        # This gives the tightest threshold that still catches that fault type.
        normal_threshold = self._recon_threshold
        best_type   = None
        best_lower  = float("inf")

        for fault_type, (mu, std) in type_stats.items():
            lower = mu - 1.0 * std
            if lower > normal_threshold and lower < best_lower:
                best_lower = lower
                best_type  = fault_type

        # Report detectability -- do NOT change threshold
        n_detectable = sum(
            1 for mu, std in type_stats.values()
            if mu > normal_threshold
        )
        print(f"  Detectable fault types: {n_detectable}/{len(type_stats)}")
        if best_type is not None:
            print(f"  Tightest detectable: {best_type} "
                  f"(lower bound={best_lower:.6f})")
        undetectable = [
            t for t, (mu, _) in type_stats.items()
            if mu <= normal_threshold
        ]
        if undetectable:
            print(f"  Undetectable (need better features): {undetectable}")
        print(f"  Threshold unchanged: {normal_threshold:.6f}")

        self._fault_recon_mean = float(
            np.mean([mu for mu, _ in type_stats.values()])
        )
        self._fault_recon_std = float(
            np.mean([std for _, std in type_stats.values()])
        )
        self._characterisation_done = True
        print(f"  Final threshold: {self._recon_threshold:.6f}")

    def _optimise_threshold(self) -> None:
        """
        Set threshold at optimal point between normal and fault distributions.

        normal_upper = current recon_threshold (mu + 5sigma from calibration)
        fault_lower  = fault_recon_mean - 1*fault_recon_std

        If gap exists: threshold = midpoint of gap
        If partial overlap: threshold = fault_mean - 0.5*std (capped at 70% of original)
        If full overlap: keep original threshold, log that fault is hard to detect
        """
        if not self._characterisation_done:
            return

        fault_lower   = self._fault_recon_mean - 1.0 * self._fault_recon_std
        old_threshold = self._recon_threshold

        if fault_lower > self._recon_threshold:
            # Clean gap -- midpoint
            self._recon_threshold = (self._recon_threshold + fault_lower) / 2.0
            print(f"[VAEDetector] Clean gap found -- threshold: "
                  f"{old_threshold:.6f} -> {self._recon_threshold:.6f}")

        elif self._fault_recon_mean > self._recon_threshold:
            # Partial overlap -- move down cautiously
            candidate = self._fault_recon_mean - 0.5 * self._fault_recon_std
            self._recon_threshold = max(candidate, old_threshold * 0.70)
            print(f"[VAEDetector] Partial overlap -- threshold: "
                  f"{old_threshold:.6f} -> {self._recon_threshold:.6f}")

        else:
            # Full overlap -- fault invisible to current features
            print(f"[VAEDetector] Fault distribution fully overlaps normal "
                  f"(fault mu={self._fault_recon_mean:.6f} <= "
                  f"threshold={self._recon_threshold:.6f})")
            print(f"  Fault type not detectable with current features.")

    @property
    def characterisation_done(self) -> bool:
        return self._characterisation_done

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        """
        Fixed-range normalisation for the 4 hand-crafted features.

        These features are already near [0,1] by design:
          efficiency_ratio:   expected ~1.0, range [0, 1.5]
          deviation_pct/100:  expected ~0.0, range [-0.5, 0.5]
          kurtosis/10:        expected ~0.3, range [0, 2.0]
          nacelle_temp/100:   expected ~0.45, range [0.2, 0.8]

        Z-score normalisation fails here because std is tiny (~0.01)
        during normal operation, causing the normalised values to have
        huge variance from tiny raw differences. The LSTM can't learn
        tight clusters after z-score.

        Fixed centering: subtract known healthy-state mean.
        No division -- features already on comparable scale.
        """
        # Known healthy-state centres for each feature
        centres = np.array([1.0, 0.0, 0.3, 0.45], dtype=np.float32)
        return X - centres

    def _make_windows(self, data: np.ndarray) -> np.ndarray:
        n = len(data) - self.window_size + 1
        if n <= 0:
            return np.array([])
        return np.array([data[i:i + self.window_size] for i in range(n)])

    def _score_windows_vae(
        self, model: TemporalVAE, X: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            reconstructed, mu, logvar = model(X)
            recon_errors  = ((X - reconstructed) ** 2).mean(dim=[1, 2]).numpy()
            kl_per_window = -0.5 * (
                1 + logvar - mu.pow(2) - logvar.exp()
            ).mean(dim=1).numpy()
        return recon_errors, kl_per_window

    def _ensemble_score(
        self, X: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Average recon error and KL across all ensemble members."""
        recon_list, kl_list = [], []
        for model in self._models:
            recon, kl = self._score_windows_vae(model, X)
            recon_list.append(recon)
            kl_list.append(kl)
        return np.mean(recon_list, axis=0), np.mean(kl_list, axis=0)

    def _combine_scores(
        self, recon: np.ndarray, kl: np.ndarray
    ) -> np.ndarray:
        """
        Combine recon + KL into [0,1] score.
        Falls back to recon-only if KL collapsed.
        """
        recon_norm = np.clip(recon / (self._recon_p95 + 1e-8), 0, 1)
        if self._kl_p95 > self.kl_collapse_threshold:
            kl_norm = np.clip(kl / (self._kl_p95 + 1e-8), 0, 1)
            kl_w    = self.kl_weight
        else:
            kl_norm = np.zeros_like(recon_norm)
            kl_w    = 0.0
        return (1 - kl_w) * recon_norm + kl_w * kl_norm

    def _compute_anomaly_score(self) -> AnomalyResult:
        if len(self._buffer) < self.window_size:
            return self._warmup_predict(None)

        recent      = np.array(
            list(self._buffer)[-self.window_size:], dtype=np.float32
        )
        recent_norm = self._normalise(recent)
        x           = torch.tensor(
            recent_norm, dtype=torch.float32
        ).unsqueeze(0)

        recon_vals, kl_vals = [], []
        for model in self._models:
            model.eval()
            with torch.no_grad():
                reconstructed, mu, logvar = model(x)
                recon_vals.append(
                    float(((x - reconstructed) ** 2).mean().item())
                )
                kl_vals.append(float(
                    (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))
                    .mean().item()
                ))

        recon_error = float(np.mean(recon_vals))
        kl_div      = float(np.mean(kl_vals))

        # Compare raw errors directly against calibration thresholds
        # recon_score > 1.0 means above normal calibration range
        recon_score = recon_error / (self._recon_threshold + 1e-10)
        if self._kl_threshold < float("inf"):
            kl_score = kl_div / (self._kl_threshold + 1e-10)
            combined = (1 - self.kl_weight) * recon_score + self.kl_weight * kl_score
        else:
            combined = recon_score

        # Map to [0,1] for display using x/(x+1) -- smooth, interpretable
        # 0.5 = at threshold, > 0.5 = anomalous, < 0.5 = normal
        display_score = float(combined / (combined + 1.0))
        z = (recon_error - self._recon_mean) / (self._recon_std + 1e-8)

        # Anomaly = combined raw score above calibration threshold
        is_anomalous = combined > 1.0

        if is_anomalous:
            self._consecutive_anomaly_count += 1
        else:
            self._consecutive_anomaly_count = 0

        return AnomalyResult(
            is_anomaly=self._consecutive_anomaly_count >= self._dwell_threshold,
            score=round(display_score, 4),
            z_score=round(z, 3),
            method="ensemble_denoising_vae",
            reconstruction_error=round(recon_error, 6),
            kl_divergence=round(kl_div, 6),
        )

    def _warmup_predict(
        self, snapshot: Optional[TwinSnapshot]
    ) -> AnomalyResult:
        if snapshot is None:
            return AnomalyResult(
                is_anomaly=False, score=0.0,
                z_score=0.0, method="warmup"
            )
        return AnomalyResult(
            is_anomaly=snapshot.efficiency_ratio < 0.6,
            score=1.0 - snapshot.efficiency_ratio,
            z_score=0.0,
            method="warmup",
        )