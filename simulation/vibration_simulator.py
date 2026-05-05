"""
simulation/vibration_simulator.py
===================================
Generates realistic 100Hz vibration signals for one SCADA timestep.

Physics background:
  Rotating machinery vibration has deterministic frequency components
  on top of random broadband noise:

  1P  = once per rotor revolution (rotor imbalance, mass imbalance)
  3P  = three times per rotor revolution (blade pass frequency)
  GMF = gear mesh frequency (rotor_rpm × gear_teeth / 60)

  Healthy turbine:
    Clean 1P and 3P peaks, low broadband noise, kurtosis ≈ 3

  Bearing fault:
    Impulsive spikes at bearing defect frequencies
    Kurtosis >> 3 (3-10 for early fault, 10+ for severe)
    Elevated broadband noise floor

  Rotor imbalance:
    Elevated 1P amplitude
    Possible 2P harmonic

  Gearbox fault:
    Elevated GMF sidebands
    Increased noise floor

Design decision: Generate full 100-sample time series per timestep.
This allows proper spectral feature extraction (FFT, kurtosis, crest
factor) that single-value RMS cannot provide.
"""

from __future__ import annotations
import math
import random
import numpy as np
from datetime import datetime
from typing import Optional

from data_pipeline.schemas import VibrationWindow


class VibrationSimulator:
    """
    Generates 100Hz vibration signals with physically realistic
    frequency components and fault signatures.

    Args:
        sampling_rate_hz:   Vibration sampling rate (default 100Hz)
        base_noise_level:   Broadband noise floor (m/s²)
        seed:               Random seed for reproducibility
    """

    # Gearbox ratio for NREL 5MW (approximate)
    GEARBOX_RATIO = 97.0
    GEAR_TEETH    = 23      # Low-speed stage teeth count

    def __init__(
        self,
        sampling_rate_hz: float = 100.0,
        base_noise_level: float = 0.05,
        seed: int = 77,
    ):
        self.fs              = sampling_rate_hz
        self.base_noise      = base_noise_level
        self._rng            = random.Random(seed)
        self._np_rng         = np.random.default_rng(seed)

        # Fault state — injected from outside
        self._fault_type:     Optional[str]  = None
        self._fault_severity: float          = 0.0

    def set_fault(
        self,
        fault_type:  Optional[str],
        severity:    float = 0.0,
    ) -> None:
        """Set active fault for next generation call."""
        self._fault_type    = fault_type
        self._fault_severity = severity

    def generate(
        self,
        rotor_speed_rpm: float,
        timestamp:       datetime,
        n_samples:       int = 100,
    ) -> VibrationWindow:
        """
        Generate one window of vibration samples.

        Args:
            rotor_speed_rpm: Current rotor speed (drives frequency components)
            timestamp:       Window timestamp
            n_samples:       Number of samples (default 100 at 100Hz = 1 second)

        Returns:
            VibrationWindow with samples and extracted properties.
        """
        if rotor_speed_rpm < 0.1:
            # Turbine idle — just noise
            samples = self._np_rng.normal(0, self.base_noise, n_samples)
            return VibrationWindow(
                timestamp=timestamp,
                samples=samples.astype(np.float32),
                rotor_speed_rpm=rotor_speed_rpm,
                sampling_rate_hz=self.fs,
            )

        # Time array for this window
        t = np.linspace(0, n_samples / self.fs, n_samples, endpoint=False)

        # Key frequencies
        f_1p  = rotor_speed_rpm / 60.0           # Rotor frequency (Hz)
        f_3p  = f_1p * 3.0                        # Blade pass frequency
        f_gmf = f_1p * self.GEAR_TEETH            # Gear mesh frequency

        # ── Base signal — healthy components ─────────────────────────
        signal = np.zeros(n_samples)

        # 1P component (always present, small in healthy turbine)
        signal += 0.08 * np.sin(2 * np.pi * f_1p * t)

        # 3P component (blade pass, always present)
        signal += 0.05 * np.sin(2 * np.pi * f_3p * t + 0.3)

        # GMF component (gearbox mesh, small in healthy state)
        if f_gmf < self.fs / 2:   # Only if below Nyquist
            signal += 0.03 * np.sin(2 * np.pi * f_gmf * t + 0.7)

        # Broadband noise floor
        signal += self._np_rng.normal(0, self.base_noise, n_samples)

        # ── Fault signatures ──────────────────────────────────────────
        signal = self._apply_fault_signature(
            signal, t, f_1p, f_3p, f_gmf, n_samples
        )

        return VibrationWindow(
            timestamp=timestamp,
            samples=signal.astype(np.float32),
            rotor_speed_rpm=rotor_speed_rpm,
            sampling_rate_hz=self.fs,
        )

    def _apply_fault_signature(
        self,
        signal:    np.ndarray,
        t:         np.ndarray,
        f_1p:      float,
        f_3p:      float,
        f_gmf:     float,
        n_samples: int,
    ) -> np.ndarray:
        """
        Apply fault-specific vibration signature.

        Each fault type has a physically motivated effect:
          bearing_fault:    impulsive spikes (kurtosis >> 3)
          rotor_imbalance:  elevated 1P amplitude
          gearbox_fault:    elevated GMF + sidebands
          overheating:      subtle broadband increase (thermal expansion)
          efficiency_drop:  slight pitch-related 1P increase
          vibration_spike:  broadband spike event
        """
        if self._fault_type is None or self._fault_severity == 0:
            return signal

        sev = self._fault_severity

        if self._fault_type == "bearing_fault":
            # Bearing defect frequency (simplified — outer race)
            f_bpfo = f_1p * 3.5   # Typical outer race defect frequency
            # Add impulsive spikes at bearing defect frequency
            # These create high kurtosis — the key bearing fault indicator
            spike_times = np.arange(0, t[-1], 1.0 / f_bpfo)
            for st in spike_times:
                idx = int(st * self.fs)
                if 0 <= idx < n_samples:
                    # Decaying exponential spike
                    width = int(self.fs * 0.005)  # 5ms spike width
                    for j in range(width):
                        if idx + j < n_samples:
                            signal[idx + j] += (
                                sev * 0.8
                                * math.exp(-j / (width * 0.3))
                                * (1 if j % 2 == 0 else -0.5)
                            )
            # Elevated broadband noise
            signal += self._np_rng.normal(0, sev * 0.15, n_samples)

        elif self._fault_type == "rotor_imbalance":
            # Elevated 1P — mass imbalance rotates at rotor frequency
            signal += sev * 0.6 * np.sin(2 * np.pi * f_1p * t + 0.5)
            # 2P harmonic (characteristic of blade pitch asymmetry)
            signal += sev * 0.2 * np.sin(2 * np.pi * f_1p * 2 * t + 1.0)

        elif self._fault_type in ("gearbox_fault", "overheating"):
            # Elevated GMF with sidebands
            if f_gmf < self.fs / 2:
                signal += sev * 0.4 * np.sin(2 * np.pi * f_gmf * t)
                # Sidebands at GMF ± 1P (characteristic gearbox fault)
                signal += sev * 0.2 * np.sin(
                    2 * np.pi * (f_gmf + f_1p) * t
                )
                signal += sev * 0.2 * np.sin(
                    2 * np.pi * (f_gmf - f_1p) * t
                )
            signal += self._np_rng.normal(0, sev * 0.1, n_samples)

        elif self._fault_type == "efficiency_drop":
            # Pitch-related — slight 1P increase
            signal += sev * 0.3 * np.sin(2 * np.pi * f_1p * t + 1.2)

        elif self._fault_type == "vibration_spike":
            # Broadband shock event
            spike_center = n_samples // 2
            spike_width  = int(n_samples * 0.1)
            for j in range(n_samples):
                dist = abs(j - spike_center)
                if dist < spike_width:
                    signal[j] += (
                        sev * 1.5
                        * math.exp(-dist / (spike_width * 0.4))
                        * self._np_rng.standard_normal()
                    )

        return signal