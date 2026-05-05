"""
data_pipeline/schemas.py
=========================
Shared data contracts for the Wind Turbine Digital Twin.

Updated for 41-feature SCADA-style sensor reading.

Design decision: SensorReading holds all raw Tier 1 signals.
Tier 2 (residuals) and Tier 3 (rolling stats) are computed
by the FeatureEngineer and stored in EnrichedReading.
This keeps raw sensor data separate from derived features —
important for debugging and for swapping real vs simulated data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np


@dataclass
class WindReading:
    """Raw wind measurement at a single point in time."""
    timestamp: datetime
    wind_speed_ms: float
    wind_direction_deg: float = 0.0


@dataclass
class SensorReading:
    """
    Full SCADA-style sensor reading — 21 Tier 1 signals.
    
    Matches the CARE to Compare dataset column structure.
    Simulation generates these directly.
    Real data ingestion maps SCADA columns to these fields.
    """
    timestamp: datetime

    # ── Aerodynamic ───────────────────────────────────────────
    wind_speed_ms: float = 0.0
    wind_direction_deg: float = 0.0
    turbulence_intensity: float = 0.0

    # ── Rotor ─────────────────────────────────────────────────
    rotor_speed_rpm: float = 0.0
    pitch_angle_blade1_deg: float = 0.0
    pitch_angle_blade2_deg: float = 0.0
    pitch_angle_blade3_deg: float = 0.0

    # ── Drivetrain ────────────────────────────────────────────
    generator_speed_rpm: float = 0.0
    torque_nm: float = 0.0
    gearbox_oil_temp_c: float = 50.0

    # ── Electrical ────────────────────────────────────────────
    active_power_kw: float = 0.0
    reactive_power_kvar: float = 0.0
    grid_frequency_hz: float = 50.0
    grid_voltage_v: float = 690.0
    cos_phi: float = 0.95

    # ── Thermal ───────────────────────────────────────────────
    nacelle_temp_c: float = 45.0
    generator_temp_c: float = 60.0
    bearing_temp_fore_c: float = 40.0
    bearing_temp_aft_c: float = 40.0
    ambient_temp_c: float = 15.0

    # ── Mechanical ────────────────────────────────────────────
    hydraulic_pressure_bar: float = 180.0

    # ── Simulation metadata (not present in real data) ────────
    is_fault_injected: bool = False
    fault_type: Optional[str] = None


@dataclass
class VibrationWindow:
    """
    100Hz vibration samples for one 1-second SCADA timestep.
    100 samples per window — used for spectral feature extraction.
    """
    timestamp: datetime
    samples: np.ndarray           # Shape: (100,) — m/s²
    rotor_speed_rpm: float        # Needed for 1P/3P frequency calc
    sampling_rate_hz: float = 100.0

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.samples ** 2)))

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.samples)))

    @property
    def crest_factor(self) -> float:
        rms = self.rms
        return self.peak / rms if rms > 0 else 0.0

    @property
    def kurtosis(self) -> float:
        """
        Statistical kurtosis — sensitive to impulsive bearing faults.
        Normal vibration: kurtosis ≈ 3
        Bearing fault:    kurtosis >> 3 (impulsive spikes)
        """
        samples = self.samples
        mean = samples.mean()
        std  = samples.std()
        if std == 0:
            return 0.0
        return float(np.mean(((samples - mean) / std) ** 4))

    def frequency_amplitude(self, freq_hz: float) -> float:
        """
        Extract amplitude at a specific frequency using FFT.
        Used for 1P (rotor) and 3P (blade pass) components.
        """
        n = len(self.samples)
        fft_vals = np.abs(np.fft.rfft(self.samples)) / n
        freqs    = np.fft.rfftfreq(n, d=1.0/self.sampling_rate_hz)
        # Find closest frequency bin
        idx = int(np.argmin(np.abs(freqs - freq_hz)))
        return float(fft_vals[idx])

    @property
    def rotor_freq_hz(self) -> float:
        """1P frequency — once per rotor revolution."""
        return self.rotor_speed_rpm / 60.0

    @property
    def blade_pass_freq_hz(self) -> float:
        """3P frequency — three times per rotor revolution (3 blades)."""
        return self.rotor_freq_hz * 3.0


@dataclass
class EnrichedReading:
    """
    Full 41-feature reading after feature engineering.
    
    Combines:
      - SensorReading (Tier 1, 21 features)
      - Tier 2 residuals (8 features, computed by FeatureEngineer)
      - Tier 3 rolling stats (12 features, computed over window)
    
    This is what the ML model consumes.
    """
    timestamp: datetime
    sensor: SensorReading

    # ── Tier 2 — Physics residuals ────────────────────────────
    power_residual_kw: float = 0.0
    power_coefficient_cp: float = 0.0
    efficiency_ratio: float = 1.0
    tip_speed_ratio: float = 0.0
    torque_residual_nm: float = 0.0
    generator_efficiency: float = 1.0
    gearbox_thermal_margin: float = 30.0
    pitch_asymmetry_deg: float = 0.0

    # ── Tier 3 — Rolling statistics ───────────────────────────
    power_mean_1min: float = 0.0
    power_std_1min: float = 0.0
    power_slope_1min: float = 0.0
    wind_mean_1min: float = 0.0
    wind_std_1min: float = 0.0
    rotor_speed_std_1min: float = 0.0
    vibration_rms: float = 0.0
    vibration_peak: float = 0.0
    vibration_crest_factor: float = 1.414
    vibration_kurtosis: float = 3.0
    temp_gradient_rate: float = 0.0
    efficiency_slope_1min: float = 0.0

    # ── Fault metadata ────────────────────────────────────────
    is_fault_injected: bool = False
    fault_type: Optional[str] = None

    def to_feature_vector(self) -> np.ndarray:
        """
        Export all 41 features as a numpy array.
        Order matches models/feature_config.py FEATURE_NAMES.
        This is what the VAE receives as input.
        """
        s = self.sensor
        return np.array([
            # Tier 1 — 21 features
            s.wind_speed_ms,
            s.wind_direction_deg,
            s.turbulence_intensity,
            s.rotor_speed_rpm,
            s.pitch_angle_blade1_deg,
            s.pitch_angle_blade2_deg,
            s.pitch_angle_blade3_deg,
            s.generator_speed_rpm,
            s.torque_nm,
            s.gearbox_oil_temp_c,
            s.active_power_kw,
            s.reactive_power_kvar,
            s.grid_frequency_hz,
            s.grid_voltage_v,
            s.cos_phi,
            s.nacelle_temp_c,
            s.generator_temp_c,
            s.bearing_temp_fore_c,
            s.bearing_temp_aft_c,
            s.ambient_temp_c,
            s.hydraulic_pressure_bar,
            # Tier 2 — 8 features
            self.power_residual_kw,
            self.power_coefficient_cp,
            self.efficiency_ratio,
            self.tip_speed_ratio,
            self.torque_residual_nm,
            self.generator_efficiency,
            self.gearbox_thermal_margin,
            self.pitch_asymmetry_deg,
            # Tier 3 — 12 features
            self.power_mean_1min,
            self.power_std_1min,
            self.power_slope_1min,
            self.wind_mean_1min,
            self.wind_std_1min,
            self.rotor_speed_std_1min,
            self.vibration_rms,
            self.vibration_peak,
            self.vibration_crest_factor,
            self.vibration_kurtosis,
            self.temp_gradient_rate,
            self.efficiency_slope_1min,
        ], dtype=np.float32)


@dataclass
class TwinSnapshot:
    """
    Core output of the digital twin for one timestep.
    Updated to carry EnrichedReading instead of raw sensor.
    """
    timestamp: datetime
    wind_speed_ms: float
    expected_power_kw: float
    actual_power_kw: float
    deviation_kw: float
    deviation_pct: float
    efficiency_ratio: float
    enriched: Optional[EnrichedReading] = None
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    fault_type: Optional[str] = None


@dataclass
class AnomalyEvent:
    """Detected anomaly event for RAG assistant ingestion."""
    event_id: str
    detected_at: datetime
    duration_seconds: int
    wind_speed_ms: float
    expected_power_kw: float
    actual_power_kw: float
    deviation_pct: float
    anomaly_score: float
    suspected_fault: Optional[str] = None
    context_snapshots: list = field(default_factory=list)