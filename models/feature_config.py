"""
models/feature_config.py
=========================
Central feature registry for the Wind Turbine Digital Twin.

Design decision: Define ALL features in one place with full metadata.
This single file drives:
  - Sensor simulator (what to generate)
  - Feature engineer (what to compute)
  - SCADA loader (how to map real columns)
  - VAE architecture (input dimension)
  - Visualiser (axis labels, units)
  - Evaluator (which features are most diagnostic)

Adding a new feature = add one entry here.
Everything else picks it up automatically.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FeatureSpec:
    """
    Full specification for one feature.
    
    Drives normalisation, validation, and model architecture.
    """
    name: str
    description: str
    unit: str
    tier: int                          # 1=raw, 2=residual, 3=rolling
    expected_min: float
    expected_max: float
    fault_sensitive: list[str]         # Which fault types affect this
    scada_column: Optional[str] = None # CARE to Compare column name
    is_primary: bool = False           # Key diagnostic signal


# ─── Tier 1 — Raw SCADA Signals ──────────────────────────────────────────────

TIER1_FEATURES = [

    # Aerodynamic
    FeatureSpec(
        name="wind_speed_ms",
        description="Hub height wind speed",
        unit="m/s",
        tier=1,
        expected_min=0.0,
        expected_max=35.0,
        fault_sensitive=[],            # Input condition, not fault-sensitive
        scada_column="Ws_avg",
    ),
    FeatureSpec(
        name="wind_direction_deg",
        description="Wind direction from North",
        unit="degrees",
        tier=1,
        expected_min=0.0,
        expected_max=360.0,
        fault_sensitive=["yaw_misalignment"],
        scada_column="Wd_avg",
    ),
    FeatureSpec(
        name="turbulence_intensity",
        description="Rolling std/mean of wind speed (60s window)",
        unit="dimensionless",
        tier=1,
        expected_min=0.0,
        expected_max=0.5,
        fault_sensitive=[],
    ),

    # Rotor
    FeatureSpec(
        name="rotor_speed_rpm",
        description="Main rotor shaft rotational speed",
        unit="RPM",
        tier=1,
        expected_min=0.0,
        expected_max=16.0,
        fault_sensitive=["bearing_fault", "gearbox_fault"],
        scada_column="Nf_avg",
        is_primary=True,
    ),
    FeatureSpec(
        name="pitch_angle_blade1_deg",
        description="Blade 1 pitch angle",
        unit="degrees",
        tier=1,
        expected_min=-5.0,
        expected_max=90.0,
        fault_sensitive=["pitch_fault", "efficiency_drop"],
        scada_column="Bf1_avg",
    ),
    FeatureSpec(
        name="pitch_angle_blade2_deg",
        description="Blade 2 pitch angle",
        unit="degrees",
        tier=1,
        expected_min=-5.0,
        expected_max=90.0,
        fault_sensitive=["pitch_fault", "efficiency_drop"],
        scada_column="Bf2_avg",
    ),
    FeatureSpec(
        name="pitch_angle_blade3_deg",
        description="Blade 3 pitch angle",
        unit="degrees",
        tier=1,
        expected_min=-5.0,
        expected_max=90.0,
        fault_sensitive=["pitch_fault", "efficiency_drop"],
        scada_column="Bf3_avg",
    ),

    # Drivetrain
    FeatureSpec(
        name="generator_speed_rpm",
        description="Generator shaft rotational speed",
        unit="RPM",
        tier=1,
        expected_min=0.0,
        expected_max=1800.0,
        fault_sensitive=["bearing_fault", "gearbox_fault"],
        scada_column="Ng_avg",
        is_primary=True,
    ),
    FeatureSpec(
        name="torque_nm",
        description="Generator torque",
        unit="Nm",
        tier=1,
        expected_min=0.0,
        expected_max=5_000_000.0,
        fault_sensitive=["gearbox_fault", "efficiency_drop"],
        scada_column="Cm_avg",
        is_primary=True,
    ),
    FeatureSpec(
        name="gearbox_oil_temp_c",
        description="Gearbox oil temperature",
        unit="°C",
        tier=1,
        expected_min=10.0,
        expected_max=100.0,
        fault_sensitive=["gearbox_fault", "overheating"],
        scada_column="To_avg",
        is_primary=True,
    ),

    # Electrical
    FeatureSpec(
        name="active_power_kw",
        description="Electrical active power output",
        unit="kW",
        tier=1,
        expected_min=-100.0,
        expected_max=5500.0,
        fault_sensitive=["efficiency_drop", "electrical_fault"],
        scada_column="P_avg",
        is_primary=True,
    ),
    FeatureSpec(
        name="reactive_power_kvar",
        description="Electrical reactive power",
        unit="kVAR",
        tier=1,
        expected_min=-2000.0,
        expected_max=2000.0,
        fault_sensitive=["electrical_fault"],
        scada_column="Q_avg",
    ),
    FeatureSpec(
        name="grid_frequency_hz",
        description="Grid frequency",
        unit="Hz",
        tier=1,
        expected_min=49.0,
        expected_max=51.0,
        fault_sensitive=["electrical_fault"],
        scada_column="Fa_avg",
    ),
    FeatureSpec(
        name="grid_voltage_v",
        description="Grid voltage",
        unit="V",
        tier=1,
        expected_min=600.0,
        expected_max=700.0,
        fault_sensitive=["electrical_fault"],
        scada_column="Vhv1_avg",
    ),
    FeatureSpec(
        name="cos_phi",
        description="Power factor (cosine of phase angle)",
        unit="dimensionless",
        tier=1,
        expected_min=0.85,
        expected_max=1.0,
        fault_sensitive=["electrical_fault"],
    ),

    # Thermal
    FeatureSpec(
        name="nacelle_temp_c",
        description="Nacelle internal temperature",
        unit="°C",
        tier=1,
        expected_min=10.0,
        expected_max=80.0,
        fault_sensitive=["overheating"],
        scada_column="Tt_avg",
    ),
    FeatureSpec(
        name="generator_temp_c",
        description="Generator winding temperature",
        unit="°C",
        tier=1,
        expected_min=20.0,
        expected_max=120.0,
        fault_sensitive=["overheating", "electrical_fault"],
        scada_column="Tg_avg",
        is_primary=True,
    ),
    FeatureSpec(
        name="bearing_temp_fore_c",
        description="Main bearing fore-side temperature",
        unit="°C",
        tier=1,
        expected_min=10.0,
        expected_max=90.0,
        fault_sensitive=["bearing_fault", "overheating"],
        scada_column="Tb1_avg",
        is_primary=True,
    ),
    FeatureSpec(
        name="bearing_temp_aft_c",
        description="Main bearing aft-side temperature",
        unit="°C",
        tier=1,
        expected_min=10.0,
        expected_max=90.0,
        fault_sensitive=["bearing_fault", "overheating"],
        scada_column="Tb2_avg",
    ),
    FeatureSpec(
        name="ambient_temp_c",
        description="Outside ambient temperature",
        unit="°C",
        tier=1,
        expected_min=-20.0,
        expected_max=45.0,
        fault_sensitive=[],
        scada_column="Ta_avg",
    ),

    # Mechanical
    FeatureSpec(
        name="hydraulic_pressure_bar",
        description="Hydraulic pitch system pressure",
        unit="bar",
        tier=1,
        expected_min=100.0,
        expected_max=250.0,
        fault_sensitive=["pitch_fault", "hydraulic_fault"],
    ),
]

# ─── Tier 2 — Physics-Informed Residuals ─────────────────────────────────────

TIER2_FEATURES = [
    FeatureSpec(
        name="power_residual_kw",
        description="Actual power minus physics model prediction",
        unit="kW",
        tier=2,
        expected_min=-2000.0,
        expected_max=500.0,
        fault_sensitive=["efficiency_drop", "pitch_fault"],
        is_primary=True,
    ),
    FeatureSpec(
        name="power_coefficient_cp",
        description="Actual Cp vs theoretical max (Betz)",
        unit="dimensionless",
        tier=2,
        expected_min=0.0,
        expected_max=0.593,
        fault_sensitive=["efficiency_drop", "pitch_fault"],
        is_primary=True,
    ),
    FeatureSpec(
        name="efficiency_ratio",
        description="Actual / expected power ratio",
        unit="dimensionless",
        tier=2,
        expected_min=0.0,
        expected_max=1.2,
        fault_sensitive=["efficiency_drop", "pitch_fault"],
        is_primary=True,
    ),
    FeatureSpec(
        name="tip_speed_ratio",
        description="Blade tip speed / wind speed",
        unit="dimensionless",
        tier=2,
        expected_min=0.0,
        expected_max=12.0,
        fault_sensitive=["pitch_fault", "rotor_imbalance"],
        is_primary=True,
    ),
    FeatureSpec(
        name="torque_residual_nm",
        description="Actual torque minus expected torque",
        unit="Nm",
        tier=2,
        expected_min=-500_000.0,
        expected_max=200_000.0,
        fault_sensitive=["gearbox_fault", "efficiency_drop"],
    ),
    FeatureSpec(
        name="generator_efficiency",
        description="Electrical / mechanical power ratio",
        unit="dimensionless",
        tier=2,
        expected_min=0.8,
        expected_max=1.0,
        fault_sensitive=["electrical_fault", "generator_fault"],
        is_primary=True,
    ),
    FeatureSpec(
        name="gearbox_thermal_margin",
        description="Gearbox temp minus ambient temp",
        unit="°C",
        tier=2,
        expected_min=5.0,
        expected_max=60.0,
        fault_sensitive=["gearbox_fault", "overheating"],
        is_primary=True,
    ),
    FeatureSpec(
        name="pitch_asymmetry_deg",
        description="Max minus min pitch angle across 3 blades",
        unit="degrees",
        tier=2,
        expected_min=0.0,
        expected_max=5.0,
        fault_sensitive=["pitch_fault", "rotor_imbalance"],
        is_primary=True,
    ),
]

# ─── Tier 3 — Rolling Window Statistics ──────────────────────────────────────

TIER3_FEATURES = [
    FeatureSpec(
        name="power_mean_1min",
        description="1-minute rolling mean of active power",
        unit="kW",
        tier=3,
        expected_min=0.0,
        expected_max=5500.0,
        fault_sensitive=["efficiency_drop"],
    ),
    FeatureSpec(
        name="power_std_1min",
        description="1-minute rolling std of active power",
        unit="kW",
        tier=3,
        expected_min=0.0,
        expected_max=500.0,
        fault_sensitive=["efficiency_drop", "rotor_imbalance"],
    ),
    FeatureSpec(
        name="power_slope_1min",
        description="Linear trend slope of power over 1 minute",
        unit="kW/s",
        tier=3,
        expected_min=-100.0,
        expected_max=100.0,
        fault_sensitive=["efficiency_drop"],
    ),
    FeatureSpec(
        name="wind_mean_1min",
        description="1-minute rolling mean of wind speed",
        unit="m/s",
        tier=3,
        expected_min=0.0,
        expected_max=35.0,
        fault_sensitive=[],
    ),
    FeatureSpec(
        name="wind_std_1min",
        description="1-minute rolling std of wind speed (turbulence)",
        unit="m/s",
        tier=3,
        expected_min=0.0,
        expected_max=5.0,
        fault_sensitive=[],
    ),
    FeatureSpec(
        name="rotor_speed_std_1min",
        description="1-minute RPM stability indicator",
        unit="RPM",
        tier=3,
        expected_min=0.0,
        expected_max=2.0,
        fault_sensitive=["rotor_imbalance", "bearing_fault"],
    ),
    FeatureSpec(
        name="vibration_rms",
        description="Root mean square vibration (100Hz window)",
        unit="m/s²",
        tier=3,
        expected_min=0.0,
        expected_max=5.0,
        fault_sensitive=["bearing_fault", "rotor_imbalance"],
        is_primary=True,
    ),
    FeatureSpec(
        name="vibration_peak",
        description="Peak vibration amplitude (100Hz window)",
        unit="m/s²",
        tier=3,
        expected_min=0.0,
        expected_max=15.0,
        fault_sensitive=["bearing_fault", "rotor_imbalance"],
    ),
    FeatureSpec(
        name="vibration_crest_factor",
        description="Peak / RMS ratio (bearing fault indicator)",
        unit="dimensionless",
        tier=3,
        expected_min=1.0,
        expected_max=10.0,
        fault_sensitive=["bearing_fault"],
        is_primary=True,
    ),
    FeatureSpec(
        name="vibration_kurtosis",
        description="Statistical kurtosis of vibration signal",
        unit="dimensionless",
        tier=3,
        expected_min=2.0,
        expected_max=20.0,
        fault_sensitive=["bearing_fault"],
        is_primary=True,
    ),
    FeatureSpec(
        name="temp_gradient_rate",
        description="Rate of temperature change per minute",
        unit="°C/min",
        tier=3,
        expected_min=-2.0,
        expected_max=5.0,
        fault_sensitive=["overheating", "bearing_fault"],
        is_primary=True,
    ),
    FeatureSpec(
        name="efficiency_slope_1min",
        description="Linear trend in efficiency ratio over 1 minute",
        unit="1/s",
        tier=3,
        expected_min=-0.01,
        expected_max=0.01,
        fault_sensitive=["efficiency_drop", "gearbox_fault"],
        is_primary=True,
    ),
]

# ─── Combined registry ────────────────────────────────────────────────────────

ALL_FEATURES: list[FeatureSpec] = TIER1_FEATURES + TIER2_FEATURES + TIER3_FEATURES

FEATURE_NAMES: list[str] = [f.name for f in ALL_FEATURES]
N_FEATURES: int = len(ALL_FEATURES)

PRIMARY_FEATURES: list[str] = [f.name for f in ALL_FEATURES if f.is_primary]

SCADA_COLUMN_MAP: dict[str, str] = {
    f.scada_column: f.name
    for f in ALL_FEATURES
    if f.scada_column is not None
}

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    f.name: (f.expected_min, f.expected_max)
    for f in ALL_FEATURES
}

FAULT_FEATURE_MAP: dict[str, list[str]] = {}
for feat in ALL_FEATURES:
    for fault in feat.fault_sensitive:
        if fault not in FAULT_FEATURE_MAP:
            FAULT_FEATURE_MAP[fault] = []
        FAULT_FEATURE_MAP[fault].append(feat.name)


def get_feature(name: str) -> FeatureSpec:
    """Lookup a feature by name."""
    for f in ALL_FEATURES:
        if f.name == name:
            return f
    raise KeyError(f"Feature '{name}' not found in registry")


def features_by_tier(tier: int) -> list[FeatureSpec]:
    """Get all features in a specific tier."""
    return [f for f in ALL_FEATURES if f.tier == tier]


def features_sensitive_to(fault_type: str) -> list[str]:
    """Get feature names sensitive to a specific fault type."""
    return FAULT_FEATURE_MAP.get(fault_type, [])


if __name__ == "__main__":
    print(f"Total features:   {N_FEATURES}")
    print(f"Tier 1 (raw):     {len(TIER1_FEATURES)}")
    print(f"Tier 2 (residual):{len(TIER2_FEATURES)}")
    print(f"Tier 3 (rolling): {len(TIER3_FEATURES)}")
    print(f"Primary features: {len(PRIMARY_FEATURES)}")
    print(f"\nSCADA columns mapped: {len(SCADA_COLUMN_MAP)}")
    print(f"\nFault → feature mapping:")
    for fault, features in FAULT_FEATURE_MAP.items():
        print(f"  {fault}: {features}")