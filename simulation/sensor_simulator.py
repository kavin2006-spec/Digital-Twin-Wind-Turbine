"""
simulation/sensor_simulator.py
================================
Produces full 21-signal Tier 1 SensorReading per timestep.

Upgrade from v1:
  v1: 5 signals (power, rpm, nacelle_temp, vibration, wind)
  v2: 21 signals — full SCADA-style Tier 1 coverage

Design decision: SensorSimulator is the "fake hardware" layer.
It takes physics-based expected outputs and adds:
  1. Gaussian sensor noise (every real sensor has this)
  2. Fault-induced deviations (from FaultInjector)
  3. Physically correlated secondary signals

Each signal group is isolated into its own method so you can
unit-test or swap individual subsystems independently.

When you have real hardware, replace this module with a hardware
adapter that reads from serial/OPC-UA/MQTT — the rest of the
system never changes. That's the value of this abstraction.

💡 Industrial insight: This "fake hardware → real hardware" swap
is standard in the industry. Siemens Gamesa uses an identical
pattern — their digital twin runs against simulated sensors in
development, then the same code reads from SCADA in production.
The interface contract (SensorReading) is what makes this possible.
"""

from __future__ import annotations
import math
import random
from datetime import datetime
from typing import Optional, Tuple

from models.physics_model import WindTurbinePhysicsModel
from simulation.fault_injector import FaultInjector
from simulation.vibration_simulator import VibrationSimulator
from data_pipeline.schemas import WindReading, SensorReading, VibrationWindow


# ── Physical constants ────────────────────────────────────────────────────────

_GEAR_RATIO          = 97.0    # Gearbox ratio (NREL 5MW)
_ROTOR_RADIUS_M      = 63.0    # Blade length (m), NREL 5MW
_RATED_RPM           = 12.1    # Rotor RPM at rated wind speed
_RATED_TORQUE_NM     = 3.95e6  # Nm at rated operation
_GRID_FREQ_HZ        = 50.0    # European grid frequency
_GRID_VOLTAGE_V      = 690.0   # Low-voltage side of transformer
_HYDRAULIC_RATED_BAR = 180.0   # Pitch hydraulic system nominal pressure

# Temperature baselines (°C) at rated operation
_BASE_NACELLE_TEMP   = 45.0
_BASE_GENERATOR_TEMP = 65.0
_BASE_BEARING_FORE   = 42.0
_BASE_BEARING_AFT    = 40.0
_BASE_GEARBOX_OIL    = 55.0


class SensorSimulator:
    """
    Produces SensorReading objects that mimic real turbine SCADA hardware.

    Args:
        physics_model:      Underlying physics model (IEC power curve)
        fault_injector:     Fault state machine
        vibration_sim:      100Hz vibration signal generator
        noise_stddev:       Fractional Gaussian noise applied to all signals
        ambient_temp_c:     Site ambient temperature (affects thermals)
        seed:               Random seed for reproducibility
    """

    def __init__(
        self,
        physics_model:   WindTurbinePhysicsModel,
        fault_injector:  FaultInjector,
        vibration_sim:   VibrationSimulator,
        noise_stddev:    float = 0.02,
        ambient_temp_c:  float = 15.0,
        seed:            int   = 7,
    ):
        self.model        = physics_model
        self.faults       = fault_injector
        self.vibration    = vibration_sim
        self.noise_std    = noise_stddev
        self.ambient_temp = ambient_temp_c
        self._rng         = random.Random(seed)

    # ── Public interface ──────────────────────────────────────────────────────

    def read(
        self,
        wind_reading: WindReading,
        timestamp:    Optional[datetime] = None,
    ) -> Tuple[SensorReading, VibrationWindow]:
        """
        Produce one full SensorReading + VibrationWindow for a wind condition.

        Returns both because the VibrationWindow feeds Tier 3 feature
        extraction (kurtosis, crest factor, spectral peaks) in the
        FeatureEngineer — it's not stored in SensorReading directly.

        Args:
            wind_reading: Current wind speed and direction
            timestamp:    Explicit timestamp (uses wind_reading.timestamp if None)

        Returns:
            (SensorReading with 21 Tier 1 signals, VibrationWindow at 100Hz)
        """
        ts = timestamp or wind_reading.timestamp
        v  = wind_reading.wind_speed_ms

        # Step fault state machine FIRST — fault state drives everything else
        fault = self.faults.step()

        # ── Core physics ──────────────────────────────────────────────────
        expected_power = self.model.expected_power_kw(v)
        faulted_power  = self.faults.apply_to_power(expected_power)
        actual_power   = max(0.0, self._add_noise(faulted_power))

        # ── Rotor & drivetrain ────────────────────────────────────────────
        rotor_rpm   = self._simulate_rotor_rpm(v)
        rotor_rpm   = self.faults.apply_to_rotor_rpm(rotor_rpm)
        gen_rpm     = self._simulate_generator_rpm(rotor_rpm)
        torque_nm   = self._simulate_torque(actual_power, rotor_rpm)
        torque_nm   = self.faults.apply_to_torque(torque_nm)

        # ── Pitch system ──────────────────────────────────────────────────
        # Three independent pitch actuators — slight natural asymmetry
        pitch_base  = self._simulate_pitch_angle(v)
        pitch1, pitch2, pitch3 = self.faults.apply_to_pitch(
            pitch_base, self._rng
        )

        # ── Electrical ───────────────────────────────────────────────────
        reactive_power = self._simulate_reactive_power(actual_power)
        cos_phi        = self._simulate_cos_phi(actual_power, reactive_power)
        grid_freq      = self._add_noise(_GRID_FREQ_HZ, scale=0.02)
        grid_voltage   = self._add_noise(_GRID_VOLTAGE_V, scale=2.0)

        # ── Thermal ──────────────────────────────────────────────────────
        load_fraction  = actual_power / max(
            self.model.config.rated_power_kw, 1.0
        )
        nacelle_temp   = self._simulate_nacelle_temp(load_fraction)
        nacelle_temp   = self.faults.apply_to_nacelle_temp(nacelle_temp)
        gen_temp       = self._simulate_generator_temp(load_fraction)
        gen_temp       = self.faults.apply_to_generator_temp(gen_temp)
        bearing_fore   = self._simulate_bearing_temp(
            load_fraction, rotor_rpm, fore=True
        )
        bearing_aft    = self._simulate_bearing_temp(
            load_fraction, rotor_rpm, fore=False
        )
        bearing_fore, bearing_aft = self.faults.apply_to_bearing_temps(
            bearing_fore, bearing_aft
        )
        gearbox_oil    = self._simulate_gearbox_oil_temp(load_fraction)
        gearbox_oil    = self.faults.apply_to_gearbox_temp(gearbox_oil)

        # ── Mechanical ───────────────────────────────────────────────────
        hydraulic_bar  = self._simulate_hydraulic_pressure(pitch_base)
        hydraulic_bar  = self.faults.apply_to_hydraulic(hydraulic_bar)

        # ── Vibration (100Hz window) ──────────────────────────────────────
        # Set fault state on vibration simulator BEFORE generating window
        if fault is not None:
            self.vibration.set_fault(
                fault_type=fault.fault_type,
                severity=fault.severity,
            )
        else:
            self.vibration.set_fault(fault_type=None, severity=0.0)

        vib_window = self.vibration.generate(
            rotor_speed_rpm=rotor_rpm,
            timestamp=ts,
        )

        # ── Assemble SensorReading ────────────────────────────────────────
        sensor = SensorReading(
            timestamp=ts,
            # Aerodynamic
            wind_speed_ms=round(v, 3),
            wind_direction_deg=round(
                self._add_noise(wind_reading.wind_direction_deg, scale=1.0), 1
            ),
            turbulence_intensity=round(
                max(0.0, self._add_noise(0.08, scale=0.01)), 4
            ),
            # Rotor
            rotor_speed_rpm=round(rotor_rpm, 3),
            pitch_angle_blade1_deg=round(pitch1, 3),
            pitch_angle_blade2_deg=round(pitch2, 3),
            pitch_angle_blade3_deg=round(pitch3, 3),
            # Drivetrain
            generator_speed_rpm=round(gen_rpm, 1),
            torque_nm=round(torque_nm, 1),
            gearbox_oil_temp_c=round(gearbox_oil, 2),
            # Electrical
            active_power_kw=round(actual_power, 3),
            reactive_power_kvar=round(reactive_power, 3),
            grid_frequency_hz=round(grid_freq, 3),
            grid_voltage_v=round(grid_voltage, 1),
            cos_phi=round(cos_phi, 4),
            # Thermal
            nacelle_temp_c=round(nacelle_temp, 2),
            generator_temp_c=round(gen_temp, 2),
            bearing_temp_fore_c=round(bearing_fore, 2),
            bearing_temp_aft_c=round(bearing_aft, 2),
            ambient_temp_c=round(
                self._add_noise(self.ambient_temp, scale=0.2), 2
            ),
            # Mechanical
            hydraulic_pressure_bar=round(hydraulic_bar, 2),
            # Fault metadata
            is_fault_injected=self.faults.is_fault_active,
            fault_type=self.faults.active_fault_type,
        )

        return sensor, vib_window

    def read_batch(
        self, wind_readings: list[WindReading]
    ) -> list[Tuple[SensorReading, VibrationWindow]]:
        """Process a list of wind readings. Returns (sensor, vib) pairs."""
        return [self.read(w) for w in wind_readings]

    # ── Signal simulation methods ─────────────────────────────────────────────

    def _add_noise(self, value: float, scale: float = None) -> float:
        """
        Add Gaussian noise. Uses relative noise by default (% of value).
        Relative noise is more physically realistic — sensor error scales
        with signal magnitude in most SCADA hardware.
        """
        if scale is None:
            scale = abs(value) * self.noise_std
        return value + self._rng.gauss(0, max(scale, 1e-6))

    def _simulate_rotor_rpm(self, wind_speed_ms: float) -> float:
        """
        Rotor RPM follows three operating regions:
          Region 1 (< cut-in):  parked, RPM = 0
          Region 2 (cut-in → rated): variable speed, RPM ∝ wind
          Region 3 (> rated):   constant speed, pitch control active

        💡 Industrial insight: Region 2 uses MPPT (Maximum Power Point
        Tracking) — the controller continuously adjusts RPM to maximise
        the power coefficient Cp. GE and Vestas both run proprietary
        MPPT algorithms; we approximate with a linear ramp here.
        """
        cfg = self.model.config
        if wind_speed_ms < cfg.cut_in_speed_ms:
            return 0.0
        if wind_speed_ms >= cfg.rated_speed_ms:
            return max(0.0, self._add_noise(_RATED_RPM, scale=0.05))

        fraction = (wind_speed_ms - cfg.cut_in_speed_ms) / (
            cfg.rated_speed_ms - cfg.cut_in_speed_ms
        )
        rpm = max(3.0, fraction * _RATED_RPM)  # 3 RPM minimum operating speed
        return max(0.0, self._add_noise(rpm, scale=0.1))

    def _simulate_generator_rpm(self, rotor_rpm: float) -> float:
        """Generator RPM = rotor RPM × gearbox ratio."""
        gen_rpm = rotor_rpm * _GEAR_RATIO
        return max(0.0, self._add_noise(gen_rpm, scale=5.0))

    def _simulate_torque(
        self, power_kw: float, rotor_rpm: float
    ) -> float:
        """
        Torque from power and angular velocity: τ = P / ω
        ω = RPM × 2π / 60

        Using actual power (post-fault, post-noise) keeps torque
        physically consistent with the power signal.
        """
        if rotor_rpm < 0.1:
            return 0.0
        omega = rotor_rpm * 2 * math.pi / 60.0   # rad/s
        torque = (power_kw * 1000.0) / omega      # W / (rad/s) = Nm
        return max(0.0, self._add_noise(torque, scale=torque * 0.01))

    def _simulate_pitch_angle(self, wind_speed_ms: float) -> float:
        """
        Pitch angle follows wind speed:
          Region 2: fine pitch (≈0°) — maximum energy capture
          Region 3: pitched out to limit power — increases with wind speed

        Pitch angle ≈ 0° below rated, ramps to ~30° at cut-out.
        """
        cfg = self.model.config
        if wind_speed_ms <= cfg.rated_speed_ms:
            return 0.0  # Fine pitch — no feathering needed
        # Linear ramp from 0° at rated to 30° at cut-out
        fraction = (wind_speed_ms - cfg.rated_speed_ms) / max(
            cfg.cut_out_speed_ms - cfg.rated_speed_ms, 1.0
        )
        return min(30.0, fraction * 30.0)

    def _simulate_reactive_power(self, active_power_kw: float) -> float:
        """
        Reactive power: turbines absorb reactive power at low loads,
        supply at rated. Target power factor ≈ 0.95.
        Q ≈ P × tan(arccos(0.95)) ≈ P × 0.329
        """
        q = active_power_kw * 0.329
        return self._add_noise(q, scale=abs(q) * 0.03)

    def _simulate_cos_phi(
        self, active_kw: float, reactive_kvar: float
    ) -> float:
        """cos φ = P / S where S = √(P² + Q²)."""
        s = math.sqrt(active_kw ** 2 + reactive_kvar ** 2)
        if s < 1.0:
            return 1.0
        cos_phi = active_kw / s
        return min(1.0, max(0.0, self._add_noise(cos_phi, scale=0.005)))

    def _simulate_nacelle_temp(self, load_fraction: float) -> float:
        """
        Nacelle temperature rises with load (electrical losses → heat).
        Also rises with ambient temperature.
        """
        temp = (
            _BASE_NACELLE_TEMP
            + 12.0 * load_fraction
            + 0.4 * (self.ambient_temp - 15.0)  # Ambient coupling
        )
        return self._add_noise(temp, scale=0.5)

    def _simulate_generator_temp(self, load_fraction: float) -> float:
        """
        Generator winding temperature — higher thermal mass than nacelle.
        Copper losses ∝ I² ∝ P², so temp rises faster at high loads.
        """
        temp = (
            _BASE_GENERATOR_TEMP
            + 25.0 * (load_fraction ** 1.5)   # Non-linear — I²R losses
            + 0.3 * (self.ambient_temp - 15.0)
        )
        return self._add_noise(temp, scale=0.8)

    def _simulate_bearing_temp(
        self,
        load_fraction: float,
        rotor_rpm:     float,
        fore:          bool = True,
    ) -> float:
        """
        Main bearing temperatures.
        Fore bearing: higher load (thrust + weight).
        Aft bearing:  lower load, slightly cooler.

        Both rise with RPM (friction) and load (transmitted force).
        """
        base = _BASE_BEARING_FORE if fore else _BASE_BEARING_AFT
        temp = (
            base
            + 8.0 * load_fraction
            + 0.15 * rotor_rpm          # Friction term
            + 0.2 * (self.ambient_temp - 15.0)
        )
        return self._add_noise(temp, scale=0.4)

    def _simulate_gearbox_oil_temp(self, load_fraction: float) -> float:
        """
        Gearbox oil temperature.
        Oil absorbs heat from gear mesh losses (≈2% of transmitted power).
        Healthy gearbox: stable temp, small gradient.
        Fault: rising temp, elevated gradient → caught by Tier 3 features.
        """
        temp = (
            _BASE_GEARBOX_OIL
            + 20.0 * load_fraction
            + 0.3 * (self.ambient_temp - 15.0)
        )
        return self._add_noise(temp, scale=0.6)

    def _simulate_hydraulic_pressure(self, pitch_angle_deg: float) -> float:
        """
        Hydraulic pitch system pressure.
        High pitch angle → higher demand on hydraulic actuators → slight
        pressure drop from the accumulator.
        Healthy: stable around 180 bar.
        """
        # Pressure slightly lower when pitching actively
        demand = pitch_angle_deg / 30.0   # 0–1 normalised pitch demand
        pressure = _HYDRAULIC_RATED_BAR - 5.0 * demand
        return max(100.0, self._add_noise(pressure, scale=1.5))