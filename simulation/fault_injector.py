"""
simulation/fault_injector.py
=============================
Injects realistic fault scenarios across all 21 Tier 1 signals.

Upgrade from v1:
  v1: 3 fault types, 3 apply methods (power, vibration, nacelle_temp)
  v2: 5 fault types, 11 apply methods — one per affected signal group

Fault types:
  efficiency_drop:  Blade fouling / pitch misalignment — power loss
  vibration_spike:  Bearing wear — vibration + bearing temps
  overheating:      Cooling failure — generator + nacelle temps
  bearing_fault:    Main bearing degradation — bearing temps + vibration
  gearbox_fault:    Gearbox wear — oil temp + torque ripple + vibration

Design principle: Each fault type has a PRIMARY effect and SECONDARY
effects. Primary is the large observable signal. Secondary effects are
subtle correlations that a good model should learn to use.

💡 Industrial insight: In real turbines, almost every fault produces
correlated signatures across multiple signals. Bearing faults show
elevated temperature AND vibration kurtosis AND torque ripple —
sometimes days before a human operator would notice any single signal
alone. This multi-signal correlation is exactly what the VAE latent
space is designed to capture: it learns the JOINT distribution, not
each signal independently.

Design note: FaultInjector is SEPARATE from SensorSimulator.
This separation gives you clean ground truth labels and lets you
test fault detection in isolation from sensor noise.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional, Tuple

# ── Fault catalogue ───────────────────────────────────────────────────────────

FAULT_TYPES = [
    "efficiency_drop",   # Blade fouling, pitch misalignment
    "vibration_spike",   # Bearing wear (general vibration)
    "overheating",       # Cooling failure
    "bearing_fault",     # Main bearing degradation
    "gearbox_fault",     # Gearbox wear
]

# Severity of secondary effects relative to primary (tuned for realism)
_SECONDARY_SCALE = 0.25


@dataclass
class ActiveFault:
    """Represents a currently active fault event."""
    fault_type:       str
    remaining_steps:  int
    severity:         float    # 0.0–1.0, set at fault start, held constant


class FaultInjector:
    """
    Stateful fault injector. Call .step() each timestep, then
    call the relevant apply_to_*() methods on each signal.

    Args:
        fault_probability:        P(new fault starts | no active fault)
        fault_duration_seconds:   Steps a fault persists (at 1Hz)
        efficiency_drop_fraction: Power loss fraction for efficiency_drop
        enabled:                  Master on/off switch
        fault_start_after_step:   Ignore faults before this step (warm-up)
        seed:                     Random seed
    """

    def __init__(
        self,
        fault_probability:        float = 0.005,
        fault_duration_seconds:   int   = 120,
        efficiency_drop_fraction: float = 0.35,
        enabled:                  bool  = True,
        fault_start_after_step:   int   = 0,
        seed:                     int   = 99,
    ):
        self.fault_prob       = fault_probability
        self.fault_duration   = fault_duration_seconds
        self.efficiency_drop  = efficiency_drop_fraction
        self.enabled          = enabled
        self.fault_start_after_step = fault_start_after_step
        self._rng             = random.Random(seed)
        self._active_fault:   Optional[ActiveFault] = None
        self._current_step    = 0

    # ── State properties ──────────────────────────────────────────────────────

    @property
    def is_fault_active(self) -> bool:
        return self._active_fault is not None

    @property
    def active_fault_type(self) -> Optional[str]:
        return self._active_fault.fault_type if self._active_fault else None

    @property
    def active_severity(self) -> float:
        return self._active_fault.severity if self._active_fault else 0.0

    # ── State machine ─────────────────────────────────────────────────────────

    def step(self) -> Optional[ActiveFault]:
        """
        Advance the fault state machine by one timestep.
        Returns the active fault if one exists, else None.
        """
        self._current_step += 1

        if not self.enabled:
            return None

        # Respect warm-up period
        if self._current_step < self.fault_start_after_step:
            return None

        # Tick down active fault
        if self._active_fault is not None:
            self._active_fault.remaining_steps -= 1
            if self._active_fault.remaining_steps <= 0:
                self._active_fault = None

        # Possibly start a new fault (only if none active)
        if self._active_fault is None:
            if self._rng.random() < self.fault_prob:
                fault_type = self._rng.choice(FAULT_TYPES)
                severity   = self._rng.uniform(0.5, 1.0)
                self._active_fault = ActiveFault(
                    fault_type=fault_type,
                    remaining_steps=self.fault_duration,
                    severity=severity,
                )

        return self._active_fault

    # ── Apply methods — one per signal group ──────────────────────────────────
    #
    # Convention:
    #   PRIMARY effect  → large, always present for that fault type
    #   SECONDARY effect → subtle correlation, present at _SECONDARY_SCALE
    #
    # If a fault has no effect on a signal: return base unchanged.

    def apply_to_power(self, base_power_kw: float) -> float:
        """
        Power output affected by:
          efficiency_drop → PRIMARY: large reduction (blade/pitch fault)
          vibration_spike → secondary: small loss (energy to oscillation)
          overheating     → secondary: thermal throttling
          bearing_fault   → secondary: slight friction loss
          gearbox_fault   → secondary: gear mesh inefficiency
        """
        f = self._active_fault
        if f is None:
            return base_power_kw

        s = f.severity
        if f.fault_type == "efficiency_drop":
            drop = self.efficiency_drop * s
            return base_power_kw * (1.0 - drop)
        elif f.fault_type == "vibration_spike":
            return base_power_kw * (1.0 - 0.08 * s)
        elif f.fault_type == "overheating":
            return base_power_kw * (1.0 - 0.20 * s)
        elif f.fault_type == "bearing_fault":
            return base_power_kw * (1.0 - 0.05 * s)   # Small friction loss
        elif f.fault_type == "gearbox_fault":
            return base_power_kw * (1.0 - 0.10 * s)   # Gear mesh loss
        return base_power_kw

    def apply_to_rotor_rpm(self, base_rpm: float) -> float:
        """
        RPM affected by:
          gearbox_fault → torque ripple causes slight speed instability
          bearing_fault → friction causes marginal speed reduction
        """
        f = self._active_fault
        if f is None:
            return base_rpm

        s = f.severity
        if f.fault_type == "gearbox_fault":
            # Add RPM noise — torque ripple creates speed fluctuation
            ripple = self._rng.gauss(0, 0.3 * s)
            return max(0.0, base_rpm + ripple)
        elif f.fault_type == "bearing_fault":
            return base_rpm * (1.0 - 0.03 * s)
        return base_rpm

    def apply_to_torque(self, base_torque_nm: float) -> float:
        """
        Torque affected by:
          gearbox_fault → PRIMARY: torque ripple + mean reduction
          bearing_fault → secondary: slight increase (friction load)
        """
        f = self._active_fault
        if f is None:
            return base_torque_nm

        s = f.severity
        if f.fault_type == "gearbox_fault":
            # Torque ripple at gear mesh frequency — captured by VAE
            ripple = self._rng.gauss(0, base_torque_nm * 0.05 * s)
            reduction = base_torque_nm * 0.08 * s
            return max(0.0, base_torque_nm - reduction + ripple)
        elif f.fault_type == "bearing_fault":
            # Friction adds resistance torque — generator sees higher load
            return base_torque_nm * (1.0 + 0.04 * s)
        return base_torque_nm

    def apply_to_pitch(
        self,
        base_pitch_deg: float,
        rng:            random.Random,
    ) -> Tuple[float, float, float]:
        """
        Returns (pitch1, pitch2, pitch3) with fault effects.

        efficiency_drop → PRIMARY: pitch asymmetry (one blade misaligned)
        All others:     natural ±0.1° per-blade scatter only.

        💡 Industrial insight: Pitch asymmetry is one of the most common
        early fault signatures. A blade accumulating ice or biofilm
        changes its aerodynamic profile, requiring a different pitch angle
        to maintain balance. Operators at Ørsted use pitch asymmetry
        (max - min across 3 blades) as a leading indicator that triggers
        a maintenance inspection within 2 weeks.
        """
        f = self._active_fault

        # Natural per-blade manufacturing scatter (always present)
        scatter = [rng.gauss(0, 0.05) for _ in range(3)]

        if f is not None and f.fault_type == "efficiency_drop":
            # One blade is misaligned — which one is random but fixed per fault
            # Use severity to control asymmetry magnitude
            s = f.severity
            # Blade 1 gets the fault (consistent for this fault instance)
            asymmetry = s * 3.5   # Up to 3.5° at full severity
            return (
                base_pitch_deg + asymmetry + scatter[0],
                base_pitch_deg + scatter[1],
                base_pitch_deg + scatter[2],
            )

        # Healthy: all three blades track the same base angle ± scatter
        return (
            base_pitch_deg + scatter[0],
            base_pitch_deg + scatter[1],
            base_pitch_deg + scatter[2],
        )

    def apply_to_nacelle_temp(self, base_temp_c: float) -> float:
        """
        Nacelle temp affected by:
          overheating → PRIMARY: cooling failure → large rise
          gearbox_fault → secondary: heat from gear mesh losses
        """
        f = self._active_fault
        if f is None:
            return base_temp_c

        s = f.severity
        if f.fault_type == "overheating":
            return base_temp_c + 18.0 * s
        elif f.fault_type == "gearbox_fault":
            return base_temp_c + 4.0 * s * _SECONDARY_SCALE
        return base_temp_c

    def apply_to_generator_temp(self, base_temp_c: float) -> float:
        """
        Generator temp affected by:
          overheating  → PRIMARY: winding temp rises significantly
          bearing_fault → secondary: heat from bearing friction
        """
        f = self._active_fault
        if f is None:
            return base_temp_c

        s = f.severity
        if f.fault_type == "overheating":
            return base_temp_c + 22.0 * s
        elif f.fault_type == "bearing_fault":
            return base_temp_c + 6.0 * s * _SECONDARY_SCALE
        return base_temp_c

    def apply_to_bearing_temps(
        self, fore_temp_c: float, aft_temp_c: float
    ) -> Tuple[float, float]:
        """
        Bearing temperatures affected by:
          bearing_fault → PRIMARY: both rise, fore > aft (thrust side)
          gearbox_fault → secondary: heat conducts through shaft
          overheating   → secondary: ambient heat in nacelle conducts in
        """
        f = self._active_fault
        if f is None:
            return fore_temp_c, aft_temp_c

        s = f.severity
        if f.fault_type == "bearing_fault":
            # Fore bearing takes more thrust load — hotter during fault
            return (
                fore_temp_c + 14.0 * s,
                aft_temp_c  + 8.0  * s,
            )
        elif f.fault_type == "gearbox_fault":
            return (
                fore_temp_c + 3.0 * s * _SECONDARY_SCALE,
                aft_temp_c  + 2.0 * s * _SECONDARY_SCALE,
            )
        elif f.fault_type == "overheating":
            return (
                fore_temp_c + 2.0 * s * _SECONDARY_SCALE,
                aft_temp_c  + 2.0 * s * _SECONDARY_SCALE,
            )
        return fore_temp_c, aft_temp_c

    def apply_to_gearbox_temp(self, base_temp_c: float) -> float:
        """
        Gearbox oil temp affected by:
          gearbox_fault → PRIMARY: increased friction → oil heats up
          overheating   → secondary: ambient heat in nacelle conducts in
        """
        f = self._active_fault
        if f is None:
            return base_temp_c

        s = f.severity
        if f.fault_type == "gearbox_fault":
            return base_temp_c + 20.0 * s
        elif f.fault_type == "overheating":
            return base_temp_c + 4.0 * s * _SECONDARY_SCALE
        return base_temp_c

    def apply_to_hydraulic(self, base_pressure_bar: float) -> float:
        """
        Hydraulic pressure affected by:
          efficiency_drop → pitch asymmetry demands more from one actuator
          bearing_fault  → slight pressure fluctuation (shaft coupling)
        """
        f = self._active_fault
        if f is None:
            return base_pressure_bar

        s = f.severity
        if f.fault_type == "efficiency_drop":
            # One actuator working harder → slight pressure drop
            return base_pressure_bar - 8.0 * s
        elif f.fault_type == "bearing_fault":
            fluctuation = self._rng.gauss(0, 3.0 * s)
            return max(100.0, base_pressure_bar + fluctuation)
        return base_pressure_bar