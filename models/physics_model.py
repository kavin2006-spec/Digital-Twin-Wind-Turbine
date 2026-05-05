"""
models/physics_model.py
=======================
Physics-based power curve model for a horizontal-axis wind turbine.

Design decisions:
  - Uses the IEC 61400-12 standard power curve shape (industry standard)
  - Three operating regions: idle, ramp-up, rated/shutdown
  - Parameterised via TurbineConfig dataclass — no magic numbers in code
  - Pure functions where possible: same input → same output, easy to test

Physics background:
  The theoretical power extractable from wind is:
      P = 0.5 * rho * A * v^3 * Cp
  where:
      rho = air density (kg/m³)
      A   = rotor swept area (m²) = pi * r²
      v   = wind speed (m/s)
      Cp  = power coefficient (dimensionless, max 0.593 by Betz law)

  Real turbines use a "power curve" that maps wind speed → power.
  We implement this as a piecewise cubic function matching that physics.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class TurbinePhysicsConfig:
    """All physical parameters needed by the power curve model."""
    name: str
    rated_power_kw: float
    cut_in_speed_ms: float
    rated_speed_ms: float
    cut_out_speed_ms: float
    rotor_diameter_m: float
    air_density_kg_m3: float
    power_coefficient_cp: float

    @classmethod
    def from_yaml(cls, path: str) -> "TurbinePhysicsConfig":
        """Load config from a YAML file. Keeps instantiation flexible."""
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        t = cfg["turbine"]
        return cls(
            name=t["name"],
            rated_power_kw=t["rated_power_kw"],
            cut_in_speed_ms=t["cut_in_speed_ms"],
            rated_speed_ms=t["rated_speed_ms"],
            cut_out_speed_ms=t["cut_out_speed_ms"],
            rotor_diameter_m=t["rotor_diameter_m"],
            air_density_kg_m3=t["air_density_kg_m3"],
            power_coefficient_cp=t["power_coefficient_cp"],
        )

    @property
    def rotor_swept_area_m2(self) -> float:
        """Swept area of the rotor disc (m²)."""
        radius = self.rotor_diameter_m / 2.0
        return math.pi * radius ** 2

    @property
    def theoretical_max_power_kw(self) -> float:
        """
        Maximum extractable power at rated wind speed using Betz formula.
        Useful as a sanity check: rated_power_kw should be ≤ this.
        """
        P_watts = (
            0.5
            * self.air_density_kg_m3
            * self.rotor_swept_area_m2
            * self.rated_speed_ms ** 3
            * self.power_coefficient_cp
        )
        return P_watts / 1000.0  # Convert W → kW


class WindTurbinePhysicsModel:
    """
    Implements the IEC power curve: wind speed → expected power output.

    Operating regions:
        Region 1  (v < cut_in):         Turbine is idle.   P = 0
        Region 2  (cut_in ≤ v < rated): Turbine ramps up.  P ∝ v³  (cubic)
        Region 3  (rated ≤ v < cut_out):Rated power.       P = P_rated (constant)
        Region 4  (v ≥ cut_out):        Shutdown.          P = 0  (storm protection)

    Design decision: cubic interpolation in Region 2 is physically correct
    because wind power scales with v³. We normalise it so the curve hits
    exactly rated_power_kw at rated_speed_ms.
    """

    def __init__(self, config: TurbinePhysicsConfig):
        self.config = config
        # Precompute the cubic scaling factor for Region 2
        # At v = rated_speed, the cubic term should give rated_power_kw
        self._cubic_scale = self.config.rated_power_kw / (
            self.config.rated_speed_ms ** 3 - self.config.cut_in_speed_ms ** 3
        )

    def expected_power_kw(self, wind_speed_ms: float) -> float:
        """
        Compute expected turbine power output for a given wind speed.

        Args:
            wind_speed_ms: Wind speed at hub height (m/s). Must be ≥ 0.

        Returns:
            Expected power output in kilowatts (kW).

        Raises:
            ValueError: If wind speed is negative (physically impossible).
        """
        if wind_speed_ms < 0:
            raise ValueError(f"Wind speed cannot be negative: {wind_speed_ms}")

        v = wind_speed_ms
        cfg = self.config

        # Region 1: Below cut-in — turbine is idle
        if v < cfg.cut_in_speed_ms:
            return 0.0

        # Region 4: Above cut-out — turbine shuts down (storm protection)
        if v >= cfg.cut_out_speed_ms:
            return 0.0

        # Region 3: At or above rated speed — clamp to rated power
        if v >= cfg.rated_speed_ms:
            return cfg.rated_power_kw

        # Region 2: Between cut-in and rated — cubic ramp up
        # P(v) = scale * (v³ - v_cutin³)
        # This naturally gives P=0 at cut_in and P=rated at rated_speed
        power = self._cubic_scale * (v ** 3 - cfg.cut_in_speed_ms ** 3)
        return max(0.0, min(power, cfg.rated_power_kw))

    def power_curve_table(
        self, v_min: float = 0.0, v_max: float = 30.0, steps: int = 100
    ) -> list[tuple[float, float]]:
        """
        Generate a (wind_speed, power) lookup table for visualisation.

        Useful for plotting the theoretical power curve before running simulation.

        Returns:
            List of (wind_speed_ms, power_kw) tuples.
        """
        step_size = (v_max - v_min) / steps
        return [
            (v_min + i * step_size, self.expected_power_kw(v_min + i * step_size))
            for i in range(steps + 1)
        ]

    def capacity_factor(self, wind_speed_ms: float) -> float:
        """
        Capacity factor: actual output / rated output (0.0 to 1.0).

        Useful as a normalised efficiency metric.
        """
        return self.expected_power_kw(wind_speed_ms) / self.config.rated_power_kw

    def __repr__(self) -> str:
        return (
            f"WindTurbinePhysicsModel("
            f"name={self.config.name!r}, "
            f"rated={self.config.rated_power_kw}kW, "
            f"cut_in={self.config.cut_in_speed_ms}m/s, "
            f"rated_speed={self.config.rated_speed_ms}m/s, "
            f"cut_out={self.config.cut_out_speed_ms}m/s)"
        )
