"""
simulation/wind_generator.py
============================
Generates realistic wind speed time series.

Design decision: We use a stochastic model (mean-reverting with
turbulence) rather than pure random noise. This matches how real
wind behaves — it has momentum, doesn't jump wildly per second,
and varies around a daily mean.

Model used: Ornstein-Uhlenbeck process
  dv = theta * (mu - v) * dt + sigma * dW
  where:
    theta = mean-reversion rate (how fast wind returns to mean)
    mu    = long-term mean wind speed
    sigma = volatility (turbulence)
    dW    = Wiener process (random noise)

This gives wind series with:
  - Temporal autocorrelation (realistic momentum)
  - Bounded around a mean (no drift to infinity)
  - Controllable turbulence intensity
"""

from __future__ import annotations
import math
import random
from datetime import datetime, timedelta
from typing import Iterator

from data_pipeline.schemas import WindReading


class WindSpeedGenerator:
    """
    Generates correlated wind speed samples using Ornstein-Uhlenbeck process.
    
    Args:
        mean_speed_ms:     Long-term mean wind speed (m/s)
        turbulence_intensity: TI = sigma/mean (IEC standard, typically 0.05–0.20)
        dt_seconds:        Timestep in seconds
        seed:              Random seed for reproducibility
    """

    def __init__(
        self,
        mean_speed_ms: float = 9.0,
        turbulence_intensity: float = 0.12,
        dt_seconds: float = 1.0,
        seed: int = 42,
    ):
        self.mean = mean_speed_ms
        self.ti = turbulence_intensity
        self.dt = dt_seconds
        self._rng = random.Random(seed)

        # OU process parameters
        # theta: mean reversion speed — 0.1 means ~10s correlation time
        self._theta = 0.1
        # sigma: derived from turbulence intensity standard definition
        self._sigma = turbulence_intensity * mean_speed_ms

        # Start at mean
        self._current_speed = mean_speed_ms

    def next_speed(self) -> float:
        """
        Advance one timestep and return next wind speed.
        
        Uses Euler-Maruyama discretisation of the OU SDE.
        """
        drift = self._theta * (self.mean - self._current_speed) * self.dt
        diffusion = self._sigma * math.sqrt(self.dt) * self._rng.gauss(0, 1)
        self._current_speed = max(0.0, self._current_speed + drift + diffusion)
        return self._current_speed

    def generate_series(
        self,
        n_steps: int,
        start_time: datetime = None,
    ) -> list[WindReading]:
        """
        Generate a full time series of wind readings.
        
        Args:
            n_steps:    Number of timesteps to generate
            start_time: Starting timestamp (defaults to now)
        
        Returns:
            List of WindReading dataclass instances.
        """
        if start_time is None:
            start_time = datetime.utcnow()

        readings = []
        for i in range(n_steps):
            ts = start_time + timedelta(seconds=i * self.dt)
            speed = self.next_speed()
            readings.append(WindReading(timestamp=ts, wind_speed_ms=speed))

        return readings

    def stream(self, start_time: datetime = None) -> Iterator[WindReading]:
        """
        Infinite generator — yields one WindReading per call.
        Useful for real-time simulation mode.
        """
        if start_time is None:
            start_time = datetime.utcnow()
        t = start_time
        step = timedelta(seconds=self.dt)
        while True:
            yield WindReading(timestamp=t, wind_speed_ms=self.next_speed())
            t += step

    @property
    def current_speed(self) -> float:
        return self._current_speed
