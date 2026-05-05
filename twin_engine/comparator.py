"""
twin_engine/comparator.py
==========================
Computes deviation metrics between expected (physics) and actual (sensor).

Design decision: Keep comparison logic STATELESS (pure functions).
The comparator doesn't know or care where the data came from —
it just computes metrics. This makes it trivially testable.

Deviation metrics we compute:
  - Absolute deviation (kW)
  - Percentage deviation (%)
  - Efficiency ratio (actual/expected, 1.0 = perfect)
  - Rolling statistics (mean, std dev over a window)
"""

from __future__ import annotations
from collections import deque
from typing import Optional
import math

from data_pipeline.schemas import SensorReading, TwinSnapshot


def compute_snapshot(
    sensor: SensorReading,
    expected_power_kw: float,
) -> TwinSnapshot:
    """
    Compute a TwinSnapshot from a sensor reading + physics expectation.

    Args:
        sensor:             Raw sensor reading (21 Tier 1 signals)
        expected_power_kw:  Physics model prediction for this wind speed

    Returns:
        TwinSnapshot with deviation metrics filled in.
    """
    actual = sensor.active_power_kw   # v1 used actual_power_kw — renamed in schema

    deviation_kw = actual - expected_power_kw

    if expected_power_kw > 1.0:
        deviation_pct    = (deviation_kw / expected_power_kw) * 100.0
        efficiency_ratio = actual / expected_power_kw
    else:
        deviation_pct    = 0.0
        efficiency_ratio = 1.0

    return TwinSnapshot(
        timestamp=sensor.timestamp,
        wind_speed_ms=sensor.wind_speed_ms,
        expected_power_kw=round(expected_power_kw, 3),
        actual_power_kw=round(actual, 3),
        deviation_kw=round(deviation_kw, 3),
        deviation_pct=round(deviation_pct, 2),
        efficiency_ratio=round(efficiency_ratio, 4),
        fault_type=sensor.fault_type,
    )


class RollingStatistics:
    """
    Maintains rolling window statistics over efficiency ratio values.

    Design decision: deque gives O(1) append/pop — important when
    processing thousands of samples per minute in production.

    Used by the twin engine to track baseline "normal" behaviour
    that the ML model compares new readings against.
    """

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self._buffer: deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> None:
        self._buffer.append(value)

    @property
    def mean(self) -> Optional[float]:
        if not self._buffer:
            return None
        return sum(self._buffer) / len(self._buffer)

    @property
    def std(self) -> Optional[float]:
        if len(self._buffer) < 2:
            return None
        m = self.mean
        variance = sum((x - m) ** 2 for x in self._buffer) / (len(self._buffer) - 1)
        return math.sqrt(variance)

    @property
    def z_score(self) -> Optional[float]:
        """
        Z-score of the most recent value vs the rolling window.
        z > 3.0 is the standard threshold for statistical anomaly detection.
        """
        if len(self._buffer) < 10 or self.std is None or self.std == 0:
            return None
        return (self._buffer[-1] - self.mean) / self.std

    def is_full(self) -> bool:
        return len(self._buffer) == self.window_size