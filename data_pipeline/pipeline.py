"""
data_pipeline/pipeline.py
==========================
Thin pipeline layer — validates and routes data between modules.

In production this handles:
  - Data validation
  - Buffering for bursty sensor input
  - Protocol adapters (MQTT, OPC-UA, REST → internal schema)
  - Dead-letter queue for bad readings

Keeping it here as a defined module means a clear insertion point
for all of the above without touching any other module.
"""

from __future__ import annotations
from typing import Optional
import csv
import io

from data_pipeline.schemas import TwinSnapshot, SensorReading, WindReading


def validate_wind_reading(reading: WindReading) -> Optional[str]:
    """Return error message if invalid, else None."""
    if reading.wind_speed_ms < 0:
        return f"Negative wind speed: {reading.wind_speed_ms}"
    if reading.wind_speed_ms > 100:
        return f"Implausible wind speed: {reading.wind_speed_ms} m/s"
    return None


def validate_sensor_reading(reading: SensorReading) -> Optional[str]:
    """Return error message if invalid, else None."""
    if reading.active_power_kw < 0:          # fixed: was actual_power_kw
        return f"Negative power reading: {reading.active_power_kw}"
    if reading.nacelle_temp_c > 150:
        return f"Implausibly high temperature: {reading.nacelle_temp_c}C"
    return None


def snapshots_to_csv(snapshots: list[TwinSnapshot]) -> str:
    """Serialise TwinSnapshots to CSV string for export and offline analysis."""
    if not snapshots:
        return ""

    output = io.StringIO()
    fieldnames = [
        "timestamp", "wind_speed_ms", "expected_power_kw",
        "actual_power_kw", "deviation_kw", "deviation_pct",
        "efficiency_ratio", "is_anomaly", "anomaly_score", "fault_type",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for s in snapshots:
        writer.writerow({
            "timestamp":        s.timestamp.isoformat(),
            "wind_speed_ms":    s.wind_speed_ms,
            "expected_power_kw": s.expected_power_kw,
            "actual_power_kw":  s.actual_power_kw,
            "deviation_kw":     s.deviation_kw,
            "deviation_pct":    s.deviation_pct,
            "efficiency_ratio": s.efficiency_ratio,
            "is_anomaly":       s.is_anomaly,
            "anomaly_score":    s.anomaly_score,
            "fault_type":       s.fault_type or "",
        })

    return output.getvalue()