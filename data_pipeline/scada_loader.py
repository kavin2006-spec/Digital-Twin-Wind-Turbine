"""
data_pipeline/scada_loader.py
==============================
Loads CARE to Compare Wind Farm A SCADA data into our schema.

Column mapping (Wind Farm A feature_description.csv):
  wind_speed_3_avg      -> wind_speed_ms
  wind_speed_3_std      -> wind_std (Tier 3)
  sensor_0_avg          -> ambient_temp_c
  sensor_1_avg          -> wind_direction_deg
  sensor_5_avg          -> pitch_angle (single channel, all 3 blades)
  sensor_11_avg         -> bearing_temp_fore_c (gearbox bearing, high speed shaft)
  sensor_12_avg         -> gearbox_oil_temp_c
  sensor_13_avg         -> bearing_temp_fore_c (generator bearing DE)
  sensor_14_avg         -> bearing_temp_aft_c  (generator bearing NDE)
  sensor_15/16/17_avg   -> generator_temp_c    (average of 3 stator phases)
  sensor_18_avg         -> generator_speed_rpm
  sensor_26_avg         -> grid_frequency_hz
  reactive_power_27_avg -> reactive_power_kvar
  power_29_avg          -> active_power_kw
  status_type_id        -> operational status filter
  train_test            -> train/test split label

Missing from real data (vs simulation):
  rotor_speed_rpm       -> derived from generator_speed_rpm / gear_ratio
  torque_nm             -> not available, set to 0.0
  hydraulic_pressure    -> not available, set to 0.0
  pitch_angle_blade2/3  -> not available, copy blade1

Data cadence: 10-minute averages (not 1Hz like simulation).
Rolling window features must account for this:
  60s window @ 1Hz   = 60 steps
  60s window @ 1/600Hz = 0.1 steps (use 1 step minimum)
  600s window @ 10min = 1 step
  3600s window @ 10min = 6 steps

💡 Industrial insight: The 10-minute SCADA averaging is an IEC 61400-12
standard requirement for power performance measurements. All utility-scale
turbines log at this resolution for grid compliance. Your simulation at
1Hz is higher resolution — when ingesting real data you're working with
pre-averaged signals, which smooths out transient faults but makes
gradual degradation more visible.
"""

from __future__ import annotations
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator
import pandas as pd
import numpy as np

from data_pipeline.schemas import SensorReading


# ── Constants ─────────────────────────────────────────────────────────────────

GEAR_RATIO    = 97.0    # NREL 5MW approximate — used to derive rotor RPM
WIND_FARM_A_SEP = ";"   # CSV delimiter for Wind Farm A

# Status type IDs that represent normal turbine operation
# (turbine producing power, not in maintenance/curtailment/fault)
# Values 0-2 are typically: 0=ok, 1=warning, 2=fault/stop
# We keep only status 0 for training
NORMAL_STATUS_IDS = {0}   # adjust if you find other valid operating states

# Column mapping: CARE column -> our internal name
COLUMN_MAP = {
    "wind_speed_3_avg":       "wind_speed_ms",
    "wind_speed_3_std":       "wind_speed_std",
    "wind_speed_4_avg":       "wind_speed_estimated_ms",
    "sensor_0_avg":           "ambient_temp_c",
    "sensor_1_avg":           "wind_direction_deg",
    "sensor_5_avg":           "pitch_angle_deg",
    "sensor_11_avg":          "gearbox_bearing_temp_c",
    "sensor_12_avg":          "gearbox_oil_temp_c",
    "sensor_13_avg":          "bearing_temp_fore_c",
    "sensor_14_avg":          "bearing_temp_aft_c",
    "sensor_15_avg":          "gen_temp_phase1_c",
    "sensor_16_avg":          "gen_temp_phase2_c",
    "sensor_17_avg":          "gen_temp_phase3_c",
    "sensor_18_avg":          "generator_speed_rpm",
    "sensor_26_avg":          "grid_frequency_hz",
    "reactive_power_27_avg":  "reactive_power_kvar",
    "power_29_avg":           "active_power_kw",
    "power_29_std":           "active_power_std",
    "status_type_id":         "status_type_id",
    "train_test":             "split",
    "time_stamp":             "timestamp",
    "asset_id":               "asset_id",
}


class ScadaLoader:
    """
    Loads CARE to Compare Wind Farm A CSV files into SensorReading objects.

    Args:
        data_dir:   Path to wind_farm_A/datasets/ directory
        event_file: Path to wind_farm_A/event_info.csv
        split:      "train" or "test" — which rows to load
        normal_only: If True, filter to normal operation rows only
    """

    def __init__(
        self,
        data_dir:    str,
        event_file:  str,
        split:       str  = "train",
        normal_only: bool = True,
    ):
        self.data_dir    = Path(data_dir)
        self.split       = split
        self.normal_only = normal_only

        # Load event info for fault labels
        self.events = self._load_events(event_file)

        # Discover all CSV files
        self.csv_files = sorted(
            glob.glob(str(self.data_dir / "*.csv"))
        )
        print(f"[ScadaLoader] Found {len(self.csv_files)} dataset files")
        print(f"[ScadaLoader] Split: {split}, Normal only: {normal_only}")

    # ── Public interface ──────────────────────────────────────────────────────

    def load_all(self) -> pd.DataFrame:
        """
        Load all CSV files, apply filters, return combined DataFrame.
        Adds 'is_fault' column from event_info labels.
        """
        dfs = []
        for path in self.csv_files:
            df = self._load_single(path)
            if df is not None and len(df) > 0:
                dfs.append(df)

        if not dfs:
            print("[ScadaLoader] No data loaded — check path and filters")
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        print(f"[ScadaLoader] Loaded {len(combined):,} rows from "
              f"{len(dfs)} files")
        print(f"  Fault rows:  {combined['is_fault'].sum():,} "
              f"({100*combined['is_fault'].mean():.1f}%)")
        print(f"  Normal rows: {(~combined['is_fault']).sum():,}")
        return combined

    def load_single_file(self, filename: str) -> Optional[pd.DataFrame]:
        """Load one specific CSV file by name."""
        path = self.data_dir / filename
        if not path.exists():
            print(f"[ScadaLoader] File not found: {path}")
            return None
        return self._load_single(str(path))

    def iter_sensor_readings(
        self, df: pd.DataFrame
    ) -> Iterator[tuple[SensorReading, bool]]:
        """
        Yield (SensorReading, is_fault) for each row in a loaded DataFrame.
        Used to feed real data through the twin pipeline.
        """
        for _, row in df.iterrows():
            yield self._row_to_sensor_reading(row), bool(row["is_fault"])

    def summary(self, df: pd.DataFrame) -> None:
        """Print dataset summary statistics."""
        if df.empty:
            print("[ScadaLoader] Empty dataframe")
            return

        print(f"\n{'='*50}")
        print(f"  CARE to Compare — Wind Farm A Summary")
        print(f"{'='*50}")
        print(f"  Rows:          {len(df):,}")
        print(f"  Turbines:      {df['asset_id'].nunique()}")
        print(f"  Date range:    {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(f"  Fault rows:    {df['is_fault'].sum():,} ({100*df['is_fault'].mean():.1f}%)")
        print(f"\n  Power stats:")
        print(f"    Mean:  {df['active_power_kw'].mean():.1f} kW")
        print(f"    Max:   {df['active_power_kw'].max():.1f} kW")
        print(f"    Zeros: {(df['active_power_kw'] <= 0).sum():,} rows (curtailed/stopped)")
        print(f"\n  Wind speed:")
        print(f"    Mean:  {df['wind_speed_ms'].mean():.1f} m/s")
        print(f"    Max:   {df['wind_speed_ms'].max():.1f} m/s")
        print(f"{'='*50}\n")

    # ── Internal loading ──────────────────────────────────────────────────────

    def _load_single(self, path: str) -> Optional[pd.DataFrame]:
        """Load one CSV file, apply column mapping and filters."""
        try:
            df = pd.read_csv(
                path,
                sep=WIND_FARM_A_SEP,
                parse_dates=["time_stamp"],
                low_memory=False,
            )
        except Exception as e:
            print(f"[ScadaLoader] Failed to load {path}: {e}")
            return None

        # Apply split filter
        if "train_test" in df.columns:
            df = df[df["train_test"] == self.split]

        # Apply normal operation filter
        if self.normal_only and "status_type_id" in df.columns:
            df = df[df["status_type_id"].isin(NORMAL_STATUS_IDS)]

        if len(df) == 0:
            return None

        # Rename columns
        df = df.rename(columns=COLUMN_MAP)

        # Derive missing fields
        df = self._derive_fields(df)

        # Add fault labels from event info
        df = self._add_fault_labels(df)

        # Keep only columns we use
        keep = list(COLUMN_MAP.values()) + [
            "rotor_speed_rpm", "generator_temp_c",
            "is_fault", "fault_label",
        ]
        keep = [c for c in keep if c in df.columns]
        return df[keep]

    def _derive_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute fields not directly available in CARE data."""

        # Rotor RPM from generator RPM / gear ratio
        if "generator_speed_rpm" in df.columns:
            df["rotor_speed_rpm"] = (
                df["generator_speed_rpm"] / GEAR_RATIO
            ).clip(lower=0)
        else:
            df["rotor_speed_rpm"] = 0.0

        # Generator temp = average of 3 stator phase temperatures
        phase_cols = [c for c in ["gen_temp_phase1_c", "gen_temp_phase2_c",
                                   "gen_temp_phase3_c"] if c in df.columns]
        if phase_cols:
            df["generator_temp_c"] = df[phase_cols].mean(axis=1)
        else:
            df["generator_temp_c"] = 60.0

        # Pitch angles: only one channel available — copy to all 3
        if "pitch_angle_deg" in df.columns:
            df["pitch_angle_blade1_deg"] = df["pitch_angle_deg"]
            df["pitch_angle_blade2_deg"] = df["pitch_angle_deg"]
            df["pitch_angle_blade3_deg"] = df["pitch_angle_deg"]

        # Fill missing columns with sensible defaults
        defaults = {
            "torque_nm":            0.0,
            "hydraulic_pressure_bar": 180.0,
            "cos_phi":              0.95,
            "grid_voltage_v":       690.0,
            "turbulence_intensity": 0.0,
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        return df

    def _add_fault_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mark rows as faults using event_info timestamps.
        A row is a fault if its timestamp falls within any event window
        for the same asset_id.
        """
        df["is_fault"]    = False
        df["fault_label"] = None

        if self.events.empty or "asset_id" not in df.columns:
            return df

        for asset_id in df["asset_id"].unique():
            asset_events = self.events[
                self.events["asset"] == asset_id
            ]
            if asset_events.empty:
                continue

            asset_mask = df["asset_id"] == asset_id
            for _, event in asset_events.iterrows():
                try:
                    start = pd.to_datetime(event["event_start"])
                    end   = pd.to_datetime(event["event_end"])
                    fault_mask = (
                        asset_mask
                        & (df["timestamp"] >= start)
                        & (df["timestamp"] <= end)
                    )
                    df.loc[fault_mask, "is_fault"]    = True
                    df.loc[fault_mask, "fault_label"] = str(
                        event.get("event_label", "unknown")
                    )
                except Exception:
                    continue

        return df

    def _load_events(self, event_file: str) -> pd.DataFrame:
        """Load event_info CSV with fault timestamps."""
        path = Path(event_file)
        if not path.exists():
            print(f"[ScadaLoader] Event file not found: {path}")
            return pd.DataFrame()
        try:
            events = pd.read_csv(path, sep=WIND_FARM_A_SEP)
            print(f"[ScadaLoader] Loaded {len(events)} fault events")
            return events
        except Exception as e:
            print(f"[ScadaLoader] Failed to load events: {e}")
            return pd.DataFrame()

    def _row_to_sensor_reading(self, row: pd.Series) -> SensorReading:
        """Convert one DataFrame row to a SensorReading."""
        def get(col: str, default: float = 0.0) -> float:
            val = row.get(col, default)
            if pd.isna(val):
                return default
            return float(val)

        return SensorReading(
            timestamp=pd.to_datetime(row.get("timestamp", datetime.utcnow())),
            wind_speed_ms=get("wind_speed_ms"),
            wind_direction_deg=get("wind_direction_deg"),
            turbulence_intensity=get("turbulence_intensity", 0.08),
            rotor_speed_rpm=get("rotor_speed_rpm"),
            pitch_angle_blade1_deg=get("pitch_angle_blade1_deg"),
            pitch_angle_blade2_deg=get("pitch_angle_blade2_deg"),
            pitch_angle_blade3_deg=get("pitch_angle_blade3_deg"),
            generator_speed_rpm=get("generator_speed_rpm"),
            torque_nm=get("torque_nm"),
            gearbox_oil_temp_c=get("gearbox_oil_temp_c", 55.0),
            active_power_kw=get("active_power_kw"),
            reactive_power_kvar=get("reactive_power_kvar"),
            grid_frequency_hz=get("grid_frequency_hz", 50.0),
            grid_voltage_v=get("grid_voltage_v", 690.0),
            cos_phi=get("cos_phi", 0.95),
            nacelle_temp_c=get("gen_temp_phase1_c", 45.0),
            generator_temp_c=get("generator_temp_c", 60.0),
            bearing_temp_fore_c=get("bearing_temp_fore_c", 42.0),
            bearing_temp_aft_c=get("bearing_temp_aft_c", 40.0),
            ambient_temp_c=get("ambient_temp_c", 15.0),
            hydraulic_pressure_bar=get("hydraulic_pressure_bar", 180.0),
            is_fault_injected=bool(row.get("is_fault", False)),
            fault_type=row.get("fault_label"),
        )