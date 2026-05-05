"""
data_pipeline/dataset_registry.py
====================================
Routes between simulated data and real CARE to Compare SCADA data.

Usage:
  registry = DatasetRegistry(db_path="data/processed/wind_farm_a.db")
  
  # Get training windows (normal operation only)
  train_df = registry.get_training_data()
  
  # Get evaluation data with fault labels
  eval_df = registry.get_evaluation_data()
  
  # Check what's available
  registry.summary()

Design decision: The registry is a thin query layer over the SQLite
database. All the heavy loading and column mapping is done once by
build_database.py. The registry just filters and returns DataFrames.
"""

from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Optional
import pandas as pd


class DatasetRegistry:
    """
    Query interface for the Wind Farm A SQLite database.

    Args:
        db_path: Path to wind_farm_a.db built by build_database.py
    """

    def __init__(self, db_path: str = "data/processed/wind_farm_a.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {db_path}\n"
                f"Run: python scripts/build_database.py --data_dir "
                f"data/raw/care_to_compare/wind_farm_A"
            )
        self._conn = sqlite3.connect(self.db_path)
        print(f"[DatasetRegistry] Connected to {db_path}")

    def get_training_data(
        self,
        normal_only:    bool = True,
        min_power_kw:   float = 10.0,
        status_type_id: int   = 0,
    ) -> pd.DataFrame:
        """
        Get training rows — normal operation, fault-free.

        Args:
            normal_only:    Only include rows where is_fault=False
            min_power_kw:   Exclude idle/stopped turbine rows
            status_type_id: Operational status filter (0 = normal)

        Returns:
            DataFrame of training rows ready for VAE training
        """
        conditions = ["split = 'train'"]
        if normal_only:
            conditions.append("is_fault = 0")
        if min_power_kw > 0:
            conditions.append(f"active_power_kw >= {min_power_kw}")
        if status_type_id is not None:
            conditions.append(f"status_type_id = {status_type_id}")

        where = " AND ".join(conditions)
        query = f"""
            SELECT * FROM turbine_readings
            WHERE {where}
            ORDER BY timestamp
        """
        df = pd.read_sql(query, self._conn, parse_dates=["timestamp"])
        print(f"[DatasetRegistry] Training data: {len(df):,} rows")
        return df

    def get_evaluation_data(
        self,
        include_normal: bool  = True,
        include_faults: bool  = True,
        min_power_kw:   float = 0.0,
    ) -> pd.DataFrame:
        """
        Get evaluation rows with fault labels.

        Returns test split rows with is_fault and fault_label columns.
        """
        conditions = ["split = 'test'"]
        if not include_normal:
            conditions.append("is_fault = 1")
        if not include_faults:
            conditions.append("is_fault = 0")
        if min_power_kw > 0:
            conditions.append(f"active_power_kw >= {min_power_kw}")

        where = " AND ".join(conditions)
        query = f"""
            SELECT * FROM turbine_readings
            WHERE {where}
            ORDER BY timestamp
        """
        df = pd.read_sql(query, self._conn, parse_dates=["timestamp"])
        print(f"[DatasetRegistry] Evaluation data: {len(df):,} rows "
              f"({df['is_fault'].sum():,} fault rows)")
        return df

    def get_fault_events(self) -> pd.DataFrame:
        """Return the fault_events table."""
        return pd.read_sql("SELECT * FROM fault_events", self._conn)

    def get_turbine_index(self) -> pd.DataFrame:
        """Return per-file summary index."""
        return pd.read_sql(
            "SELECT * FROM turbine_index ORDER BY asset_id", self._conn
        )

    def get_feature_map(self) -> pd.DataFrame:
        """Return sensor_N -> description mapping."""
        return pd.read_sql("SELECT * FROM feature_map", self._conn)

    def summary(self) -> None:
        """Print database contents summary."""
        try:
            total = pd.read_sql(
                "SELECT COUNT(*) as n FROM turbine_readings", self._conn
            ).iloc[0]["n"]
            fault = pd.read_sql(
                "SELECT COUNT(*) as n FROM turbine_readings WHERE is_fault=1",
                self._conn
            ).iloc[0]["n"]
            turbines = pd.read_sql(
                "SELECT COUNT(DISTINCT asset_id) as n FROM turbine_readings",
                self._conn
            ).iloc[0]["n"]
            train = pd.read_sql(
                "SELECT COUNT(*) as n FROM turbine_readings WHERE split='train'",
                self._conn
            ).iloc[0]["n"]
            test = pd.read_sql(
                "SELECT COUNT(*) as n FROM turbine_readings WHERE split='test'",
                self._conn
            ).iloc[0]["n"]

            print(f"\n{'='*45}")
            print(f"  Wind Farm A Database Summary")
            print(f"{'='*45}")
            print(f"  Total rows:   {total:,}")
            print(f"  Train rows:   {train:,}")
            print(f"  Test rows:    {test:,}")
            print(f"  Fault rows:   {fault:,} ({100*fault/max(total,1):.1f}%)")
            print(f"  Turbines:     {turbines}")
            print(f"{'='*45}\n")
        except Exception as e:
            print(f"[DatasetRegistry] Summary error: {e}")

    def close(self) -> None:
        self._conn.close()
