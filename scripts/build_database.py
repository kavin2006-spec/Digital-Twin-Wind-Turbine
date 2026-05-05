"""
scripts/build_database.py
==========================
Builds a SQLite database from CARE to Compare Wind Farm A data.

Run:
  python scripts/build_database.py --data_dir data/raw/care_to_compare/wind_farm_A

Creates: data/processed/wind_farm_a.db

Schema:
  turbine_readings  -- all SCADA rows, normalised columns
  fault_events      -- from event_info.csv
  turbine_index     -- one row per CSV file (asset + date range + stats)
  feature_map       -- sensor_N -> description mapping

Why SQLite:
  - Zero setup, file-based, works everywhere
  - Query across 95 CSV files without loading all into memory
  - JOIN turbine_readings with fault_events on asset_id + timestamp
  - Filter by train_test split in SQL
  - Fast enough for 50k-200k rows (Wind Farm A size)
"""

from __future__ import annotations
import argparse
import glob
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.scada_loader import ScadaLoader, COLUMN_MAP, WIND_FARM_A_SEP


def build_database(
    data_dir:   str,
    output_db:  str,
    wind_farm:  str = "A",
) -> None:
    """
    Load all Wind Farm A CSV files and write to SQLite.

    Args:
        data_dir:  Path to wind_farm_A/ directory (contains datasets/ subfolder)
        output_db: Output .db file path
        wind_farm: Which wind farm (A, B, C) — only A supported currently
    """
    data_dir  = Path(data_dir)
    output_db = Path(output_db)
    output_db.parent.mkdir(parents=True, exist_ok=True)

    datasets_dir = data_dir / "datasets"
    event_file   = data_dir / "event_info.csv"
    feature_file = data_dir / "feature_description.csv"

    if not datasets_dir.exists():
        print(f"ERROR: datasets folder not found at {datasets_dir}")
        print(f"Expected structure:")
        print(f"  {data_dir}/")
        print(f"    datasets/      <- CSV files go here")
        print(f"    event_info.csv")
        print(f"    feature_description.csv")
        sys.exit(1)

    csv_files = sorted(glob.glob(str(datasets_dir / "*.csv")))
    if not csv_files:
        print(f"ERROR: No CSV files found in {datasets_dir}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Building Wind Farm {wind_farm} database")
    print(f"  Source:  {datasets_dir}")
    print(f"  Output:  {output_db}")
    print(f"  Files:   {len(csv_files)} CSV files")
    print(f"{'='*55}\n")

    conn = sqlite3.connect(output_db)

    # ── 1. Feature map table ──────────────────────────────────────────────────
    if feature_file.exists():
        print("Loading feature descriptions...")
        feat_df = pd.read_csv(feature_file, sep=WIND_FARM_A_SEP, on_bad_lines="skip")
        feat_df.to_sql("feature_map", conn, if_exists="replace", index=False)
        print(f"  {len(feat_df)} features mapped")
    else:
        print(f"WARNING: feature_description.csv not found at {feature_file}")

    # ── 2. Fault events table ─────────────────────────────────────────────────
    if event_file.exists():
        print("Loading fault events...")
        evt_df = pd.read_csv(event_file, sep=WIND_FARM_A_SEP, on_bad_lines="skip")
        evt_df.to_sql("fault_events", conn, if_exists="replace", index=False)
        print(f"  {len(evt_df)} fault events loaded")
    else:
        print(f"WARNING: event_info.csv not found at {event_file}")
        evt_df = pd.DataFrame()

    # ── 3. Main readings table ────────────────────────────────────────────────
    print("\nLoading SCADA data files...")
    index_rows = []
    total_rows = 0

    for i, path in enumerate(csv_files):
        filename = Path(path).name
        print(f"  [{i+1:3d}/{len(csv_files)}] {filename}", end="", flush=True)

        try:
            df = pd.read_csv(
                path,
                sep=WIND_FARM_A_SEP,
                parse_dates=["time_stamp"],
                low_memory=False,
                on_bad_lines="skip",
            )

            if len(df) == 0:
                print(" — empty, skipped")
                continue

            # Rename to our schema
            df = df.rename(columns=COLUMN_MAP)

            # Add source file info
            df["source_file"] = filename
            df["wind_farm"]   = wind_farm

            # Derive rotor RPM
            if "generator_speed_rpm" in df.columns:
                df["rotor_speed_rpm"] = (
                    df["generator_speed_rpm"] / 97.0
                ).clip(lower=0)

            # Generator temp = average of 3 phases
            phase_cols = [c for c in [
                "gen_temp_phase1_c", "gen_temp_phase2_c", "gen_temp_phase3_c"
            ] if c in df.columns]
            if phase_cols:
                df["generator_temp_c"] = df[phase_cols].mean(axis=1)

            # Add fault labels
            df["is_fault"]    = False
            df["fault_label"] = None

            if not evt_df.empty and "asset_id" in df.columns:
                for asset_id in df["asset_id"].unique():
                    asset_events = evt_df[evt_df["asset"] == asset_id]
                    for _, evt in asset_events.iterrows():
                        try:
                            start = pd.to_datetime(evt["event_start"])
                            end   = pd.to_datetime(evt["event_end"])
                            mask  = (
                                (df["asset_id"] == asset_id)
                                & (df["timestamp"] >= start)
                                & (df["timestamp"] <= end)
                            )
                            df.loc[mask, "is_fault"]    = True
                            df.loc[mask, "fault_label"] = str(
                                evt.get("event_label", "unknown")
                            )
                        except Exception:
                            continue

            # Write to SQLite
            df.to_sql(
                "turbine_readings",
                conn,
                if_exists="append",
                index=False,
            )

            # Build index row
            asset_id = df["asset_id"].iloc[0] if "asset_id" in df.columns else "unknown"
            split    = df["split"].iloc[0] if "split" in df.columns else "unknown"
            n_fault  = int(df["is_fault"].sum())
            index_rows.append({
                "filename":   filename,
                "asset_id":   asset_id,
                "split":      split,
                "n_rows":     len(df),
                "n_fault":    n_fault,
                "ts_start":   str(df["timestamp"].min()) if "timestamp" in df.columns else "",
                "ts_end":     str(df["timestamp"].max()) if "timestamp" in df.columns else "",
                "power_mean": float(df["active_power_kw"].mean()) if "active_power_kw" in df.columns else 0.0,
                "power_max":  float(df["active_power_kw"].max())  if "active_power_kw" in df.columns else 0.0,
            })

            total_rows += len(df)
            print(f" — {len(df):,} rows, {n_fault} fault rows")

        except Exception as e:
            print(f" — ERROR: {e}")
            continue

    # ── 4. Turbine index table ────────────────────────────────────────────────
    if index_rows:
        idx_df = pd.DataFrame(index_rows)
        idx_df.to_sql("turbine_index", conn, if_exists="replace", index=False)

    # ── 5. Create indexes for fast querying ───────────────────────────────────
    print("\nCreating database indexes...")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_timestamp "
        "ON turbine_readings(timestamp)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_asset "
        "ON turbine_readings(asset_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_split "
        "ON turbine_readings(split)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fault "
        "ON turbine_readings(is_fault)"
    )
    conn.commit()

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Database built successfully")
    print(f"  File:       {output_db}")
    print(f"  Size:       {output_db.stat().st_size / 1e6:.1f} MB")
    print(f"  Total rows: {total_rows:,}")
    fault_total = sum(r["n_fault"] for r in index_rows)
    print(f"  Fault rows: {fault_total:,} ({100*fault_total/max(total_rows,1):.1f}%)")

    # Print useful queries
    print(f"\n  Useful SQL queries:")
    print(f"  -- Normal training rows:")
    print(f"     SELECT * FROM turbine_readings")
    print(f"     WHERE split='train' AND is_fault=0")
    print(f"     AND status_type_id=0")
    print(f"  -- Fault events with context:")
    print(f"     SELECT r.*, e.event_label")
    print(f"     FROM turbine_readings r")
    print(f"     JOIN fault_events e ON r.asset_id=e.asset")
    print(f"     WHERE r.is_fault=1")
    print(f"  -- Per-turbine fault summary:")
    print(f"     SELECT asset_id, COUNT(*) as fault_rows")
    print(f"     FROM turbine_readings WHERE is_fault=1")
    print(f"     GROUP BY asset_id ORDER BY fault_rows DESC")
    print(f"{'='*55}\n")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build SQLite database from CARE to Compare Wind Farm A data"
    )
    parser.add_argument(
        "--data_dir",
        default="data/raw/care_to_compare/wind_farm_A",
        help="Path to wind_farm_A directory",
    )
    parser.add_argument(
        "--output",
        default="data/processed/wind_farm_a.db",
        help="Output SQLite database path",
    )
    args = parser.parse_args()
    build_database(args.data_dir, args.output)
