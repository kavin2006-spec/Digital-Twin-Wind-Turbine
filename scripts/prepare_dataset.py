"""
scripts/prepare_dataset.py
===========================
Adds proper train/eval split to the Wind Farm A database.

Split strategy:
  Training set:   Turbines 0, 10, 11, 13 (normal operation only, status=0)
  Evaluation set: Turbine 21 (all rows including faults)

Why split by turbine not time:
  - Tests generalisation to unseen turbine
  - Avoids data leakage (same turbine in train and eval)
  - Turbine 21 has 5,685 fault rows in two files -- enough for evaluation

Run:
  python scripts/prepare_dataset.py
"""

import sqlite3
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = "data/processed/wind_farm_a.db"

# Turbines used for training VAE (normal operation only)
TRAIN_TURBINES = {0, 10, 11, 13}

# Turbines used for evaluation (includes faults)
EVAL_TURBINES  = {21}


def prepare(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)

    print("Adding proper_split column...")
    try:
        conn.execute(
            "ALTER TABLE turbine_readings ADD COLUMN proper_split TEXT"
        )
    except sqlite3.OperationalError:
        pass   # Column already exists

    # Mark training turbines
    # Note: keep filter simple -- NULL comparisons fail silently in SQLite
    # Power and fault filters applied at query time, not here
    conn.execute(f"""
        UPDATE turbine_readings
        SET proper_split = 'train'
        WHERE asset_id IN ({','.join(str(t) for t in TRAIN_TURBINES)})
        AND status_type_id = 0
    """)

    # Mark eval turbines (all rows, including faults and curtailment)
    conn.execute(f"""
        UPDATE turbine_readings
        SET proper_split = 'eval'
        WHERE asset_id IN ({','.join(str(t) for t in EVAL_TURBINES)})
    """)

    conn.commit()

    # Summary
    summary = pd.read_sql("""
        SELECT
            proper_split,
            COUNT(*) as total,
            SUM(CASE WHEN is_fault=1 THEN 1 ELSE 0 END) as fault_rows,
            SUM(CASE WHEN is_fault=0 THEN 1 ELSE 0 END) as normal_rows,
            COUNT(DISTINCT asset_id) as turbines
        FROM turbine_readings
        WHERE proper_split IS NOT NULL
        GROUP BY proper_split
    """, conn)

    print(f"\n{'='*50}")
    print(f"  Dataset partition complete")
    print(f"{'='*50}")
    print(summary.to_string(index=False))

    # Fault type breakdown for eval set
    fault_types = pd.read_sql("""
        SELECT fault_label, COUNT(*) as n
        FROM turbine_readings
        WHERE proper_split = 'eval' AND is_fault = 1
        GROUP BY fault_label
        ORDER BY n DESC
    """, conn)

    print(f"\n  Eval fault types:")
    print(fault_types.to_string(index=False))

    # Feature availability check
    print(f"\n  Key feature null rates (training set):")
    key_cols = [
        "wind_speed_ms", "active_power_kw", "generator_speed_rpm",
        "gearbox_oil_temp_c", "bearing_temp_fore_c", "bearing_temp_aft_c",
        "generator_temp_c", "ambient_temp_c", "grid_frequency_hz",
    ]
    sample = pd.read_sql("""
        SELECT * FROM turbine_readings
        WHERE proper_split = 'train'
        AND active_power_kw > 0
        LIMIT 10000
    """, conn)

    for col in key_cols:
        if col in sample.columns:
            null_rate = sample[col].isna().mean()
            flag = "⚠️ " if null_rate > 0.05 else "✅"
            print(f"  {flag} {col:30s} null={null_rate:.1%}")
        else:
            print(f"  ❌ {col:30s} MISSING")

    print(f"\n{'='*50}")
    print(f"  Ready. Next: python scripts/run_real_data_experiment.py")
    print(f"{'='*50}\n")

    conn.close()


if __name__ == "__main__":
    prepare()