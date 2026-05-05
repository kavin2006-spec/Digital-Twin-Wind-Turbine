import sqlite3
import sys

db = "data/processed/wind_farm_a.db"
conn = sqlite3.connect(db)

# Check 1: proper_split values
print("=== proper_split distribution ===")
for row in conn.execute("SELECT proper_split, COUNT(*) FROM turbine_readings GROUP BY proper_split"):
    print(row)

# Check 2: is_fault values in train set
print("\n=== is_fault in train rows ===")
for row in conn.execute("SELECT is_fault, COUNT(*) FROM turbine_readings WHERE proper_split='train' GROUP BY is_fault"):
    print(row)

# Check 3: active_power_kw range in train set
print("\n=== power stats in train rows ===")
for row in conn.execute("SELECT MIN(active_power_kw), MAX(active_power_kw), AVG(active_power_kw) FROM turbine_readings WHERE proper_split='train'"):
    print(row)

# Check 4: try the exact query with no filters
print("\n=== train rows no filter (limit 3) ===")
for row in conn.execute("SELECT wind_speed_ms, active_power_kw, is_fault, proper_split FROM turbine_readings WHERE proper_split='train' LIMIT 3"):
    print(row)

# Check 5: try with is_fault filter
print("\n=== train rows with is_fault=0 (limit 3) ===")
for row in conn.execute("SELECT wind_speed_ms, active_power_kw, is_fault FROM turbine_readings WHERE proper_split='train' AND is_fault=0 LIMIT 3"):
    print(row)

# Check 6: check gearbox_bearing_temp_c exists and has values
print("\n=== gearbox_bearing_temp_c sample ===")
for row in conn.execute("SELECT gearbox_bearing_temp_c FROM turbine_readings WHERE proper_split='train' LIMIT 5"):
    print(row)

conn.close()
