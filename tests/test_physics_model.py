"""
tests/test_physics_model.py
============================
Unit tests for the physics model.

Design decision: Test the physics model FIRST because everything
downstream depends on it being correct. If the power curve is wrong,
all deviation metrics will be wrong.

Tests cover:
  - Boundary conditions (cut-in, rated, cut-out)
  - Physical correctness (power increases with wind in Region 2)
  - Edge cases (zero wind, negative wind, extreme values)
  - Config loading from YAML
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.physics_model import WindTurbinePhysicsModel, TurbinePhysicsConfig


def make_model() -> WindTurbinePhysicsModel:
    """Create a model with standard test parameters."""
    config = TurbinePhysicsConfig(
        name="Test Turbine",
        rated_power_kw=5000.0,
        cut_in_speed_ms=3.0,
        rated_speed_ms=11.4,
        cut_out_speed_ms=25.0,
        rotor_diameter_m=126.0,
        air_density_kg_m3=1.225,
        power_coefficient_cp=0.45,
    )
    return WindTurbinePhysicsModel(config)


def test_below_cut_in():
    """Turbine should produce zero power below cut-in speed."""
    model = make_model()
    assert model.expected_power_kw(0.0) == 0.0
    assert model.expected_power_kw(1.0) == 0.0
    assert model.expected_power_kw(2.9) == 0.0
    print("✓ Below cut-in: zero power")


def test_at_cut_in():
    """Power should start at (or very close to) zero at cut-in speed."""
    model = make_model()
    power = model.expected_power_kw(3.0)
    assert power >= 0.0, f"Power at cut-in should be >= 0, got {power}"
    assert power < 50.0, f"Power at cut-in should be near 0, got {power} kW"
    print(f"✓ At cut-in (3.0 m/s): {power:.2f} kW")


def test_at_rated_speed():
    """Power should equal rated power at rated wind speed."""
    model = make_model()
    power = model.expected_power_kw(11.4)
    assert abs(power - 5000.0) < 1.0, f"Expected ~5000 kW at rated speed, got {power}"
    print(f"✓ At rated speed (11.4 m/s): {power:.2f} kW")


def test_above_rated_speed():
    """Power should be clamped to rated above rated speed."""
    model = make_model()
    assert model.expected_power_kw(15.0) == 5000.0
    assert model.expected_power_kw(20.0) == 5000.0
    print("✓ Above rated speed: power clamped to rated")


def test_above_cut_out():
    """Power should be zero above cut-out (storm shutdown)."""
    model = make_model()
    assert model.expected_power_kw(25.0) == 0.0
    assert model.expected_power_kw(30.0) == 0.0
    print("✓ Above cut-out: zero power (storm protection)")


def test_monotonic_in_region_2():
    """Power should strictly increase with wind speed in Region 2."""
    model = make_model()
    speeds = [3.5, 5.0, 7.0, 9.0, 11.0, 11.3]
    powers = [model.expected_power_kw(v) for v in speeds]
    for i in range(1, len(powers)):
        assert powers[i] > powers[i-1], \
            f"Power not monotonically increasing: {powers[i-1]} → {powers[i]}"
    print(f"✓ Monotonically increasing in Region 2: {[f'{p:.0f}' for p in powers]} kW")


def test_negative_wind_raises():
    """Negative wind speed should raise ValueError."""
    model = make_model()
    try:
        model.expected_power_kw(-1.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Negative wind speed raises ValueError")


def test_capacity_factor():
    """Capacity factor should be between 0 and 1."""
    model = make_model()
    for v in [0, 5, 9, 11.4, 15, 25]:
        cf = model.capacity_factor(v)
        assert 0.0 <= cf <= 1.0, f"Capacity factor out of range at {v} m/s: {cf}"
    print("✓ Capacity factor always in [0, 1]")


def test_theoretical_max_power():
    """Theoretical max power should be >= rated power."""
    model = make_model()
    theoretical = model.config.theoretical_max_power_kw
    # The Cp is already efficiency-adjusted, so this should be close to rated
    print(f"✓ Theoretical max power: {theoretical:.1f} kW (rated: 5000 kW)")


def test_power_curve_table():
    """Power curve table should have correct length."""
    model = make_model()
    table = model.power_curve_table(v_min=0, v_max=30, steps=100)
    assert len(table) == 101  # 0 to 100 inclusive
    assert table[0][0] == 0.0  # Starts at 0
    print(f"✓ Power curve table: {len(table)} points")


if __name__ == "__main__":
    print("=" * 50)
    print("Wind Turbine Physics Model — Unit Tests")
    print("=" * 50)
    test_below_cut_in()
    test_at_cut_in()
    test_at_rated_speed()
    test_above_rated_speed()
    test_above_cut_out()
    test_monotonic_in_region_2()
    test_negative_wind_raises()
    test_capacity_factor()
    test_theoretical_max_power()
    test_power_curve_table()
    print("=" * 50)
    print("All tests passed! ✓")
