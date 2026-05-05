"""
twin_engine/twin.py
====================
The Digital Twin — central orchestrator of the system.

Design decision: The Twin is the ONLY module that knows about all others.
All other modules are pure / single-responsibility. The twin wires them
together and produces TwinSnapshots for the dashboard.

Responsibilities:
  1. Accept wind readings (simulation or real sensors)
  2. Run physics model → expected power
  3. Sensor simulator → 21 Tier 1 signals + VibrationWindow
  4. Comparator → deviation metrics → TwinSnapshot
  5. [Phase 3a] FeatureEngineer → EnrichedReading (Tier 2 + Tier 3)
  6. Anomaly detector → anomaly flag
  7. Maintain rolling history for dashboard

Changes from v1:
  - VibrationSimulator added as explicit dependency
  - sensor_sim.read() now returns (SensorReading, VibrationWindow)
  - vib_window carried through process() for Phase 3a FeatureEngineer
  - SensorReading field renamed: actual_power_kw → active_power_kw
"""

from __future__ import annotations
from collections import deque
from datetime import datetime
from typing import Optional

import yaml

from data_pipeline.schemas import WindReading, SensorReading, TwinSnapshot
from data_pipeline.feature_engineer import FeatureEngineer
from models.physics_model import WindTurbinePhysicsModel, TurbinePhysicsConfig
from models.ml_anomaly_detector import LSTMAutoencoderDetector
from simulation.wind_generator import WindSpeedGenerator
from simulation.fault_injector import FaultInjector
from simulation.sensor_simulator import SensorSimulator
from simulation.vibration_simulator import VibrationSimulator
from twin_engine.comparator import compute_snapshot, RollingStatistics


class DigitalTwin:
    """
    The core Digital Twin — wires all modules together.

    Args:
        config_path:  Path to turbine_config.yaml
        history_size: Number of snapshots kept in rolling history
    """

    def __init__(
        self,
        config_path:  str = "config/turbine_config.yaml",
        history_size: int = 3600,
    ):
        # Load config
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        sim_cfg  = cfg["simulation"]
        anom_cfg = cfg["anomaly"]
        ml_cfg   = cfg["ml"]

        # ── Physics model ─────────────────────────────────────────────
        physics_config = TurbinePhysicsConfig.from_yaml(config_path)
        self.physics   = WindTurbinePhysicsModel(physics_config)

        # ── Wind generator ────────────────────────────────────────────
        self.wind_gen = WindSpeedGenerator(
            mean_speed_ms=sim_cfg["wind_mean_speed_ms"],
            turbulence_intensity=sim_cfg["wind_turbulence_intensity"],
            dt_seconds=sim_cfg["timestep_seconds"],
        )

        # ── Fault injector ────────────────────────────────────────────
        self.fault_injector = FaultInjector(
            fault_probability=anom_cfg["fault_probability"],
            fault_duration_seconds=anom_cfg["fault_duration_seconds"],
            efficiency_drop_fraction=anom_cfg["efficiency_drop_fraction"],
            enabled=anom_cfg["enabled"],
            fault_start_after_step=anom_cfg.get("fault_start_after_step", 0),
        )

        # ── Vibration simulator (NEW — 100Hz per timestep) ────────────
        # Lives here, not inside SensorSimulator, so twin controls its
        # lifecycle and can access the VibrationWindow independently
        # for feature extraction in Phase 3a.
        self.vibration_sim = VibrationSimulator(
            sampling_rate_hz=sim_cfg.get("vibration_sampling_rate_hz", 100.0),
            base_noise_level=sim_cfg.get("vibration_base_noise", 0.05),
            seed=sim_cfg.get("vibration_seed", 77),
        )

        # ── Sensor simulator ──────────────────────────────────────────
        self.sensor_sim = SensorSimulator(
            physics_model=self.physics,
            fault_injector=self.fault_injector,
            vibration_sim=self.vibration_sim,        # NEW: wired in here
            noise_stddev=sim_cfg["sensor_noise_stddev"],
            ambient_temp_c=sim_cfg.get("ambient_temp_c", 15.0),
        )

        # ── Anomaly detector ──────────────────────────────────────────
        # ── Multi-scale anomaly detection ────────────────────────────────────
        # Two detectors running in parallel at different temporal resolutions.
        # Short window (30s): catches sudden onset faults (vibration_spike)
        # Long window (120s): catches gradual degradation (overheating, bearing)
        # Combined score: max(short, long) — flag if either window detects
        #
        # Industrial insight: Multi-resolution temporal detection is used by
        # Siemens Gamesa in their SCADA monitoring platform. They call it
        # "temporal pyramiding" — short windows for transient faults, long
        # windows for drift faults. Your design independently arrived at
        # the same architecture.

        common_kwargs = dict(
            latent_dim=ml_cfg["latent_dim"],
            n_epochs=ml_cfg["n_epochs"],
            beta=ml_cfg["beta"],
            kl_weight=ml_cfg["kl_weight"],
            n_ensemble=ml_cfg.get("n_ensemble", 3),
            calibration_size=ml_cfg.get("calibration_size", 200),
        )

        self.detector_short = LSTMAutoencoderDetector(
            window_size=ml_cfg.get("window_size_short", 30),
            slide_every=ml_cfg.get("slide_every", 10),
            train_after=ml_cfg["train_after"],
            **common_kwargs,
        )
        self.detector_long = LSTMAutoencoderDetector(
            window_size=ml_cfg.get("window_size_long", 120),
            slide_every=ml_cfg.get("slide_every", 10),
            train_after=ml_cfg.get("train_after_long", 400),
            **common_kwargs,
        )

        # Compatibility alias -- main.py and evaluator access this
        self.anomaly_detector = self.detector_short

        # Combined score weight: 0=all short, 1=all long, 0.5=equal
        self._multi_scale_weight = ml_cfg.get("multi_scale_weight", 0.5)

        # ── Feature engineer (Phase 3a) ───────────────────────────────
        self.feature_engineer = FeatureEngineer(
            physics_model=self.physics,
            window_seconds=sim_cfg.get("rolling_window_seconds", 60),
            sampling_hz=1.0 / sim_cfg.get("timestep_seconds", 1),
        )

        self.rolling_stats = RollingStatistics(window_size=60)

        # History buffer (dashboard reads from this)
        self._history: deque[TwinSnapshot] = deque(maxlen=history_size)

        # Counters
        self.total_steps  = 0
        self.anomaly_count = 0

        print(f"[DigitalTwin] Initialised: {self.physics}")
        print(
            f"[DigitalTwin] Theoretical max power: "
            f"{self.physics.config.theoretical_max_power_kw:.1f} kW"
        )

    # ── Core step ─────────────────────────────────────────────────────────────

    def step(self) -> TwinSnapshot:
        """Advance the twin by one timestep using simulated wind."""
        wind_speed   = self.wind_gen.next_speed()
        wind_reading = WindReading(
            timestamp=datetime.utcnow(),
            wind_speed_ms=wind_speed,
        )
        return self.process(wind_reading)

    def process(self, wind_reading: WindReading) -> TwinSnapshot:
        """
        Process one wind reading through the full twin pipeline.

        Accepts EITHER simulated or real sensor data via the same
        interface — that's the value of the WindReading abstraction.

        Pipeline:
          wind → physics model → expected power
          wind → sensor sim   → (SensorReading, VibrationWindow)
          sensor + expected   → comparator → TwinSnapshot
          [Phase 3a]          → FeatureEngineer → EnrichedReading
          snapshot + sensor   → anomaly detector

        Args:
            wind_reading: WindReading (simulated or real anemometer)

        Returns:
            TwinSnapshot with all metrics and anomaly flag.
        """
        # 1. Physics model → expected power
        expected_power = self.physics.expected_power_kw(
            wind_reading.wind_speed_ms
        )

        # 2. Sensor simulator → actual readings + vibration window
        #    read() now returns (SensorReading, VibrationWindow)
        sensor, vib_window = self.sensor_sim.read(wind_reading)

        # 3. Comparator → deviation metrics
        snapshot = compute_snapshot(sensor, expected_power)

        # 4. Feature engineer → EnrichedReading (Tier 2 + Tier 3)
        enriched = self.feature_engineer.compute(sensor, vib_window, snapshot)
        snapshot.enriched = enriched

        # 5. Rolling statistics
        self.rolling_stats.update(snapshot.efficiency_ratio)

        # 6. Update both detectors
        self.detector_short.update(snapshot, sensor)
        self.detector_long.update(snapshot, sensor)

        # 7. Calibration — both detectors calibrate independently
        for detector in [self.detector_short, self.detector_long]:
            if detector.is_trained and not detector._calibration_done:
                just_calibrated = detector.calibrate(snapshot, sensor)
                if just_calibrated and not detector.characterisation_done:
                    self._run_fault_characterisation_for(detector)

        # 8. Predict — combine short and long window scores
        if self.detector_short.is_trained or self.detector_long.is_trained:
            result = self._combined_predict(snapshot, sensor)
            snapshot.is_anomaly    = result.is_anomaly
            snapshot.anomaly_score = result.score
            if result.is_anomaly:
                self.anomaly_count += 1

        # 9. Store in history
        self._history.append(snapshot)
        self.total_steps += 1

        return snapshot

    # ── Fault characterisation ───────────────────────────────────────────────

    def _combined_predict(
        self, snapshot, sensor
    ):
        """
        Combine short and long window anomaly scores.

        Strategy: max(short_score, long_score)
          - If either window detects an anomaly, flag it
          - Short window: better for sudden faults (vibration_spike)
          - Long window:  better for gradual faults (overheating, bearing)
          - max() preserves the strongest signal without diluting it

        Falls back to whichever detector is trained if only one is ready.
        """
        from models.ml_anomaly_detector import AnomalyResult

        short_trained = self.detector_short.is_trained
        long_trained  = self.detector_long.is_trained

        if short_trained and long_trained:
            r_short = self.detector_short.predict(snapshot, sensor)
            r_long  = self.detector_long.predict(snapshot, sensor)

            # Use max score — flag if either window triggers
            combined_score = max(r_short.score, r_long.score)
            combined_score = min(1.0, max(0.0, combined_score))

            # Anomaly if either detector flags it
            is_anomaly = r_short.is_anomaly and  r_long.is_anomaly

            return AnomalyResult(
                is_anomaly=is_anomaly,
                score=round(combined_score, 4),
                z_score=round(max(r_short.z_score, r_long.z_score), 3),
                method="multi_scale_vae",
                reconstruction_error=round(
                    (r_short.reconstruction_error + r_long.reconstruction_error) / 2, 6
                ),
                kl_divergence=round(
                    (r_short.kl_divergence + r_long.kl_divergence) / 2, 6
                ),
            )

        elif short_trained:
            return self.detector_short.predict(snapshot, sensor)
        elif long_trained:
            return self.detector_long.predict(snapshot, sensor)
        else:
            return self.detector_short._warmup_predict(snapshot)

    def _run_fault_characterisation_for(self, detector) -> None:
        """Run fault characterisation for a specific detector instance."""
        from datetime import datetime
        from data_pipeline.schemas import WindReading
        from simulation.fault_injector import FaultInjector, ActiveFault
        from simulation.sensor_simulator import SensorSimulator
        from simulation.vibration_simulator import VibrationSimulator
        from twin_engine.comparator import compute_snapshot
        from data_pipeline.feature_engineer import FeatureEngineer

        print(f"[DigitalTwin] Fault characterisation "
              f"(window={detector.window_size}s)...")

        CHAR_SEED = 9999
        isolated_fault  = FaultInjector(enabled=True, seed=CHAR_SEED)
        isolated_vib    = VibrationSimulator(seed=CHAR_SEED)
        isolated_sensor = SensorSimulator(
            physics_model=self.physics,
            fault_injector=isolated_fault,
            vibration_sim=isolated_vib,
            noise_stddev=0.02,
            seed=CHAR_SEED,
        )

        fault_types = [
            "efficiency_drop", "overheating", "vibration_spike",
            "bearing_fault", "gearbox_fault",
        ]
        n_steps    = detector.window_size + 40
        severity   = 0.8
        wind_speed = 9.0

        fault_features_by_type: dict[str, list[list[float]]] = {}

        for fault_type in fault_types:
            features_this_fault: list[list[float]] = []
            isolated_fe = FeatureEngineer(
                physics_model=self.physics,
                window_seconds=60,
                sampling_hz=1.0,
            )
            isolated_fault._active_fault = ActiveFault(
                fault_type=fault_type,
                remaining_steps=n_steps + 10,
                severity=severity,
            )

            for step in range(n_steps):
                wind_reading = WindReading(
                    timestamp=datetime.utcnow(),
                    wind_speed_ms=wind_speed + (step % 5) * 0.2,
                )
                expected_power = self.physics.expected_power_kw(
                    wind_reading.wind_speed_ms
                )
                sensor, vib_window = isolated_sensor.read(wind_reading)
                snapshot_c = compute_snapshot(sensor, expected_power)
                enriched   = isolated_fe.compute(sensor, vib_window, snapshot_c)
                snapshot_c.enriched = enriched
                features = detector._extract_features(snapshot_c, sensor)
                features_this_fault.append(features)

            fault_features_by_type[fault_type] = features_this_fault

        detector.characterise_faults_by_type(fault_features_by_type)

    def _run_fault_characterisation(self) -> None:
        """
        Generate synthetic fault windows using ISOLATED simulators.

        Critical design: uses completely separate VibrationSimulator,
        FaultInjector, and SensorSimulator instances with fixed seeds.
        The main simulation's RNG state (self.sensor_sim, self.vibration_sim,
        self.fault_injector) is NEVER touched -- zero contamination.

        Without isolation, 500 characterisation steps advance the main
        vibration simulator RNG, making every subsequent simulation step
        deterministically different. This inflates CV across Monte Carlo
        runs and degrades all stability metrics.
        """
        print("[DigitalTwin] Running fault characterisation (isolated)...")

        from datetime import datetime
        from data_pipeline.schemas import WindReading
        from simulation.fault_injector import FaultInjector, ActiveFault
        from simulation.sensor_simulator import SensorSimulator
        from simulation.vibration_simulator import VibrationSimulator
        from twin_engine.comparator import compute_snapshot

        # Fixed seed -- reproducible across all Monte Carlo runs
        CHAR_SEED = 9999

        isolated_fault  = FaultInjector(enabled=True, seed=CHAR_SEED)
        isolated_vib    = VibrationSimulator(seed=CHAR_SEED)
        isolated_sensor = SensorSimulator(
            physics_model=self.physics,
            fault_injector=isolated_fault,
            vibration_sim=isolated_vib,
            noise_stddev=0.02,
            seed=CHAR_SEED,
        )

        # Separate FeatureEngineer instance -- own rolling buffers
        from data_pipeline.feature_engineer import FeatureEngineer
        isolated_fe = FeatureEngineer(
            physics_model=self.physics,
            window_seconds=60,
            sampling_hz=1.0,
        )

        fault_types = [
            "efficiency_drop", "overheating", "vibration_spike",
            "bearing_fault", "gearbox_fault",
        ]
        window_size = self.anomaly_detector.window_size
        n_steps     = window_size + 40
        severity    = 0.8
        wind_speed  = 9.0

        fault_features_by_type: dict[str, list[list[float]]] = {}

        for fault_type in fault_types:
            features_this_fault: list[list[float]] = []

            # Reset isolated simulators for each fault type
            isolated_fault._active_fault = ActiveFault(
                fault_type=fault_type,
                remaining_steps=n_steps + 10,
                severity=severity,
            )
            # Reset feature engineer rolling buffers
            isolated_fe = FeatureEngineer(
                physics_model=self.physics,
                window_seconds=60,
                sampling_hz=1.0,
            )

            for step in range(n_steps):
                wind_reading = WindReading(
                    timestamp=datetime.utcnow(),
                    wind_speed_ms=wind_speed + (step % 5) * 0.2,
                )
                expected_power = self.physics.expected_power_kw(
                    wind_reading.wind_speed_ms
                )
                sensor, vib_window = isolated_sensor.read(wind_reading)
                snapshot = compute_snapshot(sensor, expected_power)
                enriched = isolated_fe.compute(sensor, vib_window, snapshot)
                snapshot.enriched = enriched

                features = self.anomaly_detector._extract_features(
                    snapshot, sensor
                )
                features_this_fault.append(features)

            fault_features_by_type[fault_type] = features_this_fault
            print(f"  {fault_type}: {len(features_this_fault)} steps generated")

        # Main simulation state completely untouched
        self.anomaly_detector.characterise_faults_by_type(fault_features_by_type)

    # ── Batch run ─────────────────────────────────────────────────────────────

    def run_batch(self, n_steps: int) -> list[TwinSnapshot]:
        """Run the twin for N steps. Returns all snapshots."""
        return [self.step() for _ in range(n_steps)]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def history(self) -> list[TwinSnapshot]:
        """Return history as a list (most recent last)."""
        return list(self._history)

    @property
    def latest(self) -> Optional[TwinSnapshot]:
        """Most recent snapshot."""
        return self._history[-1] if self._history else None

    @property
    def anomaly_rate(self) -> float:
        """Fraction of total steps flagged as anomalies."""
        if self.total_steps == 0:
            return 0.0
        return self.anomaly_count / self.total_steps