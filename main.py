"""
main.py
=======
Entry point for the Wind Turbine Digital Twin.

Run modes:
  python main.py simulate   — Run N-step simulation, print summary
  python main.py test       — Run physics model unit tests
  python main.py dashboard  — Launch Streamlit dashboard (Phase 2)
"""

import sys
import os
import torch
import numpy as np

# Add project root to path so all module imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_simulation(n_steps: int = 500):
    """
    Run the digital twin for N steps and print a summary.
    This is the 'smoke test' — proves all modules wire together correctly.
    """
    from twin_engine.twin import DigitalTwin

    print("\n" + "="*60)
    print("  WIND TURBINE DIGITAL TWIN — Simulation Mode")
    print("="*60)

    twin = DigitalTwin(config_path="config/turbine_config.yaml")

    print(f"\nRunning {n_steps} timesteps...")
    snapshots = twin.run_batch(n_steps)
    
    # ── Evaluation ──────────────────────────────────────────────────
    from models.evaluator import AnomalyEvaluator, StepResult

    evaluator = AnomalyEvaluator(
        model_name="LSTM Autoencoder",
        early_tolerance=3,
        late_tolerance=2,
    )

    # Rebuild sensor ground truth from twin history
    # We need fault ground truth — stored in snapshot.fault_type
    for i, snap in enumerate(snapshots):
        evaluator.add(StepResult(
            step=i,
            is_fault_ground_truth=snap.fault_type is not None,
            is_anomaly_predicted=snap.is_anomaly,
            anomaly_score=snap.anomaly_score,
            fault_type=snap.fault_type,
        ))

    report = evaluator.evaluate()
    print(report)

    # Generate plots
    from models.visualizer import generate_all_plots
    generate_all_plots(
        results=evaluator._results,
        report=report,
        output_dir="outputs/plots",
    )

    # Summary statistics
    anomalies = [s for s in snapshots if s.is_anomaly]
    fault_steps = [s for s in snapshots if s.fault_type]
    avg_wind = sum(s.wind_speed_ms for s in snapshots) / len(snapshots)
    avg_actual = sum(s.actual_power_kw for s in snapshots) / len(snapshots)
    avg_expected = sum(s.expected_power_kw for s in snapshots) / len(snapshots)
    avg_efficiency = sum(s.efficiency_ratio for s in snapshots) / len(snapshots)

    print(f"\n{'─'*60}")
    print(f"  Simulation complete: {n_steps} steps")
    print(f"{'─'*60}")
    print(f"  Avg wind speed:        {avg_wind:.2f} m/s")
    print(f"  Avg expected power:    {avg_expected:.1f} kW")
    print(f"  Avg actual power:      {avg_actual:.1f} kW")
    print(f"  Avg efficiency ratio:  {avg_efficiency:.3f}")
    print(f"  Fault steps (ground):  {len(fault_steps)} / {n_steps} ({100*len(fault_steps)/n_steps:.1f}%)")
    print(f"  ML anomalies detected: {len(anomalies)} / {n_steps} ({100*len(anomalies)/n_steps:.1f}%)")
    print(f"  Model trained:         {'Yes' if twin.anomaly_detector.is_trained else 'No (need more data)'}")
    print(f"{'─'*60}")

    # Show last 5 snapshots
    print("\n  Last 5 snapshots:")
    print(f"  {'Time':8} {'Wind(m/s)':10} {'Exp(kW)':9} {'Act(kW)':9} {'Dev%':7} {'Anomaly':8} {'Fault'}")
    print(f"  {'─'*8} {'─'*10} {'─'*9} {'─'*9} {'─'*7} {'─'*8} {'─'*20}")
    for s in snapshots[-5:]:
        ts = s.timestamp.strftime("%H:%M:%S")
        flag = "🔴 YES" if s.is_anomaly else "   no"
        fault = s.fault_type or "-"
        print(f"  {ts:8} {s.wind_speed_ms:10.2f} {s.expected_power_kw:9.1f} {s.actual_power_kw:9.1f} {s.deviation_pct:+7.1f} {flag:8} {fault}")

    print(f"\n{'='*60}")
    print("  All systems nominal. Run 'python main.py dashboard' for UI.")
    print("="*60 + "\n")

    return snapshots


def run_tests():
    """Run physics model unit tests."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "tests/test_physics_model.py"],
        capture_output=False
    )
    return result.returncode

def run_monte_carlo(n_runs: int = 50, n_steps: int = 3600):
    """
    Monte Carlo evaluation — runs simulation N times with different
    random seeds and reports mean ± std across all metrics.
    Includes fault timing stratification to test training cutoff hypothesis.
    """
    import numpy as np
    from models.evaluator import AnomalyEvaluator, StepResult
    from twin_engine.twin import DigitalTwin

    print("\n" + "="*60)
    print(f"  MONTE CARLO EVALUATION — {n_runs} runs × {n_steps} steps")
    print("="*60)

    results = {
        "precision":             [],
        "recall":                [],
        "f1":                    [],
        "pr_auc":                [],
        "cost":                  [],
        "tp":                    [],
        "fp":                    [],
        "fn":                    [],
        "pre_train_recall":      [],
        "post_train_recall":     [],
        "pre_train_fault_steps": [],
        "post_train_fault_steps":[],
        "recall_efficiency_drop": [],
        "recall_overheating":     [],
        "recall_vibration_spike": [],
    }

    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}...", end=" ", flush=True)

        # Fresh twin each run with different random seeds
        twin = DigitalTwin(config_path="config/turbine_config.yaml")
        twin.wind_gen._rng.seed(run * 42)
        twin.fault_injector._rng.seed(run * 99)
        twin.sensor_sim._rng.seed(run * 7)
        # Fix VAE training seed for reproducible convergence
        torch.manual_seed(run * 13)
        np.random.seed(run * 13)

        snapshots = twin.run_batch(n_steps)

        # Debug check
        print(f"  Trained: {twin.anomaly_detector.is_trained}, "
              f"Steps: {twin.anomaly_detector._step_count}")
        
        train_cutoff = twin.anomaly_detector.train_after

        # ── Standard evaluation ───────────────────────────────────
        evaluator = AnomalyEvaluator(
            model_name=f"Run {run+1}",
            early_tolerance=3,
            late_tolerance=2,
        )

        # ── Fault type stratification ─────────────────────────
        fault_types = ["efficiency_drop", "overheating", "vibration_spike"]
        for ft in fault_types:
            ft_steps = [
                (i, snap.is_anomaly)
                for i, snap in enumerate(snapshots)
                if snap.fault_type == ft
            ]
            if ft_steps:
                ft_recall = sum(1 for _, d in ft_steps if d) / len(ft_steps)
            else:
                ft_recall = 0.0
            results[f"recall_{ft}"].append(ft_recall)

        for i, snap in enumerate(snapshots):
            evaluator.add(StepResult(
                step=i,
                is_fault_ground_truth=snap.fault_type is not None,
                is_anomaly_predicted=snap.is_anomaly,
                anomaly_score=snap.anomaly_score,
                fault_type=snap.fault_type,
            ))

        report = evaluator.evaluate()
        cm     = report.confusion

        results["precision"].append(cm.precision)
        results["recall"].append(cm.recall)
        results["f1"].append(cm.f1)
        results["pr_auc"].append(report.pr_auc)
        results["cost"].append(report.cost.total_cost)
        results["tp"].append(cm.TP)
        results["fp"].append(cm.FP)
        results["fn"].append(cm.FN)

        # ── Fault timing stratification ───────────────────────────
        pre_faults  = []   # (step, was_detected)
        post_faults = []

        for i, snap in enumerate(snapshots):
            if snap.fault_type is not None:
                entry = (i, snap.is_anomaly)
                if i < train_cutoff:
                    pre_faults.append(entry)
                else:
                    post_faults.append(entry)

        pre_recall = (
            sum(1 for _, d in pre_faults if d) / len(pre_faults)
            if pre_faults else 0.0
        )
        post_recall = (
            sum(1 for _, d in post_faults if d) / len(post_faults)
            if post_faults else 0.0
        )

        results["pre_train_recall"].append(pre_recall)
        results["post_train_recall"].append(post_recall)
        results["pre_train_fault_steps"].append(len(pre_faults))
        results["post_train_fault_steps"].append(len(post_faults))

        print(
            f"F1={cm.f1:.3f}  Recall={cm.recall:.3f}  "
            f"PR-AUC={report.pr_auc:.3f}  |  "
            f"Pre={len(pre_faults)} steps (recall={pre_recall:.2f})  "
            f"Post={len(post_faults)} steps (recall={post_recall:.2f})"
        )

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MONTE CARLO SUMMARY — {n_runs} runs × {n_steps} steps")
    print(f"{'─'*60}")
    print(f"  {'Metric':22}  {'Mean':>8}  {'Std':>8}  "
          f"{'CV':>6}  {'Min':>8}  {'Max':>8}")
    # ── Fault type recall ─────────────────────────────────────
    print(f"{'─'*60}")
    print(f"  Fault type recall breakdown:")
    print(f"  {'─'*56}")

    fault_types = {
        "recall_efficiency_drop": "efficiency_drop  (35% power loss)",
        "recall_overheating":     "overheating      (20% power loss)",
        "recall_vibration_spike": "vibration_spike  ( 8% power loss)",
    }

    for key, label in fault_types.items():
        arr  = np.array(results[key])
        mean = arr.mean()
        std  = arr.std()
        cv   = std / mean if mean > 0 else 0.0
        flag = "✅" if mean > 0.7 else ("⚠️ " if mean > 0.4 else "❌")
        print(f"  {flag} {label}")
        print(f"     Recall: {mean:.3f} ± {std:.3f}  CV={cv:.3f}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*8}  "
          f"{'─'*6}  {'─'*8}  {'─'*8}")

    core_metrics = [
        "precision", "recall", "f1", "pr_auc", "cost", "tp", "fp", "fn"
    ]

    for metric in core_metrics:
        arr  = np.array(results[metric])
        mean = arr.mean()
        std  = arr.std()
        cv   = std / mean if mean > 0 else 0.0
        mn   = arr.min()
        mx   = arr.max()

        if metric == "cost":
            print(f"  {metric:22}  "
                  f"€{mean:>7,.0f}  "
                  f"€{std:>7,.0f}  "
                  f"{cv:>6.2f}  "
                  f"€{mn:>7,.0f}  "
                  f"€{mx:>7,.0f}")
        else:
            print(f"  {metric:22}  "
                  f"{mean:>8.3f}  "
                  f"{std:>8.3f}  "
                  f"{cv:>6.2f}  "
                  f"{mn:>8.3f}  "
                  f"{mx:>8.3f}")

    # ── Stability verdict ─────────────────────────────────────────
    print(f"{'─'*60}")
    print(f"  Stability verdict (CV < 0.15 = production ready):")
    for metric in ["precision", "recall", "f1", "pr_auc"]:
        arr  = np.array(results[metric])
        cv   = arr.std() / arr.mean()
        flag = "✅" if cv < 0.15 else ("⚠️ " if cv < 0.25 else "❌")
        print(f"  {flag} {metric:12} CV={cv:.3f}")

    # ── Fault timing stratification ───────────────────────────────
    print(f"{'─'*60}")
    print(f"  Fault timing stratification:")

    pre_recalls  = np.array(results["pre_train_recall"])
    post_recalls = np.array(results["post_train_recall"])
    pre_steps    = np.array(results["pre_train_fault_steps"])
    post_steps   = np.array(results["post_train_fault_steps"])

    print(f"  Pre-training faults  (steps 0–{train_cutoff}):")
    print(f"    Avg fault steps/run:  {pre_steps.mean():.1f}")
    print(f"    Recall: {pre_recalls.mean():.3f} ± {pre_recalls.std():.3f}  "
          f"CV={pre_recalls.std()/max(pre_recalls.mean(),1e-6):.3f}")

    print(f"  Post-training faults (steps {train_cutoff}+):")
    print(f"    Avg fault steps/run:  {post_steps.mean():.1f}")
    print(f"    Recall: {post_recalls.mean():.3f} ± {post_recalls.std():.3f}  "
          f"CV={post_recalls.std()/max(post_recalls.mean(),1e-6):.3f}")

    gap = post_recalls.mean() - pre_recalls.mean()
    print(f"\n  Post vs Pre recall gap: {gap:+.3f}")

    if gap > 0.10:
        print(f"  ✅ Hypothesis confirmed — training cutoff drives recall variance")
        print(f"     Fix: delay fault injection until after step {train_cutoff}")
    elif gap > 0.05:
        print(f"  ⚠️  Partial effect — training cutoff contributes but isn't sole cause")
    else:
        print(f"  ❌ Hypothesis rejected — variance has another cause")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "simulate"

    if mode == "test":
        sys.exit(run_tests())
    elif mode == "simulate":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
        run_simulation(n)
    elif mode == "dashboard":
        print("Launching dashboard... (run: streamlit run dashboard/app.py)")
        os.system("streamlit run dashboard/app.py")
    elif mode == "evaluate":
        n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
        run_monte_carlo(n_runs, n_steps)    
    else:
        print(f"Unknown mode: {mode}. Use: simulate | test | dashboard")
        sys.exit(1)
