"""
experiments/compare.py
========================
Side-by-side comparison of all model versions tested.

Pulls stored results and renders a clean comparison table.
Run:
  python experiments/compare.py

Results are hardcoded from Monte Carlo runs — update when new
experiments complete.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Stored results from Monte Carlo runs ─────────────────────────────────────
# Each entry: (mean, std, cv) unless single value

RESULTS = {
    "Isolation Forest (baseline)": {
        "n_runs":       1,
        "n_steps":      3600,
        "precision":    (0.32,  None, None),
        "recall":       (0.376, None, None),
        "f1":           (0.238, None, None),
        "pr_auc":       (0.32,  None, None),
        "cost":         (None,  None, None),
        "source":       "milestone doc — single run",
        "notes":        "Baseline. High false alarm rate.",
    },
    "LSTM Autoencoder": {
        "n_runs":       1,
        "n_steps":      3600,
        "precision":    (0.446, None, None),
        "recall":       (0.484, None, None),
        "f1":           (0.464, None, None),
        "pr_auc":       (0.36,  None, None),
        "cost":         (None,  None, None),
        "source":       "milestone doc — single run",
        "notes":        "Iterative self-cleaning training. Better than IF.",
    },
    "Temporal VAE (4-feat, 3σ threshold)": {
        "n_runs":       50,
        "n_steps":      3600,
        "precision":    (0.429, 0.093, 0.216),
        "recall":       (0.801, 0.122, 0.153),
        "f1":           (0.549, 0.095, 0.174),
        "pr_auc":       (0.427, 0.052, 0.123),
        "cost":         (2_933_600, 1_176_765, 0.40),
        "source":       "Monte Carlo 50×3600",
        "notes":        "KL collapse fixed. Calibration bug fixed.",
    },
    "Temporal VAE (4-feat, 4σ threshold)": {
        "n_runs":       50,
        "n_steps":      3600,
        "precision":    (0.461, 0.071, 0.154),
        "recall":       (0.780, 0.130, 0.167),
        "f1":           (0.572, 0.075, 0.131),
        "pr_auc":       (0.427, 0.052, 0.123),
        "cost":         (2_910_840, 1_184_918, 0.41),
        "source":       "Monte Carlo 50×3600",
        "notes":        "Precision CV improved. F1 CV production-ready.",
    },
    "Temporal VAE (4-feat, 5σ threshold) ★": {
        "n_runs":       50,
        "n_steps":      3600,
        "precision":    (0.476, 0.062, 0.131),
        "recall":       (0.760, 0.134, 0.176),
        "f1":           (0.578, 0.069, 0.119),
        "pr_auc":       (0.427, 0.052, 0.123),
        "cost":         (2_917_320, 1_195_195, 0.41),
        "source":       "Monte Carlo 50×3600",
        "notes":        "Best balance. Precision+F1 CV production-ready.",
    },
    "Denoising VAE + Semi-supervised (isolated)": {
        "n_runs":       50,
        "n_steps":      3600,
        "precision":    (0.429, 0.093, 0.216),
        "recall":       (0.801, 0.122, 0.153),
        "f1":           (0.549, 0.095, 0.174),
        "pr_auc":       (0.427, 0.052, 0.123),
        "cost":         (2_933_600, 1_176_765, 0.40),
        "source":       "Monte Carlo 50×3600",
        "notes":        "Diagnostic only. Threshold unchanged. "
                        "bearing_fault undetectable with current features.",
    },
    "VAE — Real Data (CARE Wind Farm A)": {
        "n_runs":       1,
        "n_steps":      215559,
        "precision":    (0.028, None, None),
        "recall":       (0.004, None, None),
        "f1":           (0.006, None, None),
        "pr_auc":       (0.072, None, None),
        "cost":         (6_058_000, None, None),
        "source":       "Single run — turbine 21 eval set",
        "notes":        "Fault labels = high-load events, not degradation. "
                        "Consistent with published CARE baselines (0.05-0.15).",
    },
    
    "Multi-scale VAE (30s+120s AND)": {
    "n_runs":    50,
    "n_steps":   3600,
    "precision": (0.588, 0.078, 0.132),
    "recall":    (0.796, 0.119, 0.149),
    "f1":        (0.669, 0.075, 0.112),
    "pr_auc":    (0.400, 0.085, 0.213),
    "cost":      (2_606_000, 1_106_006, 0.42),
    "source":    "Monte Carlo 50x3600",
    "notes":     "Best overall. 3/4 stability metrics production-ready. "
                 "FP -41% vs single-scale. bearing_fault still undetectable.",
    },
}


def _cv_flag(cv: float | None) -> str:
    if cv is None:
        return "  "
    if cv < 0.15:
        return "✅"
    if cv < 0.25:
        return "⚠️ "
    return "❌"


def print_comparison_table() -> None:
    print(f"\n{'='*90}")
    print(f"  Wind Turbine Digital Twin — Model Comparison")
    print(f"  All simulation results: 50 Monte Carlo runs × 3600 steps")
    print(f"  Metrics shown as: mean ± std  (CV)   ✅=CV<0.15  ⚠️=CV<0.25  ❌=CV≥0.25")
    print(f"{'='*90}\n")

    # Header
    col_w = 42
    print(f"  {'Model':{col_w}} {'Precision':>18} {'Recall':>18} {'F1':>18} {'PR-AUC':>14}")
    print(f"  {'─'*col_w} {'─'*18} {'─'*18} {'─'*18} {'─'*14}")

    for name, r in RESULTS.items():
        def fmt(key: str) -> str:
            mean, std, cv = r[key]
            if mean is None:
                return f"{'N/A':>18}"
            if std is None:
                return f"{mean:>18.3f}"
            flag = _cv_flag(cv)
            return f"{flag}{mean:.3f}±{std:.3f}({cv:.2f})"

        star = " ★" if "★" in name else ""
        display_name = name.replace(" ★", "")
        print(f"  {display_name:{col_w}} {fmt('precision'):>18} "
              f"{fmt('recall'):>18} {fmt('f1'):>18} {fmt('pr_auc'):>14}")

    print(f"\n  ★ = recommended production model\n")

    # Cost table
    print(f"  {'─'*60}")
    print(f"  Cost-sensitive evaluation (€500k/missed fault, €2k/false alarm)")
    print(f"  {'─'*60}")
    print(f"  {'Model':{col_w}} {'Mean Cost':>15} {'Std':>12}")
    print(f"  {'─'*col_w} {'─'*15} {'─'*12}")

    for name, r in RESULTS.items():
        mean, std, cv = r["cost"]
        display_name = name.replace(" ★", "")
        if mean is None:
            cost_str = f"{'N/A':>15}"
            std_str  = f"{'':>12}"
        else:
            cost_str = f"€{mean:>13,.0f}"
            std_str  = f"€{std:>10,.0f}" if std else f"{'N/A':>12}"
        print(f"  {display_name:{col_w}} {cost_str} {std_str}")

    # Fault type breakdown (best model only)
    print(f"\n  {'─'*60}")
    print(f"  Fault type recall — best model (5σ threshold, 50 runs):")
    print(f"  {'─'*60}")
    fault_recall = {
        "efficiency_drop  (35% power loss)": (0.700, 0.396, 0.565),
        "overheating      (20% power loss)": (0.744, 0.352, 0.473),
        "vibration_spike  ( 8% power loss)": (0.594, 0.447, 0.752),
        "bearing_fault    (vibration only)": (None,  None,  None),
    }
    for fault, (mean, std, cv) in fault_recall.items():
        if mean is None:
            print(f"  ⛔ {fault}: UNDETECTABLE — requires 100Hz vibration sensor")
        else:
            # Flag based on mean recall, not CV
            flag = "✅" if mean >= 0.70 else ("⚠️ " if mean >= 0.50 else "❌")
            print(f"  {flag} {fault}: {mean:.3f} ± {std:.3f}  CV={cv:.3f}")

    # Key findings
    print(f"\n  {'─'*60}")
    print(f"  Key findings:")
    print(f"  {'─'*60}")
    findings = [
        "Isolation Forest → VAE: F1 +0.340, PR-AUC +0.107",
        "Threshold sigma (3→5): Precision CV 0.216→0.131, F1 CV 0.174→0.119",
        "41-feature VAE failed: wind-correlated signals, insufficient training data",
        "Semi-supervised diagnostic: bearing_fault undetectable (needs 100Hz sensor)",
        "Real data (CARE): fault labels = high-load events, not degradation faults",
        "Production-ready metrics: F1 CV=0.119 ✅, PR-AUC CV=0.123 ✅",
        "Remaining variance driver: vibration_spike recall CV=0.752 (8% power loss)",
    ]
    for f in findings:
        print(f"    • {f}")

    print(f"\n{'='*90}\n")


if __name__ == "__main__":
    print_comparison_table()
