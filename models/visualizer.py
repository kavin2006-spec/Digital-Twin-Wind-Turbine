"""
models/visualizer.py
=====================
Generates evaluation plots after each simulation run.

Plots produced:
  1. Precision-Recall curve
  2. ROC curve  
  3. Reconstruction error distribution (fault vs normal)
  4. Anomaly score timeline (predicted vs ground truth)
  5. Confusion matrix heatmap

Design decision: completely separate from detector and evaluator.
It only needs a list of StepResults — doesn't care which model
produced them. Same model-agnostic principle as evaluator.py.
"""

from __future__ import annotations
import os
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — saves to file, no popup
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from models.evaluator import StepResult, EvaluationReport


# ─── Style ────────────────────────────────────────────────────────────────────

# Industrial dark theme — matches dashboard aesthetic
STYLE = {
    "bg":       "#080d13",
    "surface":  "#0d1a27",
    "border":   "#1e3a54",
    "text":     "#c8d8e8",
    "subtext":  "#4a7fa0",
    "blue":     "#7ecfff",
    "green":    "#22dd66",
    "red":      "#ff4444",
    "amber":    "#ffaa33",
    "grid":     "#1a2a3a",
}

def _apply_style(ax: plt.Axes, title: str = "") -> None:
    """Apply consistent dark industrial style to any axis."""
    ax.set_facecolor(STYLE["surface"])
    ax.tick_params(colors=STYLE["subtext"], labelsize=8)
    ax.xaxis.label.set_color(STYLE["subtext"])
    ax.yaxis.label.set_color(STYLE["subtext"])
    ax.spines["bottom"].set_color(STYLE["border"])
    ax.spines["left"].set_color(STYLE["border"])
    ax.spines["top"].set_color(STYLE["border"])
    ax.spines["right"].set_color(STYLE["border"])
    ax.grid(True, color=STYLE["grid"], linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color=STYLE["text"], fontsize=9,
                    fontweight="bold", pad=8)


# ─── Individual plots ─────────────────────────────────────────────────────────


def plot_pr_curve(
    results: list[StepResult],
    model_name: str,
    output_path: str,
) -> None:
    """
    Precision-Recall curve.
    Sweeps every possible threshold and plots precision vs recall.
    The area under this curve (PR-AUC) is our primary model metric.
    """
    # Sort by score descending — sweep from high confidence to low
    sorted_results = sorted(results, key=lambda r: r.anomaly_score, reverse=True)
    n_positive = sum(1 for r in results if r.is_fault_ground_truth)

    if n_positive == 0:
        return

    tp, fp = 0, 0
    precisions, recalls, thresholds = [], [], []

    for r in sorted_results:
        if r.is_fault_ground_truth:
            tp += 1
        else:
            fp += 1
        p = tp / (tp + fp)
        rec = tp / n_positive
        precisions.append(p)
        recalls.append(rec)
        thresholds.append(r.anomaly_score)

    # Find optimal F1 point
    f1_scores = [
        2 * p * r / (p + r) if (p + r) > 0 else 0
        for p, r in zip(precisions, recalls)
    ]
    best_idx = int(np.argmax(f1_scores))
    pr_auc = abs(np.trapezoid(precisions, recalls))

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(STYLE["bg"])
    _apply_style(ax, f"Precision-Recall Curve — {model_name}")

    ax.plot(recalls, precisions, color=STYLE["blue"], linewidth=2, label=f"PR curve (AUC={pr_auc:.3f})")
    ax.scatter(recalls[best_idx], precisions[best_idx],
               color=STYLE["amber"], s=80, zorder=5,
               label=f"Best F1={f1_scores[best_idx]:.3f} @ threshold={thresholds[best_idx]:.3f}")
    ax.axhline(y=n_positive/len(results), color=STYLE["red"],
               linestyle="--", linewidth=1, alpha=0.6, label="Random baseline")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=7, facecolor=STYLE["surface"],
              edgecolor=STYLE["border"], labelcolor=STYLE["text"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()
    print(f"  Saved: {output_path}")


def plot_roc_curve(
    results: list[StepResult],
    model_name: str,
    output_path: str,
) -> None:
    """
    ROC curve — True Positive Rate vs False Positive Rate.
    Familiar from your sound project. Included for comparison.
    """
    sorted_results = sorted(results, key=lambda r: r.anomaly_score, reverse=True)
    n_pos = sum(1 for r in results if r.is_fault_ground_truth)
    n_neg = len(results) - n_pos

    if n_pos == 0 or n_neg == 0:
        return

    tp, fp = 0, 0
    tprs, fprs = [0], [0]

    for r in sorted_results:
        if r.is_fault_ground_truth:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)

    tprs.append(1)
    fprs.append(1)
    roc_auc = abs(np.trapezoid(tprs, fprs))

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(STYLE["bg"])
    _apply_style(ax, f"ROC Curve — {model_name}")

    ax.plot(fprs, tprs, color=STYLE["green"],
            linewidth=2, label=f"ROC (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=STYLE["red"],
            linestyle="--", linewidth=1, alpha=0.6, label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8, facecolor=STYLE["surface"],
              edgecolor=STYLE["border"], labelcolor=STYLE["text"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()
    print(f"  Saved: {output_path}")


def plot_error_distribution(
    results: list[StepResult],
    model_name: str,
    output_path: str,
) -> None:
    """
    Reconstruction error distribution split by fault vs normal.

    This is the most diagnostic plot — it shows whether the model's
    error is actually separating the two classes. If the distributions
    overlap heavily, no threshold will work well. That's the root
    cause diagnosis we discussed.
    """
    normal_scores = [r.anomaly_score for r in results
                     if not r.is_fault_ground_truth]
    fault_scores  = [r.anomaly_score for r in results
                     if r.is_fault_ground_truth]

    if not normal_scores or not fault_scores:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(STYLE["bg"])
    _apply_style(ax, f"Anomaly Score Distribution — {model_name}")

    bins = np.linspace(0, 1, 40)
    ax.hist(normal_scores, bins=bins, alpha=0.6,
            color=STYLE["green"], label=f"Normal (n={len(normal_scores)})",
            density=True)
    ax.hist(fault_scores, bins=bins, alpha=0.6,
            color=STYLE["red"], label=f"Fault (n={len(fault_scores)})",
            density=True)

    # Show means
    ax.axvline(np.mean(normal_scores), color=STYLE["green"],
               linestyle="--", linewidth=1.5,
               label=f"Normal mean={np.mean(normal_scores):.3f}")
    ax.axvline(np.mean(fault_scores), color=STYLE["red"],
               linestyle="--", linewidth=1.5,
               label=f"Fault mean={np.mean(fault_scores):.3f}")

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, facecolor=STYLE["surface"],
              edgecolor=STYLE["border"], labelcolor=STYLE["text"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()
    print(f"  Saved: {output_path}")


def plot_anomaly_timeline(
    results: list[StepResult],
    model_name: str,
    output_path: str,
) -> None:
    """
    Timeline showing predicted anomaly score vs ground truth faults.

    The most intuitive plot — you can visually see where the model
    fires relative to where faults actually are.
    """
    steps  = [r.step for r in results]
    scores = [r.anomaly_score for r in results]
    truth  = [r.is_fault_ground_truth for r in results]
    preds  = [r.is_anomaly_predicted for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                    sharex=True, height_ratios=[3, 1])
    fig.patch.set_facecolor(STYLE["bg"])

    # Top: anomaly score over time
    _apply_style(ax1, f"Anomaly Detection Timeline — {model_name}")
    ax1.plot(steps, scores, color=STYLE["blue"],
             linewidth=0.8, alpha=0.8, label="Anomaly score")

    # Shade ground truth fault windows
    in_fault = False
    fault_start = 0
    for i, (step, is_fault) in enumerate(zip(steps, truth)):
        if is_fault and not in_fault:
            fault_start = step
            in_fault = True
        elif not is_fault and in_fault:
            ax1.axvspan(fault_start, step, alpha=0.15,
                       color=STYLE["red"], label="True fault" if fault_start == steps[next((j for j, f in enumerate(truth) if f), 0)] else "")
            in_fault = False
    if in_fault:
        ax1.axvspan(fault_start, steps[-1], alpha=0.15, color=STYLE["red"])

    ax1.set_ylabel("Anomaly Score")
    ax1.set_ylim([0, 1.1])
    ax1.legend(fontsize=7, facecolor=STYLE["surface"],
               edgecolor=STYLE["border"], labelcolor=STYLE["text"])

    # Bottom: binary prediction vs ground truth
    _apply_style(ax2)
    pred_binary  = [1 if p else 0 for p in preds]
    truth_binary = [0.5 if t else 0 for t in truth]

    ax2.fill_between(steps, pred_binary, alpha=0.5,
                     color=STYLE["amber"], label="Predicted anomaly")
    ax2.fill_between(steps, truth_binary, alpha=0.4,
                     color=STYLE["red"], label="True fault")
    ax2.set_ylabel("Flag")
    ax2.set_xlabel("Timestep")
    ax2.set_ylim([0, 1.5])
    ax2.legend(fontsize=7, facecolor=STYLE["surface"],
               edgecolor=STYLE["border"], labelcolor=STYLE["text"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()
    print(f"  Saved: {output_path}")


def plot_confusion_matrix(
    tp: int, fp: int, fn: int, tn: int,
    model_name: str,
    output_path: str,
) -> None:
    """Confusion matrix as a heatmap — cleaner than text output."""
    matrix = np.array([[tp, fn], [fp, tn]])
    labels = [["TP", "FN"], ["FP", "TN"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["surface"])
    _apply_style(ax, f"Confusion Matrix — {model_name}")

    colors = np.array([
        [0.1, 0.6],   # TP green-ish, FN red-ish
        [0.8, 0.2],   # FP red-ish, TN green-ish
    ])

    im = ax.imshow(colors, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: FAULT", "Predicted: NORMAL"],
                       color=STYLE["text"], fontsize=8)
    ax.set_yticklabels(["True: FAULT", "True: NORMAL"],
                       color=STYLE["text"], fontsize=8)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i][j]}",
                   ha="center", va="center",
                   color="white", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()
    print(f"  Saved: {output_path}")


# ─── Main entry point ─────────────────────────────────────────────────────────


def generate_all_plots(
    results: list[StepResult],
    report: EvaluationReport,
    output_dir: str = "outputs/plots",
) -> None:
    """
    Generate all evaluation plots and save to output_dir.
    Call this after evaluator.evaluate() in main.py.
    """
    os.makedirs(output_dir, exist_ok=True)
    name = report.model_name
    print(f"\n[Visualizer] Generating plots → {output_dir}/")

    plot_pr_curve(
        results, name,
        os.path.join(output_dir, "pr_curve.png")
    )
    plot_roc_curve(
        results, name,
        os.path.join(output_dir, "roc_curve.png")
    )
    plot_error_distribution(
        results, name,
        os.path.join(output_dir, "error_distribution.png")
    )
    plot_anomaly_timeline(
        results, name,
        os.path.join(output_dir, "anomaly_timeline.png")
    )
    plot_confusion_matrix(
        report.confusion.TP,
        report.confusion.FP,
        report.confusion.FN,
        report.confusion.TN,
        name,
        os.path.join(output_dir, "confusion_matrix.png"),
    )
    print(f"[Visualizer] All plots saved.")