"""
models/evaluator.py
====================
Model-agnostic evaluation framework for anomaly detection.

Completely separate from any detector implementation.
Feed it predictions + ground truth → get full evaluation report.

Supports:
  - Standard metrics: confusion matrix, precision, recall, F1
  - Fault-proximity scoring with asymmetric tolerance window
  - Prognostic horizon: how early does the model warn?
  - PR-AUC: better than ROC-AUC for imbalanced fault detection
  - Cost-sensitive scoring in business terms (€)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─── Data contracts ───────────────────────────────────────────────────────────


@dataclass
class StepResult:
    """
    Ground truth + prediction for a single timestep.
    This is what you feed into the evaluator.
    """
    step: int
    is_fault_ground_truth: bool     # From fault injector (simulation only)
    is_anomaly_predicted: bool      # From ML detector
    anomaly_score: float            # Raw score (0-1)
    fault_type: Optional[str] = None


@dataclass
class ConfusionMatrix:
    TP: int = 0     # Fault exists, model flagged it
    FP: int = 0     # No fault, model flagged it anyway
    FN: int = 0     # Fault exists, model missed it
    TN: int = 0     # No fault, model correctly silent

    @property
    def precision(self) -> float:
        """Of all flags raised, what fraction were real faults?"""
        denom = self.TP + self.FP
        return self.TP / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Of all real faults, what fraction did we catch?"""
        denom = self.TP + self.FN
        return self.TP / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.TP + self.FP + self.FN + self.TN
        return (self.TP + self.TN) / total if total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"\n  Confusion Matrix:\n"
            f"  ┌─────────────┬──────────┬──────────┐\n"
            f"  │             │ Pred: YES│ Pred: NO │\n"
            f"  ├─────────────┼──────────┼──────────┤\n"
            f"  │ True: FAULT │ TP={self.TP:5d}  │ FN={self.FN:5d}  │\n"
            f"  │ True: NORMAL│ FP={self.FP:5d}  │ TN={self.TN:5d}  │\n"
            f"  └─────────────┴──────────┴──────────┘\n"
            f"  Precision: {self.precision:.3f} | "
            f"Recall: {self.recall:.3f} | "
            f"F1: {self.f1:.3f}"
        )


@dataclass
class ProximityScore:
    """
    Results of fault-proximity (event-based) evaluation.
    Tolerant of detections that are slightly early or late.
    """
    # Window settings
    early_tolerance: int = 3    # Steps before fault = full credit
    late_tolerance: int = 2     # Steps after fault onset = partial credit

    # Results
    total_fault_events: int = 0
    detected_events: int = 0        # Events caught within tolerance
    missed_events: int = 0          # Events completely missed
    avg_proximity_score: float = 0.0  # Weighted score 0-1
    avg_prognostic_horizon: float = 0.0  # Avg steps of early warning

    @property
    def event_recall(self) -> float:
        """Fraction of fault events detected (with tolerance)."""
        if self.total_fault_events == 0:
            return 0.0
        return self.detected_events / self.total_fault_events

    def __str__(self) -> str:
        return (
            f"\n  Proximity Scoring (window: -{self.early_tolerance}/+{self.late_tolerance} steps):\n"
            f"  Fault events:        {self.total_fault_events}\n"
            f"  Detected (tolerant): {self.detected_events} "
            f"({self.event_recall*100:.1f}%)\n"
            f"  Missed:              {self.missed_events}\n"
            f"  Avg proximity score: {self.avg_proximity_score:.3f}\n"
            f"  Avg prognostic horizon: {self.avg_prognostic_horizon:.1f} steps"
        )


@dataclass
class CostReport:
    """Business cost evaluation of detection performance."""
    cost_per_missed_fault: float = 500_000    # € per undetected fault event
    cost_per_false_alarm: float = 2_000       # € per unnecessary callout

    missed_faults: int = 0
    false_alarms: int = 0

    @property
    def total_cost(self) -> float:
        return (
            self.missed_faults * self.cost_per_missed_fault +
            self.false_alarms * self.cost_per_false_alarm
        )

    @property
    def missed_fault_cost(self) -> float:
        return self.missed_faults * self.cost_per_missed_fault

    @property
    def false_alarm_cost(self) -> float:
        return self.false_alarms * self.cost_per_false_alarm

    def __str__(self) -> str:
        return (
            f"\n  Cost-Sensitive Evaluation:\n"
            f"  Missed faults:    {self.missed_faults} × "
            f"€{self.cost_per_missed_fault:,.0f} = "
            f"€{self.missed_fault_cost:,.0f}\n"
            f"  False alarms:     {self.false_alarms} × "
            f"€{self.cost_per_false_alarm:,.0f} = "
            f"€{self.false_alarm_cost:,.0f}\n"
            f"  ─────────────────────────────────────\n"
            f"  Total cost:       €{self.total_cost:,.0f}"
        )


@dataclass
class EvaluationReport:
    """Full evaluation report combining all metrics."""
    model_name: str
    n_steps: int
    confusion: ConfusionMatrix
    proximity: ProximityScore
    cost: CostReport
    pr_auc: float = 0.0

    def __str__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  Evaluation Report — {self.model_name}\n"
            f"  Steps evaluated: {self.n_steps}\n"
            f"{'='*60}"
            f"{self.confusion}"
            f"{self.proximity}"
            f"\n  PR-AUC: {self.pr_auc:.4f}"
            f"{self.cost}\n"
            f"{'='*60}"
        )


# ─── Evaluator ────────────────────────────────────────────────────────────────


class AnomalyEvaluator:
    """
    Model-agnostic evaluator. Feed it StepResults, get a full report.

    Usage:
        evaluator = AnomalyEvaluator(model_name="LSTM Autoencoder")
        for snapshot, sensor, prediction in zip(...):
            evaluator.add(StepResult(
                step=i,
                is_fault_ground_truth=sensor.is_fault_injected,
                is_anomaly_predicted=prediction.is_anomaly,
                anomaly_score=prediction.score,
                fault_type=sensor.fault_type,
            ))
        report = evaluator.evaluate()
        print(report)
    """

    def __init__(
        self,
        model_name: str = "Unknown",
        early_tolerance: int = 3,
        late_tolerance: int = 2,
        cost_per_missed_fault: float = 500_000,
        cost_per_false_alarm: float = 2_000,
    ):
        self.model_name = model_name
        self.early_tolerance = early_tolerance
        self.late_tolerance = late_tolerance
        self.cost_per_missed_fault = cost_per_missed_fault
        self.cost_per_false_alarm = cost_per_false_alarm
        self._results: list[StepResult] = []

    def add(self, result: StepResult) -> None:
        """Add one timestep result."""
        self._results.append(result)

    def add_batch(self, results: list[StepResult]) -> None:
        """Add multiple results at once."""
        self._results.extend(results)

    def evaluate(self) -> EvaluationReport:
        """Run full evaluation and return report."""
        if not self._results:
            raise ValueError("No results to evaluate")

        confusion = self._compute_confusion()
        proximity = self._compute_proximity()
        cost = self._compute_cost(confusion, proximity)
        pr_auc = self._compute_pr_auc()

        return EvaluationReport(
            model_name=self.model_name,
            n_steps=len(self._results),
            confusion=confusion,
            proximity=proximity,
            cost=cost,
            pr_auc=pr_auc,
        )
    
    def optimal_threshold(self) -> tuple[float, float, float, float]:
        """
        Sweep all possible thresholds and find the one that
        maximises F1 score on this dataset.

        Returns: (threshold, precision, recall, f1) at optimal point
        """
        sorted_results = sorted(
            self._results, key=lambda r: r.anomaly_score, reverse=True
        )
        n_positive = sum(1 for r in self._results if r.is_fault_ground_truth)
        if n_positive == 0:
            return (0.5, 0.0, 0.0, 0.0)
        
        tp = 0
        fp = 0
        best_f1 = 0.0
        best = (0.5, 0.0, 0.0, 0.0)

        for r in sorted_results:
            if r.is_fault_ground_truth:
                tp += 1
            else:
                fp += 1
            
            p = tp / (tp + fp)
            rec = tp / n_positive
            f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best = (r.anomaly_score, p, rec, f1)
        return best
    
    def cost_optimal_threshold(
            self,
            cost_per_missed_fault: float = 500_000,
            cost_per_false_alarm: float = 2_000,
        ) -> tuple[float, float, float, float]:
        """
        Find threshold that minimises expected business cost.

        This is different from F1-optimal — it weights missed faults
        vs false alarms by their actual business cost ratio.

        Returns:
           (threshold, expected_cost) at optimal point
        """
        sorted_results = sorted(
            self._results, key=lambda r: r.anomaly_score, reverse=True
        )
        n_positive = sum(1 for r in self._results if r.is_fault_ground_truth)
        if n_positive == 0:
            return (0.5, float("inf"))
        
        tp = 0
        fp = 0
        best_cost = float("inf")
        best_treshold = 0.5

        for r in sorted_results:
            if r.is_fault_ground_truth:
                tp += 1
            else:
                fp += 1
            
            fn = n_positive - tp
            expected_cost = fn * cost_per_missed_fault + fp * cost_per_false_alarm

            if expected_cost < best_cost:
                best_cost = expected_cost
                best_treshold = r.anomaly_score

        return (best_treshold, best_cost)

    def _compute_confusion(self) -> ConfusionMatrix:
        """Standard point-based confusion matrix."""
        cm = ConfusionMatrix()
        for r in self._results:
            if r.is_fault_ground_truth and r.is_anomaly_predicted:
                cm.TP += 1
            elif not r.is_fault_ground_truth and r.is_anomaly_predicted:
                cm.FP += 1
            elif r.is_fault_ground_truth and not r.is_anomaly_predicted:
                cm.FN += 1
            else:
                cm.TN += 1
        return cm

    def _find_fault_events(self) -> list[tuple[int, int]]:
        """
        Find contiguous blocks of fault steps.
        Returns list of (start_step, end_step) tuples.

        Example: steps [0,1,2, 10,11,12,13] → [(0,2), (10,13)]
        """
        events = []
        in_fault = False
        start = 0

        for r in self._results:
            if r.is_fault_ground_truth and not in_fault:
                in_fault = True
                start = r.step
            elif not r.is_fault_ground_truth and in_fault:
                in_fault = False
                events.append((start, r.step - 1))

        if in_fault:
            events.append((start, self._results[-1].step))

        return events

    def _compute_proximity(self) -> ProximityScore:
        """
        Event-based proximity scoring with asymmetric tolerance.

        For each fault event:
          - Look for any prediction in window [start - early, start + late]
          - Score based on how early the detection was
          - Track prognostic horizon for early detections
        """
        fault_events = self._find_fault_events()
        step_to_result = {r.step: r for r in self._results}

        proximity_scores = []
        prognostic_horizons = []
        detected = 0

        for fault_start, fault_end in fault_events:
            # Search window: early_tolerance before → late_tolerance after onset
            window_start = fault_start - self.early_tolerance
            window_end = fault_start + self.late_tolerance

            best_score = 0.0
            best_horizon = None

            for step in range(window_start, window_end + 1):
                result = step_to_result.get(step)
                if result is None or not result.is_anomaly_predicted:
                    continue

                # Calculate score based on timing
                if step <= fault_start:
                    # Early detection — full credit
                    score = 1.0
                    horizon = fault_start - step  # Steps of early warning
                    if best_horizon is None or horizon > best_horizon:
                        best_horizon = horizon
                elif step <= fault_start + 1:
                    # Slightly late
                    score = 0.8
                else:
                    # Further late but within tolerance
                    score = 0.5

                best_score = max(best_score, score)

            if best_score > 0:
                detected += 1
                proximity_scores.append(best_score)
                if best_horizon is not None:
                    prognostic_horizons.append(best_horizon)

        total = len(fault_events)
        missed = total - detected
        avg_score = sum(proximity_scores) / len(proximity_scores) if proximity_scores else 0.0
        avg_ph = sum(prognostic_horizons) / len(prognostic_horizons) if prognostic_horizons else 0.0

        return ProximityScore(
            early_tolerance=self.early_tolerance,
            late_tolerance=self.late_tolerance,
            total_fault_events=total,
            detected_events=detected,
            missed_events=missed,
            avg_proximity_score=avg_score,
            avg_prognostic_horizon=avg_ph,
        )

    def _compute_cost(
        self, confusion: ConfusionMatrix, proximity: ProximityScore
    ) -> CostReport:
        """
        Business cost evaluation.
        Uses event-level missed faults (not step-level FN).
        """
        return CostReport(
            cost_per_missed_fault=self.cost_per_missed_fault,
            cost_per_false_alarm=self.cost_per_false_alarm,
            missed_faults=proximity.missed_events,
            false_alarms=confusion.FP,
        )

    def _compute_pr_auc(self) -> float:
        """
        Compute PR-AUC using trapezoidal integration.
        Better than ROC-AUC for imbalanced datasets.
        """
        # Sort by anomaly score descending
        sorted_results = sorted(
            self._results, key=lambda r: r.anomaly_score, reverse=True
        )

        n_positive = sum(1 for r in self._results if r.is_fault_ground_truth)
        if n_positive == 0:
            return 0.0

        tp = 0
        fp = 0
        precisions = []
        recalls = []

        for r in sorted_results:
            if r.is_fault_ground_truth:
                tp += 1
            else:
                fp += 1
            precisions.append(tp / (tp + fp))
            recalls.append(tp / n_positive)

        # Trapezoidal integration
        auc = 0.0
        for i in range(1, len(recalls)):
            auc += (recalls[i] - recalls[i-1]) * (precisions[i] + precisions[i-1]) / 2

        return round(abs(auc), 4)
    
    