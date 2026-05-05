"""
api/server.py
==============
FastAPI backend for the Wind Turbine Digital Twin visualization.

Startup sequence:
  1. Run 50 Monte Carlo simulations using the real multi-scale VAE
  2. Store full time-series data in memory
  3. Serve via REST endpoints to the Three.js frontend

Run:
  pip install fastapi uvicorn
  python api/server.py

Then open: http://localhost:8000
"""

from __future__ import annotations
import os
import sys
import json
import time
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from twin_engine.twin import DigitalTwin


# ── Config ────────────────────────────────────────────────────────────────────

N_RUNS    = 50
N_STEPS   = 3600
CONFIG    = "config/turbine_config.yaml"
HTML_FILE = Path(__file__).parent / "static" / "windmill_viz.html"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class StepData:
    step:            int
    wind_speed:      float
    expected_power:  float
    actual_power:    float
    efficiency_ratio: float
    deviation_pct:   float
    anomaly_score:   float
    is_anomaly:      bool
    is_fault:        bool
    fault_type:      Optional[str]


@dataclass
class RunStats:
    run_idx:    int
    precision:  float
    recall:     float
    f1:         float
    pr_auc:     float
    fault_steps: int
    tp:         int
    fp:         int
    fn:         int
    cost:       float


@dataclass
class RunData:
    run_idx: int
    stats:   RunStats
    steps:   list[StepData]


# ── Monte Carlo runner ────────────────────────────────────────────────────────

def run_single_simulation(run_idx: int) -> RunData:
    """Run one simulation with a unique random seed."""
    twin = DigitalTwin(config_path=CONFIG)

    # Set per-run seeds for reproducible variation
    twin.wind_gen._rng.seed(run_idx * 42)
    twin.fault_injector._rng.seed(run_idx * 99)
    twin.sensor_sim._rng.seed(run_idx * 7)
    torch.manual_seed(run_idx * 13)
    np.random.seed(run_idx * 13)

    snapshots = twin.run_batch(N_STEPS)

    steps = []
    for i, snap in enumerate(snapshots):
        steps.append(StepData(
            step=i,
            wind_speed=round(snap.wind_speed_ms, 2),
            expected_power=round(snap.expected_power_kw, 1),
            actual_power=round(snap.actual_power_kw, 1),
            efficiency_ratio=round(snap.efficiency_ratio, 4),
            deviation_pct=round(snap.deviation_pct, 2),
            anomaly_score=round(snap.anomaly_score, 4),
            is_anomaly=snap.is_anomaly,
            is_fault=snap.fault_type is not None,
            fault_type=snap.fault_type,
        ))

    # Compute stats
    fault_steps  = [s for s in steps if s.is_fault]
    tp = sum(1 for s in fault_steps if s.is_anomaly)
    fp = sum(1 for s in steps if not s.is_fault and s.is_anomaly)
    fn = sum(1 for s in fault_steps if not s.is_anomaly)
    tn = sum(1 for s in steps if not s.is_fault and not s.is_anomaly)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    cost      = fn * 500_000 + fp * 2_000

    # Simple PR-AUC approximation
    scores_labels = [(s.anomaly_score, s.is_fault) for s in steps]
    scores_labels.sort(key=lambda x: -x[0])
    n_pos = sum(1 for _, l in scores_labels if l)
    n_neg = len(scores_labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        running_tp = running_fp = 0
        prev_recall = 0
        pr_auc = 0.0
        for score, label in scores_labels:
            if label: running_tp += 1
            else:     running_fp += 1
            p = running_tp / (running_tp + running_fp)
            r = running_tp / n_pos
            pr_auc += p * (r - prev_recall)
            prev_recall = r
    else:
        pr_auc = 0.0

    stats = RunStats(
        run_idx=run_idx,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        pr_auc=round(pr_auc, 4),
        fault_steps=len(fault_steps),
        tp=tp, fp=fp, fn=fn,
        cost=cost,
    )

    return RunData(run_idx=run_idx, stats=stats, steps=steps)


def precompute_all_runs() -> list[RunData]:
    """Run all Monte Carlo simulations at startup."""
    print(f"\n{'='*55}")
    print(f"  Precomputing {N_RUNS} Monte Carlo runs × {N_STEPS} steps")
    print(f"  Model: Multi-scale Denoising VAE (30s + 120s)")
    print(f"{'='*55}\n")

    runs = []
    total_start = time.time()

    for i in range(N_RUNS):
        run_start = time.time()
        print(f"  Run {i+1:2d}/{N_RUNS}...", end=" ", flush=True)
        run = run_single_simulation(i)
        elapsed = time.time() - run_start
        print(f"F1={run.stats.f1:.3f}  "
              f"Recall={run.stats.recall:.3f}  "
              f"Faults={run.stats.fault_steps}  "
              f"({elapsed:.1f}s)")
        runs.append(run)

    total = time.time() - total_start
    f1s = [r.stats.f1 for r in runs]
    print(f"\n  Complete in {total:.0f}s")
    print(f"  F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"{'='*55}\n")
    return runs


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Wind Turbine Digital Twin API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Precomputed data (populated at startup)
_runs: list[RunData] = []


@app.on_event("startup")
async def startup():
    global _runs
    _runs = precompute_all_runs()
    print("API ready. Open http://localhost:8000\n")


@app.get("/")
async def serve_frontend():
    """Serve the Three.js visualization."""
    if HTML_FILE.exists():
        return FileResponse(HTML_FILE)
    raise HTTPException(404, "Frontend not found. Put windmill_viz.html in api/static/")


@app.get("/api/runs/summary")
async def get_runs_summary():
    """Summary stats for all runs — used to color run selector buttons."""
    return JSONResponse([
        {
            "run_idx":   r.run_idx,
            "precision": r.stats.precision,
            "recall":    r.stats.recall,
            "f1":        r.stats.f1,
            "pr_auc":    r.stats.pr_auc,
            "fault_steps": r.stats.fault_steps,
            "tp": r.stats.tp,
            "fp": r.stats.fp,
            "fn": r.stats.fn,
            "cost": r.stats.cost,
        }
        for r in _runs
    ])


@app.get("/api/runs/{run_idx}/full")
async def get_run_full(run_idx: int):
    """Full time-series data for one run."""
    if run_idx < 0 or run_idx >= len(_runs):
        raise HTTPException(404, f"Run {run_idx} not found")
    r = _runs[run_idx]
    return JSONResponse({
        "run_idx": r.run_idx,
        "stats":   asdict(r.stats),
        "steps":   [asdict(s) for s in r.steps],
    })


@app.get("/api/runs/{run_idx}/summary")
async def get_run_summary(run_idx: int):
    """Stats only for one run (no time series)."""
    if run_idx < 0 or run_idx >= len(_runs):
        raise HTTPException(404, f"Run {run_idx} not found")
    return JSONResponse(asdict(_runs[run_idx].stats))


@app.get("/api/monte-carlo/stats")
async def get_monte_carlo_stats():
    """Aggregate Monte Carlo statistics across all runs."""
    f1s        = [r.stats.f1        for r in _runs]
    precisions = [r.stats.precision for r in _runs]
    recalls    = [r.stats.recall    for r in _runs]
    pr_aucs    = [r.stats.pr_auc    for r in _runs]
    costs      = [r.stats.cost      for r in _runs]

    def stat(arr):
        a = np.array(arr)
        return {
            "mean": round(float(a.mean()), 4),
            "std":  round(float(a.std()),  4),
            "cv":   round(float(a.std() / a.mean()) if a.mean() > 0 else 0, 4),
            "min":  round(float(a.min()),  4),
            "max":  round(float(a.max()),  4),
        }

    return JSONResponse({
        "n_runs":    N_RUNS,
        "n_steps":   N_STEPS,
        "precision": stat(precisions),
        "recall":    stat(recalls),
        "f1":        stat(f1s),
        "pr_auc":    stat(pr_aucs),
        "cost":      stat(costs),
    })


@app.get("/api/health")
async def health():
    return {"status": "ok", "runs_ready": len(_runs)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
