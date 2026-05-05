"""
rag_assistant/anomaly_context.py
=================================
Converts TwinSnapshot anomaly events into structured context
that can be fed to a RAG (Retrieval-Augmented Generation) system.

Design decision: Define the RAG input contract NOW, even though
the LLM integration is Phase 2. This forces us to think about
what information an LLM needs to give good maintenance advice.

The context builder produces:
  1. A structured AnomalyEvent (already in schemas.py)
  2. A natural language summary for the LLM prompt
  3. Metadata for vector retrieval (fault type, severity, etc.)

Phase 2 will add:
  - Vector store lookup of maintenance manuals
  - LLM call with retrieved context
  - Structured output (explanation + action plan)
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional

from data_pipeline.schemas import TwinSnapshot, AnomalyEvent


def build_anomaly_event(
    snapshot: TwinSnapshot,
    duration_seconds: int = 1,
    context_window: Optional[list[TwinSnapshot]] = None,
) -> AnomalyEvent:
    """
    Create an AnomalyEvent from a detected anomaly snapshot.
    
    Args:
        snapshot:         The anomalous snapshot
        duration_seconds: How long the anomaly has been active
        context_window:   Recent snapshots for temporal context
    
    Returns:
        AnomalyEvent ready for RAG system ingestion.
    """
    suspected = _infer_fault_type(snapshot)

    return AnomalyEvent(
        event_id=str(uuid.uuid4())[:8],
        detected_at=snapshot.timestamp,
        duration_seconds=duration_seconds,
        wind_speed_ms=snapshot.wind_speed_ms,
        expected_power_kw=snapshot.expected_power_kw,
        actual_power_kw=snapshot.actual_power_kw,
        deviation_pct=snapshot.deviation_pct,
        anomaly_score=snapshot.anomaly_score,
        suspected_fault=suspected or snapshot.fault_type,
        context_snapshots=context_window or [],
    )


def _infer_fault_type(snapshot: TwinSnapshot) -> Optional[str]:
    """
    Heuristic fault type inference from deviation pattern.
    
    This is pre-RAG reasoning — simple rules that give the LLM
    a starting hypothesis to refine with retrieved knowledge.
    """
    dev = snapshot.deviation_pct

    if dev < -30:
        return "efficiency_drop"  # Large negative deviation = low output
    elif dev < -10:
        return "partial_efficiency_loss"
    elif dev > 20:
        return "sensor_overcounting"  # Positive anomaly = sensor issue
    return None


def format_anomaly_prompt(event: AnomalyEvent) -> str:
    """
    Format an AnomalyEvent as a natural language prompt for an LLM.
    
    Design: This prompt will be combined with retrieved chunks from
    maintenance manuals in Phase 2. The structure here is deliberate:
      - Context first (what is happening)
      - Metrics (quantitative grounding)
      - Ask for specific outputs (structured response)
    """
    return f"""
## Wind Turbine Anomaly Report

**Detection Time:** {event.detected_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
**Event ID:** {event.event_id}
**Duration:** {event.duration_seconds} seconds

### Operating Conditions
- Wind Speed: {event.wind_speed_ms:.1f} m/s
- Expected Power Output: {event.expected_power_kw:.1f} kW
- Actual Power Output: {event.actual_power_kw:.1f} kW
- Deviation: {event.deviation_pct:+.1f}%
- Anomaly Score: {event.anomaly_score:.3f}
- Suspected Fault Type: {event.suspected_fault or 'Unknown'}

### Task
Based on the anomaly data above and the retrieved maintenance documentation:

1. **Explain** what is likely happening to this turbine in plain English
2. **Root cause analysis**: What component(s) might be failing?
3. **Urgency**: Is this an immediate shutdown risk or can it be scheduled?
4. **Recommended actions**: Step-by-step maintenance checklist
5. **Monitoring**: What metrics to watch in the next 24 hours?

Respond in structured JSON format with keys:
  explanation, root_cause, urgency_level, actions, monitoring_metrics
""".strip()


def format_anomaly_summary(event: AnomalyEvent) -> str:
    """Short one-line summary for dashboard display."""
    urgency = "🔴 CRITICAL" if abs(event.deviation_pct) > 30 else "🟡 WARNING"
    return (
        f"{urgency} | {event.suspected_fault or 'Unknown fault'} | "
        f"Power deviation: {event.deviation_pct:+.1f}% | "
        f"Score: {event.anomaly_score:.2f}"
    )
