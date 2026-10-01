"""Timing contracts and conservative R4a efficiency decisions."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


REQUIRED_STAGES = (
    "rgb_decode_preprocess", "anchor_vision_encoding", "event_npz_decode", "voxelization",
    "eraft_forward", "support_mask_grid_conversion", "visual_branch_propagation",
    "visual_injection", "multimodal_prefill", "decode", "complete_query",
)


@contextmanager
def measured_stage(torch: Any, device: Any, output: dict[str, Any], name: str) -> Iterator[None]:
    """Record synchronized CPU wall and CUDA computation time separately."""
    cuda = getattr(device, "type", str(device)) == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
    start = time.perf_counter()
    try:
        yield
    finally:
        if cuda:
            end_event.record()
            torch.cuda.synchronize(device)
            gpu_ms = float(start_event.elapsed_time(end_event))
        else:
            gpu_ms = None
        output[f"{name}_cpu_wall_ms"] = (time.perf_counter() - start) * 1000.0
        output[f"{name}_gpu_ms"] = gpu_ms


def validate_complete_timing(row: dict[str, Any]) -> None:
    missing = [stage for stage in REQUIRED_STAGES
               if f"{stage}_cpu_wall_ms" not in row or f"{stage}_gpu_ms" not in row]
    if missing:
        raise ValueError(f"missing R4 timing stages: {missing}")
    total = row["complete_query_cpu_wall_ms"]
    if total is None or not math.isfinite(float(total)) or float(total) < 0:
        raise ValueError("complete query wall time must be an independently measured finite value")
    if row.get("complete_query_derived_from_stage_percentiles") is not False:
        raise ValueError("complete query time must not be assembled from stage percentiles")


def interleaved_schedule(modes: list[str], repeats: int) -> list[str]:
    """Rotate mode order on every repeat to reduce ordering bias."""
    if repeats <= 0 or not modes:
        raise ValueError("modes and repeats must be non-empty")
    output = []
    for repeat in range(repeats):
        rotation = repeat % len(modes)
        output.extend(modes[rotation:] + modes[:rotation])
    return output


def percentile_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return {"count": len(values), "mean": float(np.mean(values)) if values else None,
            "p50": float(np.percentile(values, 50)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "sum": float(np.sum(values)) if values else None}


def r4_decision(correctness_passed: bool, dense_cost_ms: float | None,
                event_cost_ms: float | None, cadence_ms: float = 100.0) -> dict[str, Any]:
    if not correctness_passed:
        return {"status": "blocked", "reason": "injection correctness failed"}
    if dense_cost_ms is None or event_cost_ms is None:
        return {"status": "blocked", "reason": "complete cost unavailable"}
    advantage = event_cost_ms < dense_cost_ms
    realtime = event_cost_ms <= cadence_ms
    if not advantage:
        status = "efficiency_negative"
    elif not realtime:
        status = "integration_passed_not_realtime"
    else:
        status = "promising_efficiency_probe"
    return {"status": status, "dense_cost_ms": dense_cost_ms, "event_cost_ms": event_cost_ms,
            "observed_cost_advantage": advantage, "packet_cadence_ms": cadence_ms,
            "keeps_up_with_packet_cadence": realtime,
            "accuracy_claim": False, "paper_success_claim": False}
