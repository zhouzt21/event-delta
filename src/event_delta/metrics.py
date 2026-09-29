"""Fair-mask feature distance metrics and R2 aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


def masked_feature_distances(candidate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    candidate, target, mask = np.asarray(candidate), np.asarray(target), np.asarray(mask, dtype=bool)
    if candidate.shape != target.shape or candidate.ndim != 3:
        raise ValueError("candidate and target must be same-shape HxWxC arrays")
    if mask.shape != candidate.shape[:2]:
        raise ValueError("mask must match the feature grid")
    left, right = candidate[mask].astype(np.float64), target[mask].astype(np.float64)
    if not len(left):
        return {"token_count": 0, "cosine_distance": None, "l1_distance": None, "l2_distance": None}
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    usable = denominator > 1e-12
    cosine = 1.0 - np.sum(left[usable] * right[usable], axis=1) / denominator[usable]
    difference = left - right
    return {"token_count": int(len(left)),
            "cosine_distance": float(np.mean(cosine)) if np.any(usable) else None,
            "l1_distance": float(np.mean(np.abs(difference))),
            "l2_distance": float(np.mean(np.linalg.norm(difference, axis=1)))}


def compare_stale_and_propagated(stale: np.ndarray, propagated: np.ndarray, target: np.ndarray,
                                 valid_mask: np.ndarray, motion_mask: np.ndarray) -> dict[str, Any]:
    """Compare both methods on exactly the propagated-valid mask."""
    valid = np.asarray(valid_mask, dtype=bool)
    motion = valid & np.asarray(motion_mask, dtype=bool)
    output: dict[str, Any] = {"valid_coverage_ratio": float(np.mean(valid)),
                              "motion_token_ratio": float(np.mean(motion))}
    for group, mask in (("all_valid", valid), ("motion", motion)):
        stale_metrics = masked_feature_distances(stale, target, mask)
        propagated_metrics = masked_feature_distances(propagated, target, mask)
        output[f"{group}_token_count"] = stale_metrics["token_count"]
        for metric in ("cosine_distance", "l1_distance", "l2_distance"):
            stale_value, propagated_value = stale_metrics[metric], propagated_metrics[metric]
            output[f"{group}_stale_{metric}"] = stale_value
            output[f"{group}_propagated_{metric}"] = propagated_value
            output[f"{group}_{metric}_relative_error_reduction"] = (
                (stale_value - propagated_value) / stale_value
                if stale_value is not None and propagated_value is not None and abs(stale_value) > 1e-12 else None)
    for metric in ("cosine_distance", "l1_distance", "l2_distance"):
        output[f"stale_{metric}"] = output[f"all_valid_stale_{metric}"]
        output[f"propagated_{metric}"] = output[f"all_valid_propagated_{metric}"]
    output["relative_error_reduction"] = output["all_valid_cosine_distance_relative_error_reduction"]
    return output


def summarize_by_offset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["offset"])].append(row)
    output = []
    metric_names = [key for row in rows for key, value in row.items()
                    if isinstance(value, (int, float)) and key not in {"anchor_frame_id", "target_frame_id", "offset"}]
    metric_names = list(dict.fromkeys(metric_names))
    for offset in sorted(groups):
        group = groups[offset]
        summary: dict[str, Any] = {"offset": offset, "window_count": len(group)}
        for metric in metric_names:
            values = [float(row[metric]) for row in group
                      if row.get(metric) is not None and math.isfinite(float(row[metric]))]
            if values:
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_median"] = float(np.median(values))
        for scope in ("all_valid", "motion"):
            stale_key = f"{scope}_stale_cosine_distance"
            propagated_key = f"{scope}_propagated_cosine_distance"
            pairs = [(row.get(stale_key), row.get(propagated_key)) for row in group]
            pairs = [(stale, propagated) for stale, propagated in pairs
                     if stale is not None and propagated is not None]
            summary[f"{scope}_cosine_win_rate"] = (float(np.mean([propagated < stale for stale, propagated in pairs]))
                                                    if pairs else None)
        output.append(summary)
    return output
