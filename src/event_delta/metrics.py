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


def _motion_window_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse offset rows to independent window records for macro/bootstrap statistics."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["recording"]), int(row["anchor_frame_id"]))].append(row)
    output = []
    for (recording, anchor), group in groups.items():
        valid = [row for row in group
                 if row.get("motion_stale_cosine_distance") is not None
                 and row.get("motion_propagated_cosine_distance") is not None
                 and int(row.get("motion_token_count", 0)) > 0]
        token_count = sum(int(row["motion_token_count"]) for row in valid)
        output.append({
            "recording": recording,
            "anchor_frame_id": anchor,
            "motion_offset_count": len(valid),
            "motion_token_count": token_count,
            "stale_macro": (float(np.mean([row["motion_stale_cosine_distance"] for row in valid]))
                            if valid else None),
            "propagated_macro": (float(np.mean([row["motion_propagated_cosine_distance"] for row in valid]))
                                 if valid else None),
            "stale_token_sum": float(sum(row["motion_stale_cosine_distance"]
                                          * int(row["motion_token_count"]) for row in valid)),
            "propagated_token_sum": float(sum(row["motion_propagated_cosine_distance"]
                                               * int(row["motion_token_count"]) for row in valid)),
        })
    return output


def _relative_reduction(stale: float | None, propagated: float | None) -> float | None:
    if stale is None or propagated is None or abs(stale) <= 1e-12:
        return None
    return float((stale - propagated) / stale)


def aggregate_motion_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return window-equal macro and motion-token weighted micro cosine statistics."""
    windows = _motion_window_records(rows)
    motion_windows = [window for window in windows if window["motion_offset_count"] > 0]
    token_count = sum(window["motion_token_count"] for window in motion_windows)
    stale_macro = (float(np.mean([window["stale_macro"] for window in motion_windows]))
                   if motion_windows else None)
    propagated_macro = (float(np.mean([window["propagated_macro"] for window in motion_windows]))
                        if motion_windows else None)
    stale_micro = (sum(window["stale_token_sum"] for window in motion_windows) / token_count
                   if token_count else None)
    propagated_micro = (sum(window["propagated_token_sum"] for window in motion_windows) / token_count
                        if token_count else None)
    return {
        "window_count": len(windows),
        "motion_window_count": len(motion_windows),
        "motion_token_count": int(token_count),
        "motion_offset_observation_count": sum(window["motion_offset_count"] for window in motion_windows),
        "window_macro_stale_cosine_distance": stale_macro,
        "window_macro_propagated_cosine_distance": propagated_macro,
        "window_macro_cosine_error_reduction": _relative_reduction(stale_macro, propagated_macro),
        "window_macro_win_rate": (float(np.mean([window["propagated_macro"] < window["stale_macro"]
                                                  for window in motion_windows]))
                                  if motion_windows else None),
        "motion_token_micro_stale_cosine_distance": stale_micro,
        "motion_token_micro_propagated_cosine_distance": propagated_micro,
        "motion_token_micro_cosine_error_reduction": _relative_reduction(stale_micro, propagated_micro),
    }


def grouped_motion_aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build overall, recording, activity, and offset views from per-window offset rows."""
    specifications = [
        ("overall", ()),
        ("recording", ("recording",)),
        ("activity_group", ("activity_group",)),
        ("offset", ("offset",)),
        ("recording_activity_group", ("recording", "activity_group")),
        ("recording_offset", ("recording", "offset")),
    ]
    output = []
    for group_type, keys in specifications:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        if keys:
            for row in rows:
                groups[tuple(row.get(key) for key in keys)].append(row)
        else:
            groups[()] = rows
        for values, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
            identity = {"group_type": group_type, "recording": None,
                        "activity_group": None, "offset": None}
            identity.update(dict(zip(keys, values)))
            output.append({**identity, **aggregate_motion_group(group)})
    return output


def _bootstrap_summary(values: list[float], point_estimate: float | None) -> dict[str, Any]:
    if not values:
        return {"point_estimate": point_estimate, "bootstrap_mean": None,
                "bootstrap_median": None, "ci95_low": None, "ci95_high": None,
                "valid_replicates": 0}
    return {"point_estimate": point_estimate,
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_median": float(np.median(values)),
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
            "valid_replicates": len(values)}


def bootstrap_window_reductions(rows: list[dict[str, Any]], replicates: int = 2000,
                                seed: int = 20260929) -> dict[str, Any]:
    """Bootstrap cosine reduction by resampling windows, never individual tokens/offset rows."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    group_rows = [("overall", rows)]
    for recording in sorted({str(row["recording"]) for row in rows}):
        group_rows.append((recording, [row for row in rows if str(row["recording"]) == recording]))
    results = []
    for name, group in group_rows:
        windows = _motion_window_records(group)
        point = aggregate_motion_group(group)
        macro_values: list[float] = []
        micro_values: list[float] = []
        rng = np.random.default_rng(seed)
        for _ in range(replicates):
            sampled = [windows[index] for index in rng.integers(0, len(windows), size=len(windows))]
            motion = [window for window in sampled if window["motion_offset_count"] > 0]
            if not motion:
                continue
            macro = _relative_reduction(
                float(np.mean([window["stale_macro"] for window in motion])),
                float(np.mean([window["propagated_macro"] for window in motion])))
            if macro is not None:
                macro_values.append(macro)
            tokens = sum(window["motion_token_count"] for window in motion)
            micro = _relative_reduction(
                sum(window["stale_token_sum"] for window in motion) / tokens,
                sum(window["propagated_token_sum"] for window in motion) / tokens)
            if micro is not None:
                micro_values.append(micro)
        results.append({
            "group": name,
            "window_count": point["window_count"],
            "motion_window_count": point["motion_window_count"],
            "motion_token_count": point["motion_token_count"],
            "window_macro": _bootstrap_summary(
                macro_values, point["window_macro_cosine_error_reduction"]),
            "motion_token_weighted_micro": _bootstrap_summary(
                micro_values, point["motion_token_micro_cosine_error_reduction"]),
        })
    return {"sampling_unit": "window", "seed": seed, "replicates": replicates,
            "confidence_interval": "percentile 95%", "groups": results}
