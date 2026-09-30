"""Pure helpers for the R3b E-RAFT feature-propagation experiment."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .flow_warp import pixel_to_feature_displacement, resize_flow_to_grid
from .metrics import masked_feature_distances


METHODS = ("stale", "raw_eraft", "masked_eraft", "gt_oracle")
DISTANCES = ("cosine_distance", "l1_distance", "l2_distance")
LATENCY_FIELDS = (
    "dataset_npz_decode_ms",
    "voxelization_ms",
    "eraft_forward_ms",
    "flow_resize_grid_conversion_ms",
    "feature_warp_ms",
    "qwen_vit_ms",
    "complete_online_delta_path_ms",
)


def predicted_flow_to_feature_displacement(
    flow_pixels: Any,
    feature_hw: tuple[int, int],
    original_hw: tuple[int, int],
) -> Any:
    """Resize native pixel displacement and convert it to feature-grid cells."""
    resized = resize_flow_to_grid(flow_pixels, feature_hw)
    return pixel_to_feature_displacement(resized, original_hw, feature_hw)


def transition_frame_ids(anchor_frame_id: int, offset: int) -> tuple[int, int, int]:
    """Return source Event ID, target Event ID, and source-indexed GT-flow ID."""
    if offset < 1:
        raise ValueError("offset must be at least one")
    source = int(anchor_frame_id) + int(offset) - 1
    return source, source + 1, source


def event_support_mask(events: list[dict[str, np.ndarray]], source_hw: tuple[int, int],
                       dilation_radius: int) -> np.ndarray:
    """Union packet support in native x/y coordinates, with fixed square dilation."""
    if dilation_radius < 0:
        raise ValueError("dilation radius cannot be negative")
    height, width = source_hw
    mask = np.zeros((height, width), dtype=np.uint8)
    for packet in events:
        x = np.asarray(packet["x"]).reshape(-1).astype(np.int64)
        y = np.asarray(packet["y"]).reshape(-1).astype(np.int64)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        mask[y[valid], x[valid]] = 1
    if dilation_radius:
        size = 2 * dilation_radius + 1
        mask = np.asarray(Image.fromarray(mask * 255).filter(ImageFilter.MaxFilter(size=size))) > 0
    return mask.astype(bool)


def oracle_gap_closed(stale_error: float | None, event_error: float | None,
                      gt_error: float | None, epsilon: float = 1e-12) -> float | None:
    """Fraction of the GT-oracle improvement recovered by an Event method."""
    if stale_error is None or event_error is None or gt_error is None:
        return None
    denominator = stale_error - gt_error
    if not all(math.isfinite(value) for value in (stale_error, event_error, gt_error)):
        return None
    if denominator <= epsilon:
        return None
    return float((stale_error - event_error) / denominator)


def compare_feature_variants(stale: np.ndarray, raw_eraft: np.ndarray,
                             masked_eraft: np.ndarray, gt_oracle: np.ndarray,
                             target: np.ndarray, raw_valid: np.ndarray,
                             masked_valid: np.ndarray, gt_valid: np.ndarray,
                             motion_mask: np.ndarray) -> dict[str, Any]:
    """Evaluate every method on one intersection mask for a fair comparison."""
    arrays = (stale, raw_eraft, masked_eraft, gt_oracle, target)
    if len({np.asarray(value).shape for value in arrays}) != 1:
        raise ValueError("all feature arrays must have the same shape")
    common = (np.asarray(raw_valid, dtype=bool) & np.asarray(masked_valid, dtype=bool)
              & np.asarray(gt_valid, dtype=bool))
    motion = common & np.asarray(motion_mask, dtype=bool)
    output: dict[str, Any] = {
        "evaluation_mask_definition": "raw_eraft_valid & masked_eraft_valid & gt_oracle_valid",
        "all_valid_token_count": int(np.count_nonzero(common)),
        "motion_token_count": int(np.count_nonzero(motion)),
        "common_valid_coverage_ratio": float(np.mean(common)),
        "motion_token_ratio": float(np.mean(motion)),
        "raw_eraft_coverage_ratio": float(np.mean(raw_valid)),
        "masked_eraft_coverage_ratio": float(np.mean(masked_valid)),
        "gt_oracle_coverage_ratio": float(np.mean(gt_valid)),
    }
    candidates = dict(zip(METHODS, arrays[:4]))
    for scope, mask in (("all_valid", common), ("motion", motion)):
        measured = {method: masked_feature_distances(candidate, target, mask)
                    for method, candidate in candidates.items()}
        for method, metrics in measured.items():
            for distance in DISTANCES:
                output[f"{scope}_{method}_{distance}"] = metrics[distance]
        for distance in DISTANCES:
            stale_value = measured["stale"][distance]
            gt_value = measured["gt_oracle"][distance]
            for method in ("raw_eraft", "masked_eraft", "gt_oracle"):
                value = measured[method][distance]
                recovery = ((stale_value - value) / stale_value
                            if stale_value is not None and value is not None and abs(stale_value) > 1e-12
                            else None)
                output[f"{scope}_{method}_{distance}_recovery"] = recovery
            for method in ("raw_eraft", "masked_eraft"):
                output[f"{scope}_{method}_{distance}_oracle_gap_closed"] = oracle_gap_closed(
                    stale_value, measured[method][distance], gt_value)
    return output


def complete_online_delta_latency(latencies: dict[str, float | None]) -> float | None:
    """Sum only the observed online Event path; dense Qwen and GT are excluded."""
    components = (
        "dataset_npz_decode_ms", "voxelization_ms", "eraft_forward_ms",
        "flow_resize_grid_conversion_ms", "feature_warp_ms",
    )
    values = [latencies.get(name) for name in components]
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return None
    return float(sum(float(value) for value in values))


def validate_latency_fields(row: dict[str, Any]) -> None:
    missing = [field for field in LATENCY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"missing latency fields: {missing}")


def expanded_decision(rows: list[dict[str, Any]], recordings: list[str],
                      minimum_gap_closed: float = 0.20,
                      maximum_coverage_drop: float = 0.10) -> dict[str, Any]:
    """Apply the fixed R3b smoke-to-expanded gate without tuning."""
    def mean(key: str, subset: list[dict[str, Any]]) -> float | None:
        values = [float(row[key]) for row in subset if row.get(key) is not None
                  and math.isfinite(float(row[key]))]
        return float(np.mean(values)) if values else None

    per_recording: dict[str, bool] = {}
    for recording in recordings:
        subset = [row for row in rows if row["recording"] == recording]
        recovery = mean("motion_raw_eraft_cosine_distance_recovery", subset)
        per_recording[recording] = recovery is not None and recovery > 0
    raw_gap = mean("motion_raw_eraft_cosine_distance_oracle_gap_closed", rows)
    masked_gap = mean("motion_masked_eraft_cosine_distance_oracle_gap_closed", rows)
    gap_pass = any(value is not None and value >= minimum_gap_closed for value in (raw_gap, masked_gap))
    critical: dict[str, dict[str, float | None]] = {}
    for offset in (3, 5, 9):
        subset = [row for row in rows if int(row["offset"]) == offset]
        raw_value = mean("motion_raw_eraft_cosine_distance_recovery", subset)
        masked_value = mean("motion_masked_eraft_cosine_distance_recovery", subset)
        positive = [value for value in (raw_value, masked_value) if value is not None]
        critical[str(offset)] = {"raw": raw_value, "masked": masked_value,
                                 "best_fixed_variant": max(positive) if positive else None}
    critical_pass = sum(value["best_fixed_variant"] is not None
                        and value["best_fixed_variant"] > 0 for value in critical.values()) >= 2
    raw_coverage = mean("raw_eraft_coverage_ratio", rows)
    gt_coverage = mean("gt_oracle_coverage_ratio", rows)
    coverage_drop = (gt_coverage - raw_coverage
                     if gt_coverage is not None and raw_coverage is not None else None)
    coverage_pass = coverage_drop is not None and coverage_drop <= maximum_coverage_drop
    latency_pass = all(row.get("complete_online_delta_path_ms") is not None for row in rows)
    criteria = {
        "positive_motion_recovery_by_recording": per_recording,
        "both_recordings_positive_motion_recovery": bool(per_recording) and all(per_recording.values()),
        "raw_motion_oracle_gap_closed_mean": raw_gap,
        "masked_motion_oracle_gap_closed_mean": masked_gap,
        "one_event_variant_closes_minimum_oracle_gap": gap_pass,
        "minimum_oracle_gap_closed": minimum_gap_closed,
        "critical_offset_raw_recovery": critical,
        "at_least_two_of_offsets_3_5_9_positive": critical_pass,
        "raw_coverage_mean": raw_coverage,
        "gt_oracle_coverage_mean": gt_coverage,
        "raw_minus_gt_coverage_drop": coverage_drop,
        "maximum_allowed_coverage_drop": maximum_coverage_drop,
        "coverage_not_significantly_lower": coverage_pass,
        "complete_delta_latency_reported": latency_pass,
    }
    passed = (criteria["both_recordings_positive_motion_recovery"] and gap_pass
              and critical_pass and coverage_pass and latency_pass)
    return {"status": "expanded" if passed else "stop/diagnostic", "criteria": criteria,
            "automatic_parameter_tuning": False}
