"""Pure helpers for the R3b E-RAFT feature-propagation experiment."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .flow_warp import pixel_to_feature_displacement, resize_flow_to_grid
from .metrics import masked_feature_distances


METHODS = ("stale", "raw_eraft", "masked_eraft", "gt_oracle")
DISTANCES = ("cosine_distance", "l1_distance", "l2_distance")
STAT_METHODS = ("raw_eraft", "masked_eraft", "gt_oracle")
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
    """Legacy observation diagnostic against GT-flow; unstable near a zero denominator."""
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


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _window_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["recording"]), int(row["anchor_frame_id"]))].append(row)
    return [groups[key] for key in sorted(groups)]


def _aggregate_distance(rows: list[dict[str, Any]], scope: str, method: str,
                        aggregation: str, distance_metric: str = "cosine_distance",
                        ) -> tuple[float | None, int, int]:
    key = f"{scope}_{method}_{distance_metric}"
    count_key = f"{scope}_token_count"
    windows = _window_groups(rows)
    if aggregation == "window_macro":
        values, token_count = [], 0
        for window in windows:
            observations = [_finite(row.get(key)) for row in window]
            token_count += sum(int(float(row.get(count_key, 0) or 0)) for row, value
                               in zip(window, observations) if value is not None)
            observations = [value for value in observations if value is not None]
            if observations:
                values.append(float(np.mean(observations)))
        return float(np.mean(values)) if values else None, len(values), token_count
    if aggregation == "motion_token_weighted_micro" or aggregation == "all_valid_token_weighted_micro":
        weighted_sum, token_count = 0.0, 0
        eligible_windows = set()
        for row in rows:
            value = _finite(row.get(key))
            count = int(float(row.get(count_key, 0) or 0))
            if value is None or count <= 0:
                continue
            weighted_sum += value * count
            token_count += count
            eligible_windows.add((str(row["recording"]), int(row["anchor_frame_id"])))
        return (weighted_sum / token_count if token_count else None,
                len(eligible_windows), token_count)
    raise ValueError(f"unknown aggregation: {aggregation}")


def aggregate_recovery(stale_distance: float | None, method_distance: float | None,
                       epsilon: float = 1e-12) -> float | None:
    """Ratio of aggregate distances, never a mean of observation ratios."""
    if stale_distance is None or method_distance is None or abs(stale_distance) <= epsilon:
        return None
    return float((stale_distance - method_distance) / stale_distance)


def stable_reference_ratio(stale_distance: float | None, event_distance: float | None,
                           gt_distance: float | None, epsilon: float = 1e-12) -> float | None:
    """Event absolute improvement divided by GT-flow aggregate absolute improvement."""
    if stale_distance is None or event_distance is None or gt_distance is None:
        return None
    gt_improvement = stale_distance - gt_distance
    if not all(math.isfinite(value) for value in (stale_distance, event_distance, gt_distance)):
        return None
    if gt_improvement <= epsilon:
        return None
    return float((stale_distance - event_distance) / gt_improvement)


def stable_aggregate(rows: list[dict[str, Any]], scope: str, aggregation: str,
                     identity: dict[str, Any] | None = None,
                     distance_metric: str = "cosine_distance") -> list[dict[str, Any]]:
    """Aggregate paired distances first, then calculate recovery/reference ratios."""
    identity = identity or {}
    stale, eligible, tokens = _aggregate_distance(rows, scope, "stale", aggregation, distance_metric)
    gt, gt_eligible, gt_tokens = _aggregate_distance(rows, scope, "gt_oracle", aggregation, distance_metric)
    if (eligible, tokens) != (gt_eligible, gt_tokens):
        raise ValueError("stale and GT-flow geometric reference aggregation populations differ")
    output = []
    for method in STAT_METHODS:
        distance, method_eligible, method_tokens = _aggregate_distance(
            rows, scope, method, aggregation, distance_metric)
        if (eligible, tokens) != (method_eligible, method_tokens):
            raise ValueError(f"stale and {method} aggregation populations differ")
        absolute = (stale - distance if stale is not None and distance is not None else None)
        gt_absolute = (stale - gt if stale is not None and gt is not None else None)
        ratio = (stable_reference_ratio(stale, distance, gt)
                 if method != "gt_oracle" else (1.0 if gt_absolute is not None and gt_absolute > 1e-12 else None))
        output.append({
            **identity, "scope": scope, "aggregation": aggregation,
            "distance_metric": distance_metric,
            "method": method, "method_label": ("GT-flow geometric reference"
                                                  if method == "gt_oracle" else method),
            "total_window_count": len(_window_groups(rows)), "eligible_window_count": eligible,
            "token_count": tokens, "aggregate_stale_distance": stale,
            "aggregate_method_distance": distance, "aggregate_gt_flow_distance": gt,
            "aggregate_absolute_improvement": absolute,
            "aggregate_gt_flow_absolute_improvement": gt_absolute,
            "aggregate_recovery": aggregate_recovery(stale, distance),
            "stable_reference_ratio": ratio,
            "observation_mean_ratio_used_for_decision": False,
        })
    return output


def stable_aggregate_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overall, recording, offset and activity statistics with window-safe aggregation."""
    specifications: list[tuple[str, str | None]] = [
        ("overall", None), ("recording", "recording"), ("offset", "offset"),
        ("activity", "activity_group"),
    ]
    output: list[dict[str, Any]] = []
    for group_type, key in specifications:
        grouped: dict[str, list[dict[str, Any]]] = {"overall": rows}
        if key is not None:
            grouped = defaultdict(list)
            for row in rows:
                grouped[str(row[key])].append(row)
        for group, subset in sorted(grouped.items()):
            identity = {"group_type": group_type, "group": group}
            for scope in ("all_valid", "motion"):
                modes = ("window_macro", ("motion_token_weighted_micro" if scope == "motion"
                                           else "all_valid_token_weighted_micro"))
                for mode in modes:
                    for distance_metric in DISTANCES:
                        output.extend(stable_aggregate(subset, scope, mode, identity, distance_metric))
    return output


def observation_ratio_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain mean-of-ratios solely to explain legacy spikes; never feed the gate."""
    output = []
    for scope in ("all_valid", "motion"):
        for method in ("raw_eraft", "masked_eraft"):
            key = f"{scope}_{method}_cosine_distance_oracle_gap_closed"
            values = [_finite(row.get(key)) for row in rows]
            values = [value for value in values if value is not None]
            output.append({"scope": scope, "method": method, "diagnostic": "mean_of_observation_ratios",
                           "observation_count": len(values),
                           "mean": float(np.mean(values)) if values else None,
                           "median": float(np.median(values)) if values else None,
                           "minimum": min(values) if values else None,
                           "maximum": max(values) if values else None,
                           "used_for_decision": False,
                           "warning": "unstable when per-observation GT-flow improvement is small"})
    return output


def _comparison_recovery(rows: list[dict[str, Any]], scope: str, aggregation: str,
                         baseline: str, candidate: str,
                         distance_metric: str = "cosine_distance") -> float | None:
    left, _, _ = _aggregate_distance(rows, scope, baseline, aggregation, distance_metric)
    right, _, _ = _aggregate_distance(rows, scope, candidate, aggregation, distance_metric)
    return aggregate_recovery(left, right)


def paired_window_bootstrap(rows: list[dict[str, Any]], replicates: int = 2000,
                            seed: int = 20260929) -> dict[str, Any]:
    """Paired bootstrap that samples complete windows with all of their offsets."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    windows = _window_groups(rows)
    if not windows:
        raise ValueError("bootstrap requires at least one complete window")
    comparisons = {
        "raw_event_vs_stale": ("stale", "raw_eraft"),
        "masked_event_vs_stale": ("stale", "masked_eraft"),
        "gt_flow_geometric_reference_vs_stale": ("stale", "gt_oracle"),
        "masked_event_vs_raw_event": ("raw_eraft", "masked_eraft"),
    }
    rng = np.random.default_rng(seed)
    samples: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for _ in range(replicates):
        selected = [windows[index] for index in rng.integers(0, len(windows), size=len(windows))]
        sampled_rows = []
        # Synthetic bootstrap IDs preserve duplicate draws as separate window units.
        for draw, window in enumerate(selected):
            for row in window:
                sampled_rows.append({**row, "recording": f"bootstrap-{draw}",
                                     "anchor_frame_id": draw})
        for scope in ("all_valid", "motion"):
            modes = ("window_macro", ("motion_token_weighted_micro" if scope == "motion"
                                       else "all_valid_token_weighted_micro"))
            for mode in modes:
                for distance_metric in DISTANCES:
                    for name, (baseline, candidate) in comparisons.items():
                        value = _comparison_recovery(sampled_rows, scope, mode, baseline, candidate,
                                                     distance_metric)
                        if value is not None:
                            samples[(scope, mode, distance_metric, name)].append(value)
    results = []
    for scope in ("all_valid", "motion"):
        modes = ("window_macro", ("motion_token_weighted_micro" if scope == "motion"
                                   else "all_valid_token_weighted_micro"))
        for mode in modes:
            for distance_metric in DISTANCES:
                for name, (baseline, candidate) in comparisons.items():
                    values = samples[(scope, mode, distance_metric, name)]
                    point = _comparison_recovery(rows, scope, mode, baseline, candidate, distance_metric)
                    results.append({"scope": scope, "aggregation": mode, "comparison": name,
                                    "distance_metric": distance_metric,
                                    "baseline": baseline, "candidate": candidate,
                                    "point_estimate": point,
                                    "ci95_low": (float(np.percentile(values, 2.5)) if values else None),
                                    "ci95_high": (float(np.percentile(values, 97.5)) if values else None),
                                    "valid_replicates": len(values)})
    return {"sampling_unit": "complete_window", "offsets_resampled_independently": False,
            "paired": True, "seed": seed, "replicates": replicates,
            "window_count": len(windows), "exploratory": len(windows) < 30,
            "small_sample_warning": ("Only six windows; confidence intervals are exploratory and are not a hard gate."
                                      if len(windows) < 30 else None),
            "results": results}


def _find_stat(stats: list[dict[str, Any]], group_type: str, group: str, scope: str,
               aggregation: str, method: str,
               distance_metric: str = "cosine_distance") -> dict[str, Any]:
    matches = [row for row in stats if row["group_type"] == group_type and row["group"] == group
               and row["scope"] == scope and row["aggregation"] == aggregation
               and row["method"] == method and row["distance_metric"] == distance_metric]
    if len(matches) != 1:
        raise ValueError(f"expected one stable aggregate row, found {len(matches)}")
    return matches[0]


def expanded_decision(rows: list[dict[str, Any]], recordings: list[str],
                      maximum_coverage_drop: float = 0.10,
                      stable_stats: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Apply the audited ratio-of-aggregates gate; observation ratios are diagnostic only."""
    stats = stable_stats or stable_aggregate_report(rows)
    mode = "window_macro"
    per_recording = {}
    for recording in recordings:
        value = _find_stat(stats, "recording", recording, "motion", mode,
                           "raw_eraft")["aggregate_recovery"]
        per_recording[recording] = {"recovery": value, "positive": value is not None and value > 0}
    critical = {}
    for offset in (3, 5, 9):
        raw = _find_stat(stats, "offset", str(offset), "motion", mode, "raw_eraft")["aggregate_recovery"]
        masked = _find_stat(stats, "offset", str(offset), "motion", mode,
                            "masked_eraft")["aggregate_recovery"]
        critical[str(offset)] = {"raw": raw, "masked_diagnostic": masked,
                                 "raw_positive": raw is not None and raw > 0}
    critical_pass = sum(item["raw_positive"] for item in critical.values()) >= 2
    raw_all = _find_stat(stats, "overall", "overall", "all_valid", mode,
                         "raw_eraft")["aggregate_recovery"]
    masked_all = _find_stat(stats, "overall", "overall", "all_valid", mode,
                            "masked_eraft")["aggregate_recovery"]
    raw_motion = _find_stat(stats, "overall", "overall", "motion", mode, "raw_eraft")
    masked_motion = _find_stat(stats, "overall", "overall", "motion", mode, "masked_eraft")
    gt_motion = _find_stat(stats, "overall", "overall", "motion", mode, "gt_oracle")
    mean = lambda key: (float(np.mean([float(row[key]) for row in rows if _finite(row.get(key)) is not None]))
                        if any(_finite(row.get(key)) is not None for row in rows) else None)
    raw_coverage, gt_coverage = mean("raw_eraft_coverage_ratio"), mean("gt_oracle_coverage_ratio")
    coverage_drop = (gt_coverage - raw_coverage
                     if gt_coverage is not None and raw_coverage is not None else None)
    coverage_pass = coverage_drop is not None and coverage_drop <= maximum_coverage_drop
    latency_pass = all(_finite(row.get("complete_online_delta_path_ms")) is not None for row in rows)
    consistency = {}
    for name, item in (("raw", raw_motion), ("masked", masked_motion)):
        numerator = item["aggregate_absolute_improvement"]
        denominator = item["aggregate_gt_flow_absolute_improvement"]
        recomputed = (numerator / denominator if numerator is not None and denominator is not None
                      and denominator > 1e-12 else None)
        reported = item["stable_reference_ratio"]
        consistent = ((reported is None and recomputed is None) or
                      (reported is not None and recomputed is not None
                       and math.isclose(reported, recomputed, rel_tol=1e-12, abs_tol=1e-12)))
        consistency[name] = {"reported": reported, "recomputed": recomputed, "consistent": consistent}
    criteria = {
        "aggregation_used_for_gate": "window_macro ratio-of-aggregates",
        "observation_mean_oracle_gap_closed_used": False,
        "positive_motion_recovery_by_recording": per_recording,
        "both_recordings_positive_motion_recovery": bool(per_recording)
        and all(item["positive"] for item in per_recording.values()),
        "critical_offset_recovery": critical,
        "raw_positive_at_least_two_of_offsets_3_5_9": critical_pass,
        "masked_all_valid_aggregate_recovery": masked_all,
        "masked_all_valid_positive": masked_all is not None and masked_all > 0,
        "raw_all_valid_aggregate_recovery": raw_all,
        "background_drift_warning": ("raw Event improves motion tokens but hurts all-valid tokens"
                                     if raw_all is not None and raw_all < 0 else None),
        "raw_motion_stable_reference_ratio": raw_motion["stable_reference_ratio"],
        "masked_motion_stable_reference_ratio": masked_motion["stable_reference_ratio"],
        "stable_reference_ratio_is_informational_not_bounded_at_one": True,
        "gt_flow_motion_aggregate_improvement": gt_motion["aggregate_absolute_improvement"],
        "stable_reference_ratio_consistency": consistency,
        "raw_coverage_mean": raw_coverage, "gt_flow_reference_coverage_mean": gt_coverage,
        "coverage_drop": coverage_drop, "coverage_not_significantly_lower": coverage_pass,
        "complete_delta_latency_reported": latency_pass,
        "bootstrap_ci_is_hard_gate": False,
    }
    passed = (criteria["both_recordings_positive_motion_recovery"] and critical_pass
              and criteria["masked_all_valid_positive"] and coverage_pass and latency_pass
              and all(item["consistent"] for item in consistency.values()))
    return {"status": "expanded" if passed else "stop/diagnostic", "criteria": criteria,
            "automatic_parameter_tuning": False}
