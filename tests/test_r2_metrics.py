from pathlib import Path

import numpy as np
import pytest

from src.event_delta.flow_warp import forward_splat_numpy
from src.event_delta.metrics import (
    aggregate_motion_group,
    bootstrap_window_reductions,
    compare_stale_and_propagated,
    grouped_motion_aggregates,
    summarize_by_offset,
)


def test_stale_and_propagated_use_identical_valid_mask_and_propagation_wins():
    height, width = 4, 6
    yy, xx = np.indices((height, width), dtype=np.float32)
    stale = np.stack((xx + 1, yy + 1, (xx + 1) * (yy + 1)), axis=-1)
    displacement = np.zeros((height, width, 2), dtype=np.float32)
    displacement[..., 0] = 1
    propagated, valid, _confidence = forward_splat_numpy(stale, displacement)
    target = np.zeros_like(stale)
    target[:, 1:] = stale[:, :-1]
    motion = np.ones((height, width), dtype=bool)
    metrics = compare_stale_and_propagated(stale, propagated, target, valid, motion)
    assert metrics["all_valid_token_count"] == int(valid.sum())
    assert metrics["motion_token_count"] == int(valid.sum())
    assert metrics["all_valid_propagated_l2_distance"] == 0
    assert metrics["all_valid_propagated_cosine_distance"] < metrics["all_valid_stale_cosine_distance"]


def test_offset_summary_reports_mean_median_and_win_rate():
    rows = [
        {"offset": 1, "anchor_frame_id": 1, "target_frame_id": 2,
         "all_valid_stale_cosine_distance": 0.4, "all_valid_propagated_cosine_distance": 0.2,
         "motion_stale_cosine_distance": 0.5, "motion_propagated_cosine_distance": 0.3},
        {"offset": 1, "anchor_frame_id": 11, "target_frame_id": 12,
         "all_valid_stale_cosine_distance": 0.3, "all_valid_propagated_cosine_distance": 0.4,
         "motion_stale_cosine_distance": 0.6, "motion_propagated_cosine_distance": 0.2},
    ]
    summary = summarize_by_offset(rows)[0]
    assert summary["all_valid_stale_cosine_distance_mean"] == pytest.approx(0.35)
    assert summary["all_valid_cosine_win_rate"] == 0.5
    assert summary["motion_cosine_win_rate"] == 1.0


def test_missing_frame_or_flow_has_explicit_error(tmp_path):
    from tools.run_r2_oracle import validate_window_files

    rgb = {10: tmp_path / "10.png"}
    flow = {}
    with pytest.raises(FileNotFoundError, match="RGB frame 11"):
        validate_window_files(rgb, flow, 10, 1)
    rgb[11] = tmp_path / "11.png"
    with pytest.raises(FileNotFoundError, match="flow frame 10"):
        validate_window_files(rgb, flow, 10, 1)


def expanded_rows():
    return [
        {"recording": "fixed-3", "anchor_frame_id": 10, "offset": 1,
         "activity_group": "high", "motion_token_count": 10,
         "motion_stale_cosine_distance": 0.4, "motion_propagated_cosine_distance": 0.2},
        {"recording": "fixed-3", "anchor_frame_id": 10, "offset": 2,
         "activity_group": "high", "motion_token_count": 30,
         "motion_stale_cosine_distance": 0.6, "motion_propagated_cosine_distance": 0.3},
        {"recording": "fixed-4", "anchor_frame_id": 20, "offset": 1,
         "activity_group": "mid", "motion_token_count": 60,
         "motion_stale_cosine_distance": 0.2, "motion_propagated_cosine_distance": 0.15},
        {"recording": "fixed-4", "anchor_frame_id": 30, "offset": 1,
         "activity_group": "low", "motion_token_count": 0,
         "motion_stale_cosine_distance": None, "motion_propagated_cosine_distance": None},
    ]


def test_window_macro_and_motion_token_weighted_micro_aggregation():
    aggregate = aggregate_motion_group(expanded_rows())
    assert aggregate["window_count"] == 3
    assert aggregate["motion_window_count"] == 2
    assert aggregate["motion_token_count"] == 100
    assert aggregate["window_macro_stale_cosine_distance"] == pytest.approx(0.35)
    assert aggregate["window_macro_propagated_cosine_distance"] == pytest.approx(0.2)
    assert aggregate["motion_token_micro_stale_cosine_distance"] == pytest.approx(0.34)
    assert aggregate["motion_token_micro_propagated_cosine_distance"] == pytest.approx(0.2)


def test_no_motion_activity_group_is_na_not_failure():
    groups = grouped_motion_aggregates(expanded_rows())
    low = next(row for row in groups
               if row["group_type"] == "activity_group" and row["activity_group"] == "low")
    assert low["motion_window_count"] == 0
    assert low["motion_token_count"] == 0
    assert low["window_macro_win_rate"] is None
    assert low["window_macro_cosine_error_reduction"] is None


def test_window_bootstrap_is_deterministic_and_reports_confidence_interval():
    first = bootstrap_window_reductions(expanded_rows(), replicates=100, seed=20260929)
    second = bootstrap_window_reductions(expanded_rows(), replicates=100, seed=20260929)
    assert first == second
    assert first["sampling_unit"] == "window"
    overall = first["groups"][0]
    assert overall["window_count"] == 3
    assert overall["motion_window_count"] == 2
    assert overall["window_macro"]["valid_replicates"] <= 100
    assert overall["window_macro"]["ci95_low"] > 0


def test_total_window_allocation_is_balanced_in_cli_order():
    from tools.run_r2_oracle import allocate_window_counts, select_qualitative_anchors

    assert allocate_window_counts(["fixed-3", "fixed-4"], 60) == {"fixed-3": 30, "fixed-4": 30}
    assert allocate_window_counts(["a", "b"], 5) == {"a": 3, "b": 2}
    with pytest.raises(ValueError, match="at least one"):
        allocate_window_counts(["a", "b"], 1)
    selected = [
        {"recording": "a", "anchor_frame_id": 10, "flow_activity_mean_pixels": 0.1},
        {"recording": "a", "anchor_frame_id": 20, "flow_activity_mean_pixels": 2.0},
        {"recording": "b", "anchor_frame_id": 30, "flow_activity_mean_pixels": 0.2},
        {"recording": "b", "anchor_frame_id": 40, "flow_activity_mean_pixels": 3.0},
    ]
    assert select_qualitative_anchors(selected, ["a", "b"]) == {"a": 20, "b": 40}


def test_summary_formats_numpy_comparisons_as_json_booleans():
    from tools.run_r2_oracle import evaluate_criteria, summary_markdown

    rows = []
    for recording, anchor in (("fixed-3", 10), ("fixed-4", 20)):
        for offset in (3, 5, 9):
            rows.append({
                "recording": recording, "anchor_frame_id": anchor, "target_frame_id": anchor + offset,
                "offset": offset, "activity_group": "high", "valid_coverage_ratio": 0.9,
                "motion_token_count": 10, "motion_stale_cosine_distance": 0.4,
                "motion_propagated_cosine_distance": 0.2,
                "all_valid_stale_cosine_distance": 0.3,
                "all_valid_propagated_cosine_distance": 0.2,
            })
    offsets = summarize_by_offset(rows)
    aggregates = grouped_motion_aggregates(rows)
    bootstrap = bootstrap_window_reductions(rows, replicates=20, seed=20260929)
    efficiency = {"warp_to_vision_p50_ratio": 0.1, "theoretical_speedup": 5.0,
                  "vision_encoder": {"p50_ms": 10.0}, "single_feature_warp": {"p50_ms": 1.0}}
    decision = evaluate_criteria(rows, offsets, efficiency, 0.5, 0.25, bootstrap)
    assert all(type(value) is bool for value in decision["criteria"]["improvement_consistent_by_recording"].values())
    markdown = summary_markdown(rows, offsets, efficiency, decision, aggregates, bootstrap, 2,
                                {"fixed-3": 1, "fixed-4": 1})
    assert "np.True_" not in markdown
    assert '"fixed-3": true' in markdown


def test_high_activity_qualitative_plot_contains_required_inputs(tmp_path):
    from tools.run_r2_oracle import plot_qualitative

    height, width, channels = 4, 5, 3
    example = {
        "recording": "fixed-3", "anchor": 10, "offset": 9,
        "activity_stratum": "activity_quantile_1.00", "activity_group": "high",
        "stale": np.ones((height, width, channels)),
        "propagated": np.full((height, width, channels), 1.5),
        "target": np.full((height, width, channels), 2.0),
        "valid": np.ones((height, width), dtype=bool),
        "motion": np.ones((height, width), dtype=bool),
        "all_valid_token_count": height * width, "motion_token_count": height * width,
        "anchor_rgb": np.zeros((8, 10, 3), dtype=np.uint8),
        "target_rgb": np.full((8, 10, 3), 255, dtype=np.uint8),
        "flow_magnitude_pixels": np.ones((8, 10)),
    }
    output = tmp_path / "qualitative_features_fixed-3_high.png"
    plot_qualitative(example, output)
    assert output.is_file() and output.stat().st_size > 0
