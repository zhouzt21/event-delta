from __future__ import annotations

import csv
import json

import numpy as np

from src.event_delta.flow_warp import forward_splat_numpy
from src.event_delta.json_io import atomic_write_json
from src.event_delta.r3_propagation import (
    LATENCY_FIELDS,
    compare_feature_variants,
    complete_online_delta_latency,
    event_support_mask,
    expanded_decision,
    observation_ratio_diagnostics,
    oracle_gap_closed,
    paired_window_bootstrap,
    predicted_flow_to_feature_displacement,
    stable_aggregate,
    stable_aggregate_report,
    stable_reference_ratio,
    transition_frame_ids,
    validate_latency_fields,
)
from tools.run_r2_oracle import write_csv
from tools.audit_r3_statistics import main as audit_statistics_main, read_metrics_csv


def test_predicted_flow_feature_grid_scale() -> None:
    flow = np.zeros((8, 12, 2), dtype=np.float32)
    flow[..., 0] = 4.0
    flow[..., 1] = -2.0
    displacement = predicted_flow_to_feature_displacement(flow, (2, 3), (8, 12))
    assert displacement.shape == (2, 3, 2)
    np.testing.assert_allclose(displacement[..., 0], 1.0)
    np.testing.assert_allclose(displacement[..., 1], -0.5)


def test_recurrent_transition_mapping() -> None:
    assert transition_frame_ids(4591, 1) == (4591, 4592, 4591)
    assert transition_frame_ids(4591, 9) == (4599, 4600, 4599)


def test_feature_variants_share_one_evaluation_mask() -> None:
    target = np.ones((2, 3, 2), dtype=np.float32)
    stale = np.zeros_like(target)
    raw = target.copy()
    masked = target.copy()
    gt = target.copy()
    raw_valid = np.asarray([[1, 1, 1], [1, 1, 1]], dtype=bool)
    masked_valid = np.asarray([[1, 0, 1], [1, 1, 1]], dtype=bool)
    gt_valid = np.asarray([[1, 1, 1], [0, 1, 1]], dtype=bool)
    motion = np.asarray([[1, 1, 0], [1, 0, 1]], dtype=bool)
    metrics = compare_feature_variants(stale, raw, masked, gt, target,
                                       raw_valid, masked_valid, gt_valid, motion)
    assert metrics["all_valid_token_count"] == 4
    assert metrics["motion_token_count"] == 2
    assert metrics["evaluation_mask_definition"].startswith("raw_eraft_valid")
    for method in ("stale", "raw_eraft", "masked_eraft", "gt_oracle"):
        assert f"all_valid_{method}_l1_distance" in metrics


def test_oracle_gap_closed_handles_zero_or_negative_denominator() -> None:
    assert oracle_gap_closed(1.0, 0.6, 0.2) == 0.5
    assert oracle_gap_closed(1.0, 0.5, 1.0) is None
    assert oracle_gap_closed(1.0, 0.5, 1.1) is None
    assert oracle_gap_closed(None, 0.5, 0.2) is None


def test_latency_schema_and_online_sum() -> None:
    row = {field: float(index + 1) for index, field in enumerate(LATENCY_FIELDS)}
    validate_latency_fields(row)
    components = {"dataset_npz_decode_ms": 1.0, "voxelization_ms": 2.0,
                  "eraft_forward_ms": 3.0, "flow_resize_grid_conversion_ms": 4.0,
                  "feature_warp_ms": 5.0, "qwen_vit_ms": 100.0}
    assert complete_online_delta_latency(components) == 15.0
    components["eraft_forward_ms"] = None
    assert complete_online_delta_latency(components) is None


def test_cpu_synthetic_recurrent_regression_and_support_mask() -> None:
    features = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    flow = np.zeros((3, 4, 2), dtype=np.float32)
    first, valid1, confidence1 = forward_splat_numpy(features, flow)
    second, valid2, _confidence2 = forward_splat_numpy(first, flow, confidence1)
    np.testing.assert_allclose(second, features)
    assert valid1.all() and valid2.all()
    packet = {"x": np.asarray([1]), "y": np.asarray([1])}
    support = event_support_mask([packet], (3, 4), dilation_radius=1)
    assert support.sum() == 9
    masked_flow = flow.copy()
    masked_flow[~support] = 0
    assert np.isfinite(masked_flow).all()


def synthetic_stat_rows() -> list[dict[str, object]]:
    rows = []
    for recording in ("fixed-3", "fixed-4"):
        for window_index, activity in enumerate(("low", "mid", "high")):
            for offset in range(1, 10):
                has_motion = activity != "low"
                row = {
                    "recording": recording, "anchor_frame_id": 1000 + window_index * 100,
                    "offset": offset, "activity_group": activity,
                    "all_valid_token_count": 20, "motion_token_count": 5 if has_motion else 0,
                    "all_valid_stale_cosine_distance": 1.0,
                    "all_valid_raw_eraft_cosine_distance": 1.1,
                    "all_valid_masked_eraft_cosine_distance": 0.9,
                    "all_valid_gt_oracle_cosine_distance": 0.98,
                    "motion_stale_cosine_distance": 1.0 if has_motion else None,
                    "motion_raw_eraft_cosine_distance": 0.6 if has_motion else None,
                    "motion_masked_eraft_cosine_distance": 0.65 if has_motion else None,
                    "motion_gt_oracle_cosine_distance": 0.7 if has_motion else None,
                    "motion_raw_eraft_cosine_distance_oracle_gap_closed": 4.0 if has_motion else None,
                    "motion_masked_eraft_cosine_distance_oracle_gap_closed": 3.5 if has_motion else None,
                    "all_valid_raw_eraft_cosine_distance_oracle_gap_closed": -5.0,
                    "all_valid_masked_eraft_cosine_distance_oracle_gap_closed": 5.0,
                    "raw_eraft_coverage_ratio": 0.9,
                    "gt_oracle_coverage_ratio": 0.92,
                    "complete_online_delta_path_ms": 12.5,
                }
                for scope in ("all_valid", "motion"):
                    for method in ("stale", "raw_eraft", "masked_eraft", "gt_oracle"):
                        cosine = row[f"{scope}_{method}_cosine_distance"]
                        row[f"{scope}_{method}_l1_distance"] = cosine
                        row[f"{scope}_{method}_l2_distance"] = cosine
                rows.append(row)
    return rows


def test_ratio_of_aggregates_and_small_denominator_counterexample() -> None:
    rows = [
        {"recording": "r", "anchor_frame_id": 1, "all_valid_token_count": 1,
         "all_valid_stale_cosine_distance": 1.0, "all_valid_raw_eraft_cosine_distance": 0.5,
         "all_valid_masked_eraft_cosine_distance": 0.5, "all_valid_gt_oracle_cosine_distance": 0.99,
         "all_valid_raw_eraft_cosine_distance_oracle_gap_closed": 50.0,
         "all_valid_masked_eraft_cosine_distance_oracle_gap_closed": 50.0},
        {"recording": "r", "anchor_frame_id": 2, "all_valid_token_count": 1,
         "all_valid_stale_cosine_distance": 1.0, "all_valid_raw_eraft_cosine_distance": 0.9,
         "all_valid_masked_eraft_cosine_distance": 0.9, "all_valid_gt_oracle_cosine_distance": 0.5,
         "all_valid_raw_eraft_cosine_distance_oracle_gap_closed": 0.2,
         "all_valid_masked_eraft_cosine_distance_oracle_gap_closed": 0.2},
    ]
    stable = stable_aggregate(rows, "all_valid", "window_macro")
    raw = next(row for row in stable if row["method"] == "raw_eraft")
    assert np.isclose(raw["stable_reference_ratio"], 0.3 / 0.255)
    diagnostic = observation_ratio_diagnostics(rows)
    raw_diagnostic = next(row for row in diagnostic
                          if row["scope"] == "all_valid" and row["method"] == "raw_eraft")
    assert np.isclose(raw_diagnostic["mean"], 25.1)
    assert raw_diagnostic["used_for_decision"] is False


def test_nonpositive_gt_improvement_makes_reference_ratio_na() -> None:
    assert stable_reference_ratio(1.0, 0.5, 1.0) is None
    assert stable_reference_ratio(1.0, 0.5, 1.1) is None


def test_window_bootstrap_and_recording_overall_aggregation() -> None:
    rows = synthetic_stat_rows()
    bootstrap = paired_window_bootstrap(rows, replicates=20, seed=7)
    assert bootstrap["window_count"] == 6
    assert bootstrap["offsets_resampled_independently"] is False
    assert bootstrap["exploratory"] is True
    stats = stable_aggregate_report(rows)
    recordings = {row["group"] for row in stats if row["group_type"] == "recording"}
    assert recordings == {"fixed-3", "fixed-4"}
    overall = next(row for row in stats if row["group_type"] == "overall"
                   and row["scope"] == "motion" and row["aggregation"] == "window_macro"
                   and row["method"] == "raw_eraft" and row["distance_metric"] == "cosine_distance")
    gt_overall = next(row for row in stats if row["group_type"] == "overall"
                      and row["scope"] == "motion" and row["aggregation"] == "window_macro"
                      and row["method"] == "gt_oracle" and row["distance_metric"] == "cosine_distance")
    assert overall["total_window_count"] == 6
    assert overall["eligible_window_count"] == 4
    assert np.isclose(overall["stable_reference_ratio"],
                      overall["aggregate_recovery"] / gt_overall["aggregate_recovery"])


def test_new_expanded_gate_and_raw_background_warning() -> None:
    rows = synthetic_stat_rows()
    stats = stable_aggregate_report(rows)
    decision = expanded_decision(rows, ["fixed-3", "fixed-4"], stable_stats=stats)
    assert decision["status"] == "expanded"
    criteria = decision["criteria"]
    assert criteria["observation_mean_oracle_gap_closed_used"] is False
    assert criteria["masked_all_valid_positive"] is True
    assert "hurts all-valid" in criteria["background_drift_warning"]
    assert all(item["consistent"] for item in
               criteria["stable_reference_ratio_consistency"].values())


def test_statistical_results_serialize(tmp_path) -> None:
    rows = synthetic_stat_rows()
    decision = expanded_decision(rows, ["fixed-3", "fixed-4"])
    row = {
        "recording": "fixed-3", "value": np.float32(0.3),
    }
    json_path = tmp_path / "result.json"
    atomic_write_json(json_path, {"decision": decision, "numpy": np.float32(1.5)})
    assert json.loads(json_path.read_text())["numpy"] == 1.5
    csv_path = tmp_path / "result.csv"
    write_csv(csv_path, [row])
    with csv_path.open(newline="", encoding="utf-8") as handle:
        loaded = next(csv.DictReader(handle))
    assert loaded["recording"] == "fixed-3"


def test_existing_per_offset_csv_can_be_reaggregated_without_gpu(tmp_path) -> None:
    source = tmp_path / "per_offset_metrics.csv"
    write_csv(source, synthetic_stat_rows())
    loaded = read_metrics_csv(source)
    assert len(loaded) == 54
    assert loaded[0]["anchor_frame_id"] == 1000
    assert stable_aggregate_report(loaded)
    output = tmp_path / "audit"
    assert audit_statistics_main(["--per-offset-metrics", str(source),
                                  "--output-dir", str(output),
                                  "--bootstrap-replicates", "10"]) == 0
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "promising exploratory smoke" in summary
    assert "not yet a VLM downstream accuracy result" in summary
    assert (output / "oracle_gap_closed.png").is_file()
