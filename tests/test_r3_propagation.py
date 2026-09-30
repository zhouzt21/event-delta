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
    oracle_gap_closed,
    predicted_flow_to_feature_displacement,
    transition_frame_ids,
    validate_latency_fields,
)
from tools.run_r2_oracle import write_csv


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


def test_expanded_gate_and_serialization(tmp_path) -> None:
    rows = []
    for recording in ("fixed-3", "fixed-4"):
        for offset in range(1, 10):
            row = {
                "recording": recording, "offset": offset,
                "motion_raw_eraft_cosine_distance_recovery": np.float32(0.3),
                "motion_raw_eraft_cosine_distance_oracle_gap_closed": np.float64(0.25),
                "motion_masked_eraft_cosine_distance_oracle_gap_closed": np.float64(0.1),
                "raw_eraft_coverage_ratio": np.float32(0.9),
                "gt_oracle_coverage_ratio": np.float32(0.92),
                "complete_online_delta_path_ms": np.float64(12.5),
            }
            rows.append(row)
    decision = expanded_decision(rows, ["fixed-3", "fixed-4"])
    assert decision["status"] == "expanded"
    json_path = tmp_path / "result.json"
    atomic_write_json(json_path, {"decision": decision, "numpy": np.float32(1.5)})
    assert json.loads(json_path.read_text())["numpy"] == 1.5
    csv_path = tmp_path / "result.csv"
    write_csv(csv_path, rows[:1])
    with csv_path.open(newline="", encoding="utf-8") as handle:
        loaded = next(csv.DictReader(handle))
    assert loaded["recording"] == "fixed-3"
