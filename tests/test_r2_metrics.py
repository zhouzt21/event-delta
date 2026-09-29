from pathlib import Path

import numpy as np
import pytest

from src.event_delta.flow_warp import forward_splat_numpy
from src.event_delta.metrics import compare_stale_and_propagated, summarize_by_offset


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
