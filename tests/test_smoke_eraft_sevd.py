import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.smoke_eraft_sevd import (
    checkpoint_prefix_action,
    normalize_state_dict_prefix,
    prediction_statistics,
    read_sevd_events,
    require_prediction,
    resize_prediction_native,
    resolve_transition_paths,
    signed_metric_report,
    smoke_decision,
    validate_events,
    voxelize_events,
)
from src.event_delta.json_io import atomic_write_json


EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("t", "<i8"), ("pol", "?")])


def packet(x, y, t, pol):
    return {"x": np.asarray(x), "y": np.asarray(y), "t": np.asarray(t, dtype=np.int64),
            "pol": np.asarray(pol, dtype=bool)}


def test_voxelization_trilinear_sign_timestamp_and_nonzero_normalization():
    # Native (1,1) maps to (0.5,0.5); middle time maps to 14/3 and splats to bins 4/5.
    events = packet([0, 1, 0], [0, 1, 0], [10**18, 10**18 + 1, 10**18 + 3],
                    [True, False, True])
    voxel, stats = voxelize_events(events, bins=15, input_hw=(2, 2), source_hw=(4, 4))
    assert voxel.shape == (1, 15, 2, 2)
    assert voxel.dtype == np.float32
    assert stats["event_count"] == 3
    assert stats["timestamp_span"] == 3
    assert stats["positive_event_count"] == 2
    assert stats["negative_event_count"] == 1
    assert stats["coordinate_scale_xy"] == [0.5, 0.5]
    assert stats["polarity_mapping"] == {"False": -1.0, "True": 1.0}
    assert stats["voxel_finite"] is True
    raw = np.zeros((15, 2, 2), dtype=np.float32)
    raw[0, 0, 0] = 1
    raw[14, 0, 0] = 1
    raw[4, :, :] = -1 / 12  # 0.25 spatial weight * 1/3 temporal weight.
    raw[5, :, :] = -1 / 6   # 0.25 spatial weight * 2/3 temporal weight.
    mask = raw != 0
    expected = raw.copy()
    expected[mask] = (raw[mask] - raw[mask].mean(dtype=np.float32)) / raw[mask].std(
        ddof=1, dtype=np.float32)
    np.testing.assert_allclose(voxel[0], expected, atol=2e-6)
    nonzero = voxel[voxel != 0]
    assert float(nonzero.mean()) == pytest.approx(0, abs=2e-7)
    assert float(nonzero.std(ddof=1)) == pytest.approx(1, abs=2e-7)


@pytest.mark.parametrize("timestamps", [[], [7], [7, 7]])
def test_voxelization_rejects_empty_or_single_timestamp(timestamps):
    count = len(timestamps)
    with pytest.raises(ValueError, match="empty|two distinct"):
        voxelize_events(packet([0] * count, [0] * count, timestamps, [True] * count))


def test_resize_scales_vector_components_to_native_pixels():
    prediction = np.zeros((1, 2, 2, 4), dtype=np.float32)
    prediction[:, 0] = 3
    prediction[:, 1] = -2
    native = resize_prediction_native(prediction, native_hw=(4, 8))
    assert native.shape == (4, 8, 2)
    assert native.dtype == np.float32
    np.testing.assert_allclose(native[..., 0], 6)
    np.testing.assert_allclose(native[..., 1], -4)


def write_event(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    events = np.zeros(2, dtype=EVENT_DTYPE)
    events["t"] = [1, 2]
    np.savez(path, dvs_events=events)


def test_structured_event_schema_validation_and_bounds(tmp_path):
    path = tmp_path / "41-xytp.npz"
    write_event(path)
    events = read_sevd_events(path)
    stats = validate_events(events, path, source_hw=(2, 3))
    assert stats["fields"] == {"x": "uint16", "y": "uint16", "t": "int64", "pol": "bool"}
    assert stats["timestamp_span"] == 1
    bad_schema = tmp_path / "bad-xytp.npz"
    np.savez(bad_schema, x=np.array([0]), y=np.array([0]), t=np.array([1]), pol=np.array([True]))
    with pytest.raises(ValueError, match="dvs_events"):
        read_sevd_events(bad_schema)
    non_monotonic = packet([0, 0, 0], [0, 0, 0], [1, 3, 2], [True, False, True])
    with pytest.raises(ValueError, match="monotonic"):
        validate_events(non_monotonic, path, source_hw=(2, 3))
    out_of_bounds = packet([0, 3], [0, 1], [1, 2], [True, False])
    with pytest.raises(ValueError, match="bounds"):
        validate_events(out_of_bounds, path, source_hw=(2, 3))


def test_pair_resolution_uses_event_i_event_i_plus_one_and_flow_i(tmp_path):
    recording = tmp_path / "extracted" / "fixed-3-archive"
    write_event(recording / "dvs_camera" / "41-xytp.npz")
    write_event(recording / "dvs_camera" / "42-xytp.npz")
    np.savez(recording / "dvs_camera" / "41-unrelated.npz", ignored=np.zeros(1))
    (recording / "optical_flow_camera").mkdir(parents=True)
    np.savez(recording / "optical_flow_camera" / "41.npz", flow=np.zeros((2, 3, 2), np.float32))
    (recording / "rgb_camera").mkdir(parents=True)
    Image.new("RGB", (3, 2)).save(recording / "rgb_camera" / "41.png")
    Image.new("RGB", (3, 2)).save(recording / "rgb_camera" / "42.png")
    paths = resolve_transition_paths(tmp_path, "fixed-3", 41)
    assert paths["old_event"].name == "41-xytp.npz"
    assert paths["new_event"].name == "42-xytp.npz"
    assert paths["gt_flow"].name == "41.npz"
    assert paths["source_rgb"].name == "41.png"
    assert paths["target_rgb"].name == "42.png"


def test_pair_resolution_fails_when_new_packet_is_missing(tmp_path):
    recording = tmp_path / "fixed-3"
    write_event(recording / "dvs_camera" / "41-xytp.npz")
    (recording / "optical_flow_camera").mkdir(parents=True)
    (recording / "rgb_camera").mkdir(parents=True)
    np.savez(recording / "optical_flow_camera" / "41.npz", flow=np.zeros((2, 3, 2), np.float32))
    Image.new("RGB", (3, 2)).save(recording / "rgb_camera" / "41.png")
    Image.new("RGB", (3, 2)).save(recording / "rgb_camera" / "42.png")
    with pytest.raises(FileNotFoundError, match="new_event frame 42"):
        resolve_transition_paths(tmp_path, "fixed-3", 41)


def test_prefix_normalization_accepts_only_exact_or_uniform_module_prefix():
    exact = OrderedDict((key, object()) for key in ("a", "b"))
    assert set(normalize_state_dict_prefix(exact, {"a", "b"})) == {"a", "b"}
    prefixed = OrderedDict((f"module.{key}", object()) for key in ("a", "b"))
    assert set(normalize_state_dict_prefix(prefixed, {"a", "b"})) == {"a", "b"}
    assert checkpoint_prefix_action(exact, {"a", "b"}) == "none"
    assert checkpoint_prefix_action(prefixed, {"a", "b"}) == "stripped_uniform_module_prefix"
    with pytest.raises(ValueError, match="mixes"):
        normalize_state_dict_prefix({"module.a": 1, "b": 2}, {"a", "b"})
    with pytest.raises(ValueError, match="strictly match"):
        normalize_state_dict_prefix({"wrong": 1}, {"a", "b"})


def test_shape_and_finite_guard():
    good = np.zeros((1, 2, 480, 640), dtype=np.float32)
    assert require_prediction(good) is good
    with pytest.raises(ValueError, match="shape"):
        require_prediction(np.zeros((1, 2, 240, 320), dtype=np.float32))
    bad = good.copy()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        require_prediction(bad)


def test_reversed_prediction_is_diagnostic_and_never_selected():
    ground_truth = np.zeros((2, 3, 2), dtype=np.float32)
    ground_truth[..., 0] = 2.0
    motion = np.ones((2, 3), dtype=bool)
    correct = signed_metric_report(ground_truth, ground_truth, motion)
    swapped = signed_metric_report(-ground_truth, ground_truth, motion)
    statistics = prediction_statistics(ground_truth)
    passed = smoke_decision(correct, swapped, statistics)
    assert passed["status"] == "passed"
    assert passed["formal_prediction_changed"] is False
    reversed_formal = smoke_decision(swapped, correct, statistics)
    assert reversed_formal["status"] == "blocked"
    assert "positive_sign_not_clearly_better_than_negative" in reversed_formal["reasons"]


def test_smoke_json_accepts_numpy_scalars(tmp_path):
    destination = tmp_path / "metrics.json"
    atomic_write_json(destination, {"count": np.int64(7), "score": np.float32(1.5),
                                    "finite": np.bool_(True)})
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "count": 7, "score": 1.5, "finite": True,
    }
