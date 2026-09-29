import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.audit_sevd import audit_recording
from tools.calibrate_sevd import (
    aggregate_event_scores,
    best_and_status_event,
    candidate_displacement,
    eligible_frame_ids,
    event_count_map,
    event_sample_scores,
    flow_frame_statistics,
    main,
    match_kitti_objects,
    read_event_npz,
    robust_background,
    stratified_sample,
)


EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("t", "<i8"), ("pol", "?")])


def write_kitti(path: Path, box: tuple[int, int, int, int], object_id: str = "42") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    left, top, right, bottom = box
    path.write_text(
        f"car 0 0 0 {left} {top} {right} {bottom} 1 1 1 0 0 0 0 {object_id}\n",
        encoding="utf-8",
    )


def make_recording(root: Path, name: str = "fixed-3", frame_count: int = 7,
                   event_hflip: bool = False) -> Path:
    recording = root / "extracted" / f"{name}-archive"
    for directory in ("rgb_camera", "dvs_camera", "optical_flow_camera"):
        (recording / directory).mkdir(parents=True, exist_ok=True)
    height, width = 18, 24
    previous = None
    for index in range(frame_count):
        identifier = 100 + index
        left = 3 + index
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[5:13, left:left + 7] = (220, 180, 80)
        Image.fromarray(rgb).save(recording / "rgb_camera" / f"{identifier}.png")
        if previous is None:
            changed = np.zeros((height, width), dtype=bool)
        else:
            changed = np.any(rgb != previous, axis=-1)
        y, x = np.nonzero(changed)
        if event_hflip:
            x = width - 1 - x
        events = np.zeros(len(x), dtype=EVENT_DTYPE)
        events["x"], events["y"] = x, y
        events["t"] = identifier * 100_000_000 + np.arange(len(x))
        events["pol"] = np.arange(len(x)) % 2 == 0
        np.savez(recording / "dvs_camera" / f"{identifier}-xytp.npz", dvs_events=events)
        flow = np.full((height, width, 2), -1 / 1024, dtype=np.float32)
        # Stored motion plus the robust background; source box center region sees +1 px after image scaling.
        source_left = max(3, left - 1)
        flow[5:13, source_left:source_left + 7, 0] = np.float32(1 / width - 1 / 1024)
        np.savez(recording / "optical_flow_camera" / f"{identifier}.npz", flow=flow)
        write_kitti(recording / "annotations" / "rgb_camera" / "KITTI" / f"{identifier}.txt",
                    (left, 5, left + 7, 13))
        previous = rgb
    return recording


def synthetic_event_scores(transform: str = "identity"):
    height, width = 30, 40
    desired = np.zeros((height, width), dtype=np.float32)
    desired[7:11, 5:9] = 2
    desired[20:23, 28:35] = 1
    event = desired if transform == "identity" else desired[:, ::-1]
    previous = desired.copy()
    next_difference = np.zeros_like(desired)
    next_difference[12:18, 15:21] = 1
    return event_sample_scores(event, previous, next_difference)


def test_identity_previous_to_current_is_selected():
    rows = aggregate_event_scores(synthetic_event_scores(), "fixed-3")
    assert rows[0]["temporal_alignment"] == "previous_to_current"
    assert rows[0]["spatial_transform"] == "identity"


def test_horizontal_flip_is_identified():
    rows = aggregate_event_scores(synthetic_event_scores("horizontal_flip"), "fixed-3")
    assert rows[0]["spatial_transform"] == "horizontal_flip"
    assert rows[0]["temporal_alignment"] == "previous_to_current"


def test_structured_event_dtype_is_read(tmp_path):
    path = tmp_path / "event.npz"
    events = np.zeros(2, dtype=EVENT_DTYPE)
    events["x"] = [1, 2]
    events["y"] = [2, 3]
    events["t"] = [10, 11]
    events["pol"] = [True, False]
    np.savez(path, dvs_events=events)
    arrays = read_event_npz(path)
    assert arrays["x"].dtype == np.dtype("<u2")
    assert np.isclose(event_count_map(arrays, (5, 6)).sum(), 2 * np.log(2))


def test_kitti_object_ids_are_matched():
    source = {"same": (1, 2, 7, 8), "old": (0, 0, 4, 4)}
    target = {"same": (4, 1, 10, 7), "new": (0, 0, 4, 4)}
    pairs = match_kitti_objects(source, target)
    assert [pair["object_id"] for pair in pairs] == ["same"]
    np.testing.assert_array_equal(pairs[0]["displacement"], [3, -1])


def test_flow_background_vector_statistics():
    flow = np.full((10, 12, 2), -1 / 1024, dtype=np.float32)
    flow[0, 0] = (0.25, -0.5)
    np.testing.assert_allclose(robust_background(flow), [-1 / 1024, -1 / 1024])
    stats = flow_frame_statistics(flow, 7)
    assert stats["exact_minus_1_over_1024_ratio"] == 119 / 120
    assert stats["finite_vector_count"] == 120


def test_image_size_scale_matches_synthetic_bbox_displacement():
    height, width = 20, 30
    flow = np.zeros((height, width, 2), dtype=np.float32)
    flow[..., 0] = 3 / width
    truth = np.array([3.0, 0.0])
    errors = {}
    for scale in ("raw", "image_size", "half_image_size"):
        estimate = np.median(candidate_displacement(flow, scale, 1, "raw").reshape(-1, 2), axis=0)
        errors[scale] = np.linalg.norm(estimate - truth)
    assert min(errors, key=errors.get) == "image_size"


def test_missing_adjacent_rgb_is_skipped_and_reported(tmp_path):
    root = tmp_path / "SEVD"
    recording = make_recording(root)
    (recording / "rgb_camera" / "103.png").unlink()
    _audit, state = audit_recording(root, "fixed-3")
    eligible, skipped = eligible_frame_ids(state)
    assert 102 not in eligible and 104 not in eligible
    assert any(item["frame_id"] == 102 and "missing_adjacent_rgb" in item["reasons"] for item in skipped)


def test_cli_outputs_json_csv_png_and_summary(tmp_path):
    root = tmp_path / "SEVD"
    make_recording(root)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"delta_seconds": 0.1, "duration": 1, "egos": 0, "fixed-views": 1}),
                        encoding="utf-8")
    output = tmp_path / "artifacts" / "r1_2"
    result = main(["--root", str(root), "--metadata-path", str(metadata), "--recording", "fixed-3",
                   "--samples", "3", "--seed", "20260929", "--output-dir", str(output)])
    assert result == 0
    expected = {
        "calibration.json", "event_candidate_scores.csv", "flow_candidate_scores.csv", "summary.md",
        "event_alignment_fixed-3.png", "flow_calibration_fixed-3.png", "best_warp_fixed-3.png",
    }
    assert expected <= {path.name for path in output.iterdir()}
    report = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    assert report["sampling"]["seed"] == 20260929
    assert report["warp_equation"].startswith("forward bilinear splat")


def test_sampling_is_deterministic_and_stratified():
    items = list(range(100))
    first = stratified_sample(items, 12, 77)
    assert first == stratified_sample(items, 12, 77)
    assert first != stratified_sample(items, 12, 78)
    assert len(first) == len(set(first)) == 12
    assert first == sorted(first)


def test_ambiguous_event_case_is_not_confirmed():
    zeros = np.zeros((20, 30), dtype=np.float32)
    rows = aggregate_event_scores(event_sample_scores(zeros, zeros, zeros), "fixed-3")
    result = best_and_status_event({"fixed-3": rows})
    assert result["status"] == "ambiguous"
