import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.audit_sevd import LIMIT, audit_recording, classify, main, manifest_candidates


EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("t", "<i8"), ("pol", "?")])


def write_kitti(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("truck 0.0 0 0.1 1 1 6 5 1 2 3 4 5 6 0.2 99\n", encoding="utf-8")


def add_frame(recording: Path, identifier: int, timestamps: list[int] | None = None) -> None:
    rgb = Image.new("RGB", (8, 6), (10, 20, 30))
    rgb.save(recording / "rgb_camera" / f"{identifier}.png")
    rgb.save(recording / "dvs_camera" / f"{identifier}.png")
    rgb.save(recording / "optical_flow_camera" / f"{identifier}.png")
    rgb.save(recording / "depth_camera" / f"{identifier:05d}.png")
    rgb.save(recording / "semantic_segmentation_camera" / f"{identifier:05d}.png")
    rgb.save(recording / "instance_segmentation_camera" / f"{identifier:05d}.png")
    times = timestamps or [identifier * 100_000, identifier * 100_000 + 10]
    events = np.zeros(len(times), dtype=EVENT_DTYPE)
    events["x"] = np.arange(len(times)) + 1
    events["y"] = np.arange(len(times)) + 2
    events["t"] = times
    events["pol"] = np.arange(len(times)) % 2 == 0
    np.savez(recording / "dvs_camera" / f"{identifier}-xytp.npz", dvs_events=events)
    np.savez(recording / "optical_flow_camera" / f"{identifier}.npz",
             flow=np.ones((6, 8, 2), dtype=np.float32))
    write_kitti(recording / "annotations" / "rgb_camera" / "KITTI" / f"{identifier}.txt")
    write_kitti(recording / "annotations" / "dvs_camera" / "KITTI" / f"{identifier}.txt")
    voc = recording / "annotations" / "rgb_camera" / "VOC" / f"{identifier}.xml"
    voc.parent.mkdir(parents=True, exist_ok=True)
    voc.write_text("<annotation/>", encoding="utf-8")
    coco = recording / "annotations" / "dvs_camera" / "COCO" / f"{identifier}.json"
    coco.parent.mkdir(parents=True, exist_ok=True)
    coco.write_text("{}", encoding="utf-8")


def make_dataset(tmp_path: Path, frame_count: int = 3) -> tuple[Path, Path, Path]:
    root = tmp_path / "SEVD"
    recording = root / "extracted" / "fixed-3-archive"
    for directory in ("rgb_camera", "dvs_camera", "optical_flow_camera", "depth_camera",
                      "semantic_segmentation_camera", "instance_segmentation_camera"):
        (recording / directory).mkdir(parents=True, exist_ok=True)
    for identifier in range(4591, 4591 + frame_count):
        add_frame(recording, identifier)
    metadata = tmp_path / "explicit-metadata.json"
    metadata.write_text(json.dumps({"duration": 1800, "delta_seconds": 0.1,
                                    "egos": 0, "fixed-views": 4}), encoding="utf-8")
    return root, recording, metadata


def test_recording_relative_classification_prioritizes_annotations(tmp_path):
    root, recording, _metadata = make_dataset(tmp_path, 1)
    annotation = recording / "annotations" / "rgb_camera" / "KITTI" / "4591.txt"
    assert classify(annotation, recording) == "annotation"
    report, _state = audit_recording(root, "fixed-3")
    assert report["file_counts"]["rgb_frame"] == 1
    assert report["file_counts"]["annotation"] == 4


def test_structured_event_and_previews_are_not_duplicates(tmp_path):
    root, _recording, metadata = make_dataset(tmp_path, 3)
    report, _state = audit_recording(root, "fixed-3", metadata)
    assert report["file_counts"]["event_npz"] == 3
    assert report["file_counts"]["event_preview"] == 3
    assert report["file_counts"]["flow_preview"] == 3
    assert report["frame_id_alignment"]["modalities"]["event_npz"]["duplicate_count"] == 0
    sample = report["event_audit"]["samples"][0]
    assert sample["schema_kind"] == "structured_dvs_events"
    assert sample["event_count"] == 2
    assert sample["timestamp_monotonic_nondecreasing"] is True
    assert sample["positive_polarity_count"] == 1
    assert sample["negative_polarity_count"] == 1


def test_annotation_encodings_are_separate_from_canonical_duplicates(tmp_path):
    root, _recording, metadata = make_dataset(tmp_path, 2)
    report, _state = audit_recording(root, "fixed-3", metadata)
    counts = report["annotations"]["counts"]
    assert counts["rgb_KITTI"] == 2
    assert counts["dvs_KITTI"] == 2
    assert counts["rgb_VOC"] == 2
    assert counts["COCO"] == 2
    assert report["annotations"]["alignment_bbox_source"] == "rgb_camera/KITTI"
    assert all(item["duplicate_count"] == 0
               for item in report["frame_id_alignment"]["modalities"].values())


def test_dvs_kitti_fallback_is_reported(tmp_path):
    root, recording, metadata = make_dataset(tmp_path, 3)
    for path in (recording / "annotations" / "rgb_camera" / "KITTI").glob("*.txt"):
        path.unlink()
    report, state = audit_recording(root, "fixed-3", metadata)
    assert report["annotations"]["alignment_bbox_source"] == "dvs_camera/KITTI"
    assert report["annotations"]["alignment_bbox_fallback_used"] is True
    assert "dvs_kitti_bbox_fallback" in {item["code"] for item in report["warnings"]}
    assert manifest_candidates(state, root)[0]["rgb_annotation_path"] is None


def test_structured_event_non_monotonic_warning(tmp_path):
    root, recording, metadata = make_dataset(tmp_path, 1)
    add_frame(recording, 4591, [10, 9, 11])
    report, _state = audit_recording(root, "fixed-3", metadata)
    assert report["event_audit"]["non_monotonic_count"] == 1
    assert "non_monotonic_event_timestamp" in {item["code"] for item in report["warnings"]}


def test_explicit_metadata_path_and_timestamp_inference(tmp_path):
    root, _recording, metadata = make_dataset(tmp_path, 3)
    report, _state = audit_recording(root, "fixed-3", metadata)
    assert report["metadata"]["values"] == {"delta_seconds": 0.1, "duration": 1800,
                                               "egos": 0, "fixed-views": 4}
    assert report["event_audit"]["timestamp_unit"]["status"] == "inference"


def test_metadata_falls_back_to_recording_parent(tmp_path):
    root, recording, metadata = make_dataset(tmp_path, 2)
    parent_metadata = recording.parent / "metadata.json"
    parent_metadata.write_text(metadata.read_text(encoding="utf-8"), encoding="utf-8")
    report, _state = audit_recording(root, "fixed-3")
    assert report["metadata"]["path"] == "extracted/metadata.json"
    assert report["metadata"]["values"]["delta_seconds"] == 0.1


def test_manifest_and_alignment_preview_are_generated(tmp_path):
    root, _recording, metadata = make_dataset(tmp_path, 21)
    output = tmp_path / "artifacts" / "r1" / "audit.json"
    result = main(["--root", str(root), "--recording", "fixed-3",
                   "--metadata-path", str(metadata), "--output", str(output)])
    assert result == 0
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["recordings"][0]["mapping_confirmed"] is True
    assert audit["recordings"][0]["mapping_checks"] == {
        "canonical_directories_identified": True,
        "event_npz_schema_parseable": True,
        "sampled_flow_npz_schema_parseable": True,
        "canonical_frame_intersection_nonempty": True,
    }
    assert audit["recordings"][0]["alignment_preview_frame_ids"] == [4591, 4601, 4611]
    manifest = output.parent / "manifest_preview.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == LIMIT
    assert set(rows[0]) >= {"recording", "frame_id", "rgb_path", "event_npz_path",
                            "flow_npz_path", "rgb_annotation_path"}
    assert all(not Path(row["rgb_path"]).is_absolute() for row in rows)
    assert (output.parent / "alignment_preview_fixed-3.png").is_file()


def test_missing_and_duplicate_examples_are_capped(tmp_path):
    root, recording, metadata = make_dataset(tmp_path, 25)
    for identifier in range(4591, 4616):
        (recording / "rgb_camera" / f"{identifier}.png").unlink()
        np.savez(recording / "optical_flow_camera" / f"copy-{identifier}.npz",
                 flow=np.ones((6, 8, 2), dtype=np.float32))
    report, _state = audit_recording(root, "fixed-3", metadata)
    rgb = report["frame_id_alignment"]["modalities"]["rgb_frame"]
    flow = report["frame_id_alignment"]["modalities"]["flow_npz"]
    assert rgb["missing_count"] == 25
    assert len(rgb["missing_examples"]) == LIMIT
    assert flow["duplicate_count"] == 25
    assert len(flow["duplicate_examples"]) == LIMIT
