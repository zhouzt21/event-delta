#!/usr/bin/env python3
"""Read-only SEVD directory, schema, timestamp, and alignment audit."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


FRAME_RE = re.compile(r"(\d+)")
LIMIT = 20
KINDS = (
    "rgb_frame", "event_npz", "event_preview", "flow_npz", "flow_preview",
    "depth", "semantic", "instance", "annotation", "metadata", "unclassified",
)
CANONICAL = ("rgb_frame", "event_npz", "flow_npz")


def scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def display_path(path: Path, root: Path) -> str:
    try:
        return relative(path, root)
    except ValueError:
        return str(path.resolve())


def warning(report: dict[str, Any], code: str, message: str) -> None:
    item = {"code": code, "message": message}
    if item not in report["warnings"]:
        report["warnings"].append(item)


def frame_id(path: Path) -> int | None:
    matches = FRAME_RE.findall(path.stem)
    return int(matches[-1]) if matches else None


def sample_evenly(items: list[Any], limit: int = 3) -> list[Any]:
    if len(items) <= limit:
        return items
    if limit == 3:
        return [items[0], items[len(items) // 2], items[-1]]
    step = (len(items) - 1) / (limit - 1)
    return [items[round(index * step)] for index in range(limit)]


def find_recording(root: Path, recording: str) -> Path:
    if root.name in {recording, f"{recording}-archive"}:
        return root
    matches: list[Path] = []
    for current, directories, _files in os.walk(root):
        directories[:] = [name for name in directories if not name.startswith(".")]
        base = Path(current)
        matches.extend(base / name for name in directories if name in {recording, f"{recording}-archive"})
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(f"recording {recording!r} was not found below {root}")
    if len(matches) != 1:
        raise RuntimeError(f"recording {recording!r} is ambiguous: {', '.join(map(str, matches))}")
    return matches[0]


def classify(path: Path, recording_root: Path) -> str:
    """Classify only from the recording-relative top directory and suffix."""
    parts = path.relative_to(recording_root).parts
    if not parts:
        return "unclassified"
    top = parts[0].lower()
    suffix = path.suffix.lower()
    name = path.name.lower()
    if top == "annotations":
        return "annotation"
    if "metadata" in name and suffix == ".txt":
        return "metadata"
    if top == "rgb_camera" and suffix == ".png" and len(parts) == 2:
        return "rgb_frame"
    if top == "dvs_camera" and suffix == ".npz" and name.endswith("-xytp.npz") and len(parts) == 2:
        return "event_npz"
    if top == "dvs_camera" and suffix == ".png" and len(parts) == 2:
        return "event_preview"
    if top == "optical_flow_camera" and suffix == ".npz" and len(parts) == 2:
        return "flow_npz"
    if top == "optical_flow_camera" and suffix == ".png" and len(parts) == 2:
        return "flow_preview"
    if top == "depth_camera" and suffix == ".png" and len(parts) == 2:
        return "depth"
    if top == "semantic_segmentation_camera" and suffix == ".png" and len(parts) == 2:
        return "semantic"
    if top == "instance_segmentation_camera" and suffix == ".png" and len(parts) == 2:
        return "instance"
    return "unclassified"


def annotation_group(path: Path, recording_root: Path) -> tuple[str, str]:
    parts = path.relative_to(recording_root).parts
    sensor = parts[1].lower() if len(parts) > 1 else "unknown"
    encoding = parts[2].upper() if len(parts) > 2 else "UNKNOWN"
    if encoding not in {"KITTI", "VOC", "COCO"}:
        encoding = "UNKNOWN"
    sensor = {"rgb_camera": "rgb", "dvs_camera": "dvs"}.get(sensor, sensor)
    return sensor, encoding


def id_index(paths: list[Path], dataset_root: Path) -> dict[str, Any]:
    by_id: dict[int, list[Path]] = defaultdict(list)
    without_id: list[Path] = []
    for path in paths:
        identifier = frame_id(path)
        (by_id[identifier] if identifier is not None else without_id).append(path)
    duplicate_ids = sorted(identifier for identifier, values in by_id.items() if len(values) > 1)
    return {
        "ids": set(by_id), "by_id": by_id, "without_id": without_id, "duplicate_ids": duplicate_ids,
        "summary": {
            "frame_id_count": len(by_id),
            "without_frame_id_count": len(without_id),
            "without_frame_id_examples": [relative(path, dataset_root) for path in without_id[:LIMIT]],
            "duplicate_count": len(duplicate_ids),
            "duplicate_examples": [
                {"frame_id": identifier, "paths": [relative(path, dataset_root) for path in by_id[identifier]][:LIMIT]}
                for identifier in duplicate_ids[:LIMIT]
            ],
        },
    }


def compact_id_set(ids: set[int]) -> dict[str, Any]:
    ordered = sorted(ids)
    return {"count": len(ordered), "first": ordered[0] if ordered else None,
            "last": ordered[-1] if ordered else None, "examples": ordered[:LIMIT]}


def image_info(path: Path, dataset_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": relative(path, dataset_root), "frame_id": frame_id(path)}
    try:
        with Image.open(path) as image:
            result.update(width=image.width, height=image.height, mode=image.mode)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def array_summary(array: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.dtype.names:
        result["fields"] = [{"name": name, "dtype": str(array.dtype.fields[name][0])} for name in array.dtype.names]
    return result


def event_arrays(archive: Any) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    keys = list(archive.files)
    schema: dict[str, Any] = {"keys": keys, "arrays": {key: array_summary(archive[key]) for key in keys}}
    if "dvs_events" in archive.files:
        events = np.asarray(archive["dvs_events"])
        fields = set(events.dtype.names or ())
        if {"x", "y", "t", "pol"}.issubset(fields):
            schema.update(schema_kind="structured_dvs_events", event_key="dvs_events",
                          fields=[{"name": name, "dtype": str(events.dtype.fields[name][0])}
                                  for name in events.dtype.names or ()])
            return {name: events[name] for name in ("x", "y", "t", "pol")}, schema
    aliases = {"x": ("x", "xs"), "y": ("y", "ys"),
               "t": ("t", "ts", "time", "timestamp", "timestamps"),
               "pol": ("pol", "p", "polarity", "polarities")}
    lower = {key.lower(): key for key in keys}
    mapping: dict[str, str] = {}
    for canonical, choices in aliases.items():
        found = next((lower[name] for name in choices if name in lower), None)
        if found is None:
            schema["schema_kind"] = "unknown"
            return None, schema
        mapping[canonical] = found
    schema.update(schema_kind="separate_arrays", field_mapping=mapping)
    return {name: np.asarray(archive[key]) for name, key in mapping.items()}, schema


def inspect_event(path: Path, dataset_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    detail: dict[str, Any] = {"path": relative(path, dataset_root), "frame_id": frame_id(path)}
    endpoint: dict[str, Any] = {"path": detail["path"], "frame_id": detail["frame_id"],
                                "first": None, "last": None, "monotonic": None, "schema_parsed": False}
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays, schema = event_arrays(archive)
            detail.update(schema)
            if arrays is None:
                return detail, endpoint
            flattened = {name: np.asarray(value).reshape(-1) for name, value in arrays.items()}
            if len({len(value) for value in flattened.values()}) != 1:
                detail["schema_error"] = "x/y/t/polarity lengths differ"
                return detail, endpoint
            timestamps = flattened["t"]
            polarity = flattened["pol"].astype(bool)
            monotonic = bool(np.all(np.diff(timestamps) >= 0)) if len(timestamps) > 1 else True
            positive = int(np.count_nonzero(polarity))
            detail.update(
                event_count=len(timestamps),
                ranges={name: [scalar(values.min()), scalar(values.max())] if len(values) else None
                        for name, values in flattened.items() if name in {"x", "y", "t"}},
                timestamp_first=scalar(timestamps[0]) if len(timestamps) else None,
                timestamp_last=scalar(timestamps[-1]) if len(timestamps) else None,
                timestamp_monotonic_nondecreasing=monotonic,
                positive_polarity_count=positive,
                negative_polarity_count=int(len(polarity) - positive),
            )
            endpoint.update(first=detail["timestamp_first"], last=detail["timestamp_last"],
                            monotonic=monotonic, schema_parsed=True)
    except Exception as exc:
        detail["error"] = f"{type(exc).__name__}: {exc}"
    return detail, endpoint


def inspect_flow(path: Path, dataset_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": relative(path, dataset_root), "frame_id": frame_id(path)}
    try:
        with np.load(path, allow_pickle=False) as archive:
            result["keys"] = list(archive.files)
            result["arrays"] = {key: array_summary(archive[key]) for key in archive.files}
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def summarize_boundaries(endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    overlap: list[dict[str, Any]] = []
    gap: list[dict[str, Any]] = []
    unknown = touching = 0
    for previous, current in zip(endpoints, endpoints[1:]):
        if previous["last"] is None or current["first"] is None:
            unknown += 1
            continue
        delta = scalar(current["first"] - previous["last"])
        item = {"previous_path": previous["path"], "current_path": current["path"],
                "previous_last": previous["last"], "current_first": current["first"], "boundary_delta": delta}
        if delta < 0:
            overlap.append(item)
        elif delta > 0:
            gap.append(item)
        else:
            touching += 1
    return {"pairs_checked": max(0, len(endpoints) - 1), "unknown_count": unknown,
            "overlap_count": len(overlap), "overlap_examples": overlap[:LIMIT],
            "gap_count": len(gap), "gap_examples": gap[:LIMIT], "touching_count": touching,
            "note": "gap means current_first > previous_last in unknown units; it does not prove missing events"}


def infer_timestamp_unit(endpoints: list[dict[str, Any]], delta_seconds: Any) -> dict[str, Any] | None:
    if not isinstance(delta_seconds, (int, float)) or delta_seconds <= 0:
        return None
    deltas = [current["first"] - previous["first"] for previous, current in zip(endpoints, endpoints[1:])
              if previous["first"] is not None and current["first"] is not None
              and current["first"] > previous["first"]]
    if not deltas:
        return None
    observed = float(np.median(np.asarray(deltas, dtype=np.float64)))
    candidates = {"seconds": 1.0, "milliseconds": 1e-3, "microseconds": 1e-6, "nanoseconds": 1e-9}
    scores = {name: abs(math.log10(max(observed * scale, 1e-300) / float(delta_seconds)))
              for name, scale in candidates.items()}
    candidate = min(scores, key=scores.get)
    return {"status": "inference", "candidate_unit": candidate,
            "observed_median_first_timestamp_delta": observed,
            "metadata_delta_seconds": delta_seconds,
            "scaled_candidate_seconds": observed * candidates[candidate],
            "warning": "numeric-scale inference is not confirmation of the timestamp unit"}


def resolve_metadata_path(explicit: Path | None, recording_root: Path) -> Path | None:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        return candidate if candidate.is_file() else None
    candidates = [recording_root / "metadata.json", recording_root.parent / "metadata.json"]
    return next((path for path in candidates if path.is_file()), None)


def parse_metadata(path: Path | None, dataset_root: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    result: dict[str, Any] = {"path": display_path(path, dataset_root)}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        result["values"] = {key: value.get(key) for key in ("delta_seconds", "duration", "egos", "fixed-views")}
        result["available_keys"] = sorted(value)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_kitti(path: Path, dataset_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": relative(path, dataset_root), "frame_id": frame_id(path), "format": "KITTI",
                              "schema": {"class_id": "token 0 (KITTI class label)",
                                         "bbox_left_top_right_bottom": "tokens 4:8",
                                         "object_id": "last token"}}
    parsed = []
    invalid = 0
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            tokens = line.split()
            if not tokens:
                continue
            try:
                parsed.append({"class_id": tokens[0], "bbox": [float(value) for value in tokens[4:8]],
                               "object_id": tokens[-1]})
            except (IndexError, ValueError):
                invalid += 1
        result.update(object_count=len(parsed), invalid_line_count=invalid, examples=parsed[:3])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def annotation_summary(paths: list[Path], recording_root: Path, dataset_root: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in paths:
        groups[annotation_group(path, recording_root)].append(path)
    counts = {"rgb_KITTI": len(groups[("rgb", "KITTI")]), "dvs_KITTI": len(groups[("dvs", "KITTI")]),
              "rgb_VOC": len(groups[("rgb", "VOC")]), "dvs_VOC": len(groups[("dvs", "VOC")]),
              "COCO": sum(len(values) for (_sensor, encoding), values in groups.items() if encoding == "COCO"),
              "rgb_COCO": len(groups[("rgb", "COCO")]), "dvs_COCO": len(groups[("dvs", "COCO")]),
              "unknown": sum(len(values) for (sensor, encoding), values in groups.items()
                             if sensor not in {"rgb", "dvs"} or encoding == "UNKNOWN")}
    details: dict[str, Any] = {"counts": counts, "groups": {}}
    for (sensor, encoding), values in sorted(groups.items()):
        values = sorted(values)
        index = id_index(values, dataset_root)
        key = f"{sensor}_{encoding}"
        details["groups"][key] = {"file_count": len(values), "frame_ids": compact_id_set(index["ids"]),
                                    "duplicate_count": index["summary"]["duplicate_count"],
                                    "duplicate_examples": index["summary"]["duplicate_examples"],
                                    "samples": [relative(path, dataset_root) for path in sample_evenly(values)]}
        if encoding == "KITTI" and values:
            details["groups"][key]["kitti_samples"] = [inspect_kitti(path, dataset_root)
                                                          for path in sample_evenly(values)]
    selected: dict[str, Any] | None = None
    for sensor in ("rgb", "dvs"):
        candidates = sorted(groups[(sensor, "KITTI")])
        if candidates:
            selected = {"source": f"{sensor}_camera/KITTI", "sensor": sensor,
                        "index": id_index(candidates, dataset_root)}
            break
    details["alignment_bbox_source"] = selected["source"] if selected else None
    details["alignment_bbox_fallback_used"] = bool(selected and selected["sensor"] == "dvs")
    return details, selected


def audit_recording(dataset_root: Path, recording: str,
                    metadata_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    recording_root = find_recording(dataset_root, recording)
    report: dict[str, Any] = {"recording": recording, "recording_directory": relative(recording_root, dataset_root),
                              "mapping_confirmed": False, "warnings": []}
    files: dict[str, list[Path]] = {kind: [] for kind in KINDS}
    directories: list[str] = []
    for current, child_dirs, names in os.walk(recording_root):
        child_dirs[:] = sorted(name for name in child_dirs if not name.startswith("."))
        current_path = Path(current)
        directories.append(relative(current_path, dataset_root))
        for name in sorted(names):
            path = current_path / name
            files[classify(path, recording_root)].append(path)
    report["discovered_directories"] = directories
    report["file_counts"] = {kind: len(files[kind]) for kind in KINDS}
    report["major_directories"] = {
        kind: dict(sorted(Counter(relative(path.parent, dataset_root) for path in values).items()))
        for kind, values in files.items() if values}
    report["samples"] = {kind: [{"path": relative(path, dataset_root), "frame_id": frame_id(path)}
                                for path in sample_evenly(sorted(values))]
                         for kind, values in files.items() if values}

    indexes = {kind: id_index(sorted(files[kind]), dataset_root) for kind in CANONICAL}
    union = set().union(*(index["ids"] for index in indexes.values()))
    intersection = set.intersection(*(index["ids"] for index in indexes.values()))
    report["frame_id_alignment"] = {"canonical_modalities": list(CANONICAL),
                                     "intersection": compact_id_set(intersection), "union_count": len(union),
                                     "modalities": {}}
    for kind, index in indexes.items():
        modality = dict(index["summary"])
        missing = sorted(union - index["ids"])
        modality.update(missing_count=len(missing), missing_examples=missing[:LIMIT])
        report["frame_id_alignment"]["modalities"][kind] = modality
        if modality["duplicate_count"]:
            warning(report, "duplicate_canonical_frame_id", f"{kind} has duplicate canonical frame IDs")

    image_groups = ("rgb_frame", "event_preview", "flow_preview", "semantic", "instance", "depth")
    report["image_resolutions"] = {kind: [image_info(path, dataset_root)
                                               for path in sample_evenly(sorted(files[kind]))]
                                     for kind in image_groups if files[kind]}
    resolved_metadata = resolve_metadata_path(metadata_path, recording_root)
    report["metadata"] = parse_metadata(resolved_metadata, dataset_root)
    if metadata_path is not None and resolved_metadata is None:
        warning(report, "metadata_path_missing", f"the supplied metadata path does not exist: {metadata_path}")
    elif report["metadata"] is None:
        warning(report, "metadata_not_found", "metadata.json was not found at the recording root or its parent")
    elif "error" in report["metadata"]:
        warning(report, "metadata_parse_error", report["metadata"]["error"])

    event_paths = sorted(files["event_npz"], key=lambda path: (frame_id(path) is None, frame_id(path), str(path)))
    event_samples = set(sample_evenly(event_paths))
    event_details: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    unknown_schema_paths: list[str] = []
    non_monotonic_paths: list[str] = []
    for path in event_paths:
        detail, endpoint = inspect_event(path, dataset_root)
        endpoints.append(endpoint)
        if path in event_samples:
            event_details.append(detail)
        if not endpoint["schema_parsed"]:
            unknown_schema_paths.append(relative(path, dataset_root))
        if endpoint["monotonic"] is False:
            non_monotonic_paths.append(relative(path, dataset_root))
    report["event_audit"] = {"files_checked": len(event_paths),
                              "schema_parsed_count": sum(endpoint["schema_parsed"] for endpoint in endpoints),
                              "non_monotonic_count": sum(endpoint["monotonic"] is False for endpoint in endpoints),
                              "samples": event_details, "adjacent_ranges": summarize_boundaries(endpoints)}
    if unknown_schema_paths:
        warning(report, "unknown_event_npz_schema",
                f"{len(unknown_schema_paths)} Event NPZ files could not be parsed; examples: {unknown_schema_paths[:LIMIT]}")
    if non_monotonic_paths:
        warning(report, "non_monotonic_event_timestamp",
                f"timestamps decrease in {len(non_monotonic_paths)} Event files; examples: {non_monotonic_paths[:LIMIT]}")
    metadata_values = (report["metadata"] or {}).get("values", {})
    report["event_audit"]["timestamp_unit"] = infer_timestamp_unit(
        endpoints, metadata_values.get("delta_seconds")) or {
            "status": "unknown",
            "warning": "timestamp unit cannot be inferred without usable timestamps and metadata.delta_seconds"}
    flow_paths = sorted(files["flow_npz"], key=lambda path: (frame_id(path) is None, frame_id(path), str(path)))
    report["flow_npz_samples"] = [inspect_flow(path, dataset_root) for path in sample_evenly(flow_paths)]
    flow_schema_clear = bool(report["flow_npz_samples"]) and all(
        "error" not in sample
        and "flow" in sample.get("keys", [])
        and sample.get("arrays", {}).get("flow", {}).get("shape", [])[-1:] == [2]
        for sample in report["flow_npz_samples"]
    )
    annotations, selected_annotation = annotation_summary(files["annotation"], recording_root, dataset_root)
    report["annotations"] = annotations

    expected_dirs = {"rgb_frame": "rgb_camera", "event_npz": "dvs_camera", "flow_npz": "optical_flow_camera"}
    directories_clear = all(files[kind] and
                            {path.relative_to(recording_root).parts[0] for path in files[kind]} == {expected}
                            for kind, expected in expected_dirs.items())
    schema_clear = bool(event_paths) and all(endpoint["schema_parsed"] for endpoint in endpoints)
    report["mapping_checks"] = {"canonical_directories_identified": directories_clear,
                                 "event_npz_schema_parseable": schema_clear,
                                 "sampled_flow_npz_schema_parseable": flow_schema_clear,
                                 "canonical_frame_intersection_nonempty": bool(intersection)}
    report["mapping_confirmed"] = all(report["mapping_checks"].values())
    warning(report, "timestamp_unit_unconfirmed", "timestamp unit remains an inference even when a candidate is reported")
    warning(report, "event_interval_causality_unconfirmed", "matching frame IDs do not prove the Event interval ends at the query frame")
    warning(report, "event_coordinate_direction_unconfirmed", "visual inspection must confirm x/y direction and polarity rendering")
    warning(report, "flow_semantics_unconfirmed", "visual inspection must confirm flow direction and source/target frame convention")
    if files["unclassified"]:
        warning(report, "unclassified_files", f"{len(files['unclassified'])} files did not match the confirmed SEVD layout")
    if selected_annotation is None:
        warning(report, "kitti_annotation_missing", "neither RGB nor DVS KITTI annotations were found for bbox overlay")
    elif selected_annotation["sensor"] == "dvs":
        warning(report, "dvs_kitti_bbox_fallback", "RGB KITTI was unavailable; alignment preview uses DVS KITTI boxes")
    state = {"recording": recording, "recording_root": recording_root, "files": files, "indexes": indexes,
             "intersection": sorted(intersection), "annotation": selected_annotation}
    return report, state


def manifest_candidates(state: dict[str, Any], dataset_root: Path) -> list[dict[str, Any]]:
    annotation = state["annotation"]
    annotation_by_id = annotation["index"]["by_id"] if annotation else {}
    rows = []
    for identifier in state["intersection"]:
        row = {"recording": state["recording"], "frame_id": identifier}
        for kind, field in {"rgb_frame": "rgb_path", "event_npz": "event_npz_path",
                            "flow_npz": "flow_npz_path"}.items():
            candidates = state["indexes"][kind]["by_id"][identifier]
            if len(candidates) != 1:
                break
            row[field] = relative(candidates[0], dataset_root)
        else:
            matches = annotation_by_id.get(identifier, []) if annotation and annotation["sensor"] == "rgb" else []
            row["rgb_annotation_path"] = relative(matches[0], dataset_root) if len(matches) == 1 else None
            rows.append(row)
    return rows


def balanced_manifest(candidates: list[list[dict[str, Any]]], limit: int = LIMIT) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    position = 0
    while len(output) < limit:
        added = False
        for rows in candidates:
            if position < len(rows) and len(output) < limit:
                output.append(rows[position])
                added = True
        if not added:
            break
        position += 1
    return output


def load_event_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with np.load(path, allow_pickle=False) as archive:
        arrays, _schema = event_arrays(archive)
        if arrays is None:
            raise ValueError("unrecognized Event NPZ schema")
        x = np.asarray(arrays["x"]).reshape(-1).astype(np.int64)
        y = np.asarray(arrays["y"]).reshape(-1).astype(np.int64)
        polarity = np.asarray(arrays["pol"]).reshape(-1).astype(bool)
    width, height = size
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    positive = np.zeros((height, width), dtype=np.float32)
    negative = np.zeros((height, width), dtype=np.float32)
    np.add.at(positive, (y[valid & polarity], x[valid & polarity]), 1)
    np.add.at(negative, (y[valid & ~polarity], x[valid & ~polarity]), 1)
    activity = np.maximum(positive, negative)
    nonzero = activity[activity > 0]
    robust = max(float(np.percentile(nonzero, 99)) if nonzero.size else 0.0, 1.0)
    canvas = np.full((height, width, 3), 245, dtype=np.float32)
    canvas[..., 0] -= 220 * negative / robust
    canvas[..., 1] -= 180 * np.maximum(positive, negative) / robust
    canvas[..., 2] -= 220 * positive / robust
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def load_flow_magnitude(path: Path) -> Image.Image:
    with np.load(path, allow_pickle=False) as archive:
        flow = np.asarray(archive["flow"]) if "flow" in archive.files else next(
            (np.asarray(archive[key]) for key in archive.files if np.asarray(archive[key]).ndim == 3), None)
    if flow is None:
        raise ValueError("no flow array found")
    if flow.ndim == 3 and flow.shape[0] == 2 and flow.shape[-1] != 2:
        flow = np.moveaxis(flow, 0, -1)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"expected HxWx2 flow, got {flow.shape}")
    magnitude = np.linalg.norm(flow.astype(np.float32), axis=-1)
    finite = magnitude[np.isfinite(magnitude)]
    robust = float(np.percentile(finite, 99)) if finite.size else 0.0
    normalized = np.clip(magnitude / robust, 0, 1) if robust > 0 else np.zeros_like(magnitude)
    heat = np.stack((255 * normalized, 255 * np.sqrt(normalized), 255 * (1 - normalized)), axis=-1)
    return Image.fromarray(heat.astype(np.uint8))


def kitti_boxes(path: Path | None) -> list[tuple[float, float, float, float, str]]:
    if path is None:
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split()
        try:
            left, top, right, bottom = (float(value) for value in tokens[4:8])
            boxes.append((left, top, right, bottom, tokens[0]))
        except (IndexError, ValueError):
            continue
    return boxes


def overlay_boxes(rgb: Image.Image, annotation: Path | None) -> Image.Image:
    output = rgb.copy()
    draw = ImageDraw.Draw(output)
    for left, top, right, bottom, label in kitti_boxes(annotation):
        draw.rectangle((left, top, right, bottom), outline=(255, 32, 32), width=3)
        draw.text((left + 2, max(0, top - 12)), label, fill=(255, 32, 32))
    return output


def panel(image: Image.Image, label: str, cell: tuple[int, int]) -> Image.Image:
    width, height = cell
    output = Image.new("RGB", (width, height + 24), "white")
    image = image.convert("RGB")
    image.thumbnail((width, height))
    output.paste(image, ((width - image.width) // 2, 24 + (height - image.height) // 2))
    ImageDraw.Draw(output).text((5, 5), label, fill="black")
    return output


def create_alignment_preview(state: dict[str, Any], dataset_root: Path, destination: Path) -> list[int]:
    identifiers = sample_evenly(state["intersection"], 3)
    if len(identifiers) < 3:
        raise ValueError("fewer than three common frame IDs")
    annotation = state["annotation"]
    annotation_by_id = annotation["index"]["by_id"] if annotation else {}
    cell = (480, 270)
    canvas = Image.new("RGB", (cell[0] * 3, (cell[1] + 24) * 3), "white")
    for column, identifier in enumerate(identifiers):
        rgb_path = state["indexes"]["rgb_frame"]["by_id"][identifier][0]
        event_path = state["indexes"]["event_npz"]["by_id"][identifier][0]
        flow_path = state["indexes"]["flow_npz"]["by_id"][identifier][0]
        matches = annotation_by_id.get(identifier, [])
        annotation_path = matches[0] if len(matches) == 1 else None
        rgb = Image.open(rgb_path).convert("RGB")
        images = [overlay_boxes(rgb, annotation_path), load_event_image(event_path, rgb.size),
                  load_flow_magnitude(flow_path)]
        bbox_source = annotation["source"] if annotation else "no KITTI"
        labels = (f"RGB + {bbox_source} bbox", "Raw Event accumulation", "Flow magnitude (P99)")
        for row, (label, image) in enumerate(zip(labels, images)):
            canvas.paste(panel(image, f"{label}; frame_id={identifier}", cell),
                         (column * cell[0], row * (cell[1] + 24)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return identifiers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="SEVD dataset root")
    parser.add_argument("--recording", required=True, action="append", help="recording name; repeat this option")
    parser.add_argument("--output", required=True, type=Path, help="audit JSON path")
    parser.add_argument("--metadata-path", type=Path, help="optional metadata.json path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.root.expanduser().resolve()
    if not dataset_root.is_dir():
        print(f"error: dataset root is not a directory: {dataset_root}", file=sys.stderr)
        return 2
    output = args.output.expanduser().resolve()
    combined: dict[str, Any] = {"dataset_root": str(dataset_root), "scope": "static read-only SEVD R1.1 audit",
                                "recordings_requested": args.recording, "recordings": [], "generated": {}}
    states: list[dict[str, Any]] = []
    try:
        for recording in args.recording:
            report, state = audit_recording(dataset_root, recording, args.metadata_path)
            combined["recordings"].append(report)
            states.append(state)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    eligible_candidates = []
    for report, state in zip(combined["recordings"], states):
        if not report["mapping_confirmed"]:
            continue
        eligible_candidates.append(manifest_candidates(state, dataset_root))
        preview = output.parent / f"alignment_preview_{state['recording']}.png"
        try:
            report["alignment_preview_frame_ids"] = create_alignment_preview(state, dataset_root, preview)
            combined["generated"][f"alignment_preview_{state['recording']}"] = preview.name
        except Exception as exc:
            warning(report, "alignment_preview_failed", f"{type(exc).__name__}: {exc}")
    rows = balanced_manifest(eligible_candidates)
    if rows:
        manifest = output.parent / "manifest_preview.jsonl"
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        combined["generated"]["manifest_preview"] = manifest.name
        combined["generated"]["manifest_preview_rows"] = len(rows)
    output.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
