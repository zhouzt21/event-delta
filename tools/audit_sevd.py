#!/usr/bin/env python3
"""Audit SEVD recordings without modifying the dataset.

The auditor discovers modalities from the directory and file names it actually
finds.  It deliberately reports ambiguity instead of assuming the layout or
timestamp units documented by any external source.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ANNOTATION_SUFFIXES = {".json", ".jsonl", ".txt", ".csv", ".xml"}
MODALITIES = ("rgb", "event", "flow", "semantic", "instance", "annotation")
FRAME_RE = re.compile(r"(\d+)")


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def warning(report: dict[str, Any], code: str, message: str) -> None:
    item = {"code": code, "message": message}
    if item not in report["warnings"]:
        report["warnings"].append(item)


def classify(path: Path) -> str | None:
    """Classify by names present on disk; ambiguous files remain unclassified."""
    parts = [part.lower().replace("-", "_") for part in path.parts]
    joined = "/".join(parts)
    name = path.name.lower()
    if name == "metadata.json":
        return "metadata"
    if any("semantic" in part for part in parts):
        return "semantic"
    if any("instance" in part for part in parts):
        return "instance"
    if any("optical_flow" in part or part == "flow" or part.endswith("_flow") for part in parts):
        return "flow"
    if any(part == "dvs" or "event" in part for part in parts):
        return "event"
    if any(part == "rgb" or "rgb" in part for part in parts):
        return "rgb"
    if any(token in joined for token in ("annotation", "bounding_box", "bbox", "/boxes", "/labels")):
        return "annotation"
    if path.suffix.lower() in ANNOTATION_SUFFIXES and name != "metadata.json":
        return "annotation"
    return None


def frame_id(path: Path) -> int | None:
    numbers = FRAME_RE.findall(path.stem)
    return int(numbers[-1]) if numbers else None


def sample_evenly(items: list[Any], limit: int = 3) -> list[Any]:
    if len(items) <= limit:
        return items
    indexes = sorted({0, len(items) // 2, len(items) - 1})
    return [items[index] for index in indexes]


def find_recording(root: Path, recording: str) -> Path:
    if root.name in {recording, f"{recording}-archive"}:
        return root
    matches: list[Path] = []
    for current, dirs, _files in os.walk(root):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        current_path = Path(current)
        for name in dirs:
            if name in {recording, f"{recording}-archive"}:
                matches.append(current_path / name)
    unique = sorted(set(matches))
    if not unique:
        raise FileNotFoundError(f"recording {recording!r} was not found below {root}")
    if len(unique) > 1:
        rendered = ", ".join(str(path) for path in unique)
        raise RuntimeError(f"recording {recording!r} is ambiguous: {rendered}")
    return unique[0]


def array_summary(array: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.size and np.issubdtype(array.dtype, np.number):
        finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.inexact) else array
        if finite.size:
            result["min"] = scalar(finite.min())
            result["max"] = scalar(finite.max())
    return result


def scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def identify_event_keys(keys: Iterable[str]) -> dict[str, str] | None:
    aliases = {
        "x": ("x", "xs"),
        "y": ("y", "ys"),
        "p": ("p", "ps", "polarity", "polarities"),
        "t": ("t", "ts", "time", "times", "timestamp", "timestamps"),
    }
    lower_to_real = {key.lower(): key for key in keys}
    result: dict[str, str] = {}
    for canonical, choices in aliases.items():
        found = next((lower_to_real[item] for item in choices if item in lower_to_real), None)
        if found is None:
            return None
        result[canonical] = found
    return result


def inspect_event_npz(path: Path, dataset_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    item: dict[str, Any] = {"path": relative(path, dataset_root), "frame_id": frame_id(path)}
    endpoint: dict[str, Any] = {
        "path": item["path"],
        "frame_id": item["frame_id"],
        "first": None,
        "last": None,
        "monotonic": None,
        "recognized": False,
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)
            item["keys"] = keys
            item["arrays"] = {key: array_summary(archive[key]) for key in keys}
            mapping = identify_event_keys(keys)
            item["recognized_key_mapping"] = mapping
            if mapping is not None:
                arrays = {name: np.asarray(archive[key]).reshape(-1) for name, key in mapping.items()}
                item["ranges"] = {
                    name: ([scalar(values.min()), scalar(values.max())] if values.size else None)
                    for name, values in arrays.items()
                }
                timestamps = arrays["t"]
                monotonic = bool(np.all(np.diff(timestamps) >= 0)) if timestamps.size > 1 else True
                item["timestamp_monotonic_nondecreasing"] = monotonic
                item["timestamp_first"] = scalar(timestamps[0]) if timestamps.size else None
                item["timestamp_last"] = scalar(timestamps[-1]) if timestamps.size else None
                endpoint = {
                    "path": item["path"],
                    "frame_id": item["frame_id"],
                    "first": item["timestamp_first"],
                    "last": item["timestamp_last"],
                    "monotonic": monotonic,
                    "recognized": True,
                }
    except Exception as exc:  # Keep auditing other files.
        item["error"] = f"{type(exc).__name__}: {exc}"
    return item, endpoint


def inspect_flow_npz(path: Path, dataset_root: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": relative(path, dataset_root), "frame_id": frame_id(path)}
    try:
        with np.load(path, allow_pickle=False) as archive:
            item["keys"] = list(archive.files)
            item["arrays"] = {key: array_summary(archive[key]) for key in archive.files}
    except Exception as exc:
        item["error"] = f"{type(exc).__name__}: {exc}"
    return item


def image_info(path: Path, dataset_root: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": relative(path, dataset_root), "frame_id": frame_id(path)}
    try:
        with Image.open(path) as image:
            item.update({"width": image.width, "height": image.height, "mode": image.mode})
    except Exception as exc:
        item["error"] = f"{type(exc).__name__}: {exc}"
    return item


def summarize_json(value: Any, depth: int = 0) -> Any:
    if depth >= 3:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        return {str(key): summarize_json(item, depth + 1) for key, item in list(value.items())[:30]}
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "sample": [summarize_json(item, depth + 1) for item in value[:3]]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def collect_keys(value: Any, output: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            output.add(str(key))
            collect_keys(item, output)
    elif isinstance(value, list):
        for item in value[:100]:
            collect_keys(item, output)


def inspect_annotation(path: Path, dataset_root: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": relative(path, dataset_root),
        "frame_id": frame_id(path),
        "format": path.suffix.lower().lstrip("."),
    }
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            keys: set[str] = set()
            collect_keys(value, keys)
            normalized = {key.lower().replace("-", "_") for key in keys}
            item["root_type"] = type(value).__name__
            item["fields"] = sorted(keys)
            item["class_id_fields"] = sorted(keys_for(normalized, keys, {"class_id", "classid", "category_id", "category"}))
            item["object_id_fields"] = sorted(keys_for(normalized, keys, {"object_id", "objectid", "instance_id", "track_id", "id"}))
            bbox_names = {"bbox", "bounding_box", "box", "boxes", "xmin", "x_min", "xmax", "x_max", "ymin", "y_min", "ymax", "y_max", "width", "height"}
            item["bbox_fields"] = sorted(keys_for(normalized, keys, bbox_names))
        else:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            item["line_count"] = len(lines)
            item["sample_lines"] = lines[:3]
            item["field_semantics"] = "unknown"
    except Exception as exc:
        item["error"] = f"{type(exc).__name__}: {exc}"
    return item


def keys_for(normalized: set[str], original: set[str], targets: set[str]) -> set[str]:
    if not normalized.intersection(targets):
        return set()
    return {key for key in original if key.lower().replace("-", "_") in targets}


def summarize_metadata(paths: list[Path], dataset_root: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        item: dict[str, Any] = {"path": relative(path, dataset_root)}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            item["content_summary"] = summarize_json(value)
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        summaries.append(item)
    return summaries


def id_index(paths: list[Path], dataset_root: Path) -> dict[str, Any]:
    grouped: dict[int, list[str]] = defaultdict(list)
    no_id: list[str] = []
    for path in paths:
        identifier = frame_id(path)
        if identifier is None:
            no_id.append(relative(path, dataset_root))
        else:
            grouped[identifier].append(relative(path, dataset_root))
    return {
        "ids": set(grouped),
        "by_id": grouped,
        "duplicates": {str(key): value for key, value in grouped.items() if len(value) > 1},
        "without_frame_id": no_id,
    }


def stream_directories(paths: list[Path], dataset_root: Path) -> dict[str, int]:
    counts = Counter(relative(path.parent, dataset_root) for path in paths)
    return dict(sorted(counts.items()))


def explicit_timestamp_unit(paths: list[Path], dataset_root: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(value: Any, prefix: str, source: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_path = f"{prefix}.{key}" if prefix else str(key)
                normalized = str(key).lower().replace("-", "_")
                if normalized in {"timestamp_unit", "time_unit", "timestamps_unit"}:
                    found.append({"path": source, "field": key_path, "value": item})
                visit(item, key_path, source)
        elif isinstance(value, list):
            for index, item in enumerate(value[:20]):
                visit(item, f"{prefix}[{index}]", source)

    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        visit(value, "", relative(path, dataset_root))
    return found


def audit_recording(dataset_root: Path, recording: str) -> tuple[dict[str, Any], dict[str, list[Path]], Path]:
    recording_root = find_recording(dataset_root, recording)
    report: dict[str, Any] = {
        "recording": recording,
        "recording_directory": relative(recording_root, dataset_root),
        "frame_id_extraction": "last decimal digit group in the filename stem",
        "warnings": [],
    }
    warning(
        report,
        "frame_id_filename_convention_unconfirmed",
        "frame IDs are extracted from the last decimal digit group in each filename stem; metadata or visual inspection must confirm this convention",
    )
    files: dict[str, list[Path]] = {name: [] for name in (*MODALITIES, "metadata", "unclassified")}
    directories: list[str] = []
    for current, dirs, names in os.walk(recording_root):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        current_path = Path(current)
        directories.append(relative(current_path, dataset_root))
        for name in sorted(names):
            path = current_path / name
            kind = classify(path)
            files[kind or "unclassified"].append(path)
    report["discovered_directories"] = directories
    report["major_directories"] = {
        kind: stream_directories(paths, dataset_root)
        for kind, paths in files.items()
        if kind != "unclassified" and paths
    }
    report["file_counts"] = {kind: len(files[kind]) for kind in MODALITIES}
    report["file_counts"].update(
        {
            "event_npz": sum(path.suffix.lower() == ".npz" for path in files["event"]),
            "event_preview_images": sum(path.suffix.lower() in IMAGE_SUFFIXES for path in files["event"]),
            "flow_npz": sum(path.suffix.lower() == ".npz" for path in files["flow"]),
            "metadata": len(files["metadata"]),
            "unclassified": len(files["unclassified"]),
        }
    )
    report["samples"] = {
        kind: [{"path": relative(path, dataset_root), "frame_id": frame_id(path)} for path in sample_evenly(paths)]
        for kind, paths in files.items()
        if paths
    }
    report["stream_directories"] = {
        kind: stream_directories(files[kind], dataset_root) for kind in MODALITIES if files[kind]
    }

    mapping_files = {
        "rgb": [path for path in files["rgb"] if path.suffix.lower() in IMAGE_SUFFIXES],
        "event": [path for path in files["event"] if path.suffix.lower() == ".npz"],
        "flow": [path for path in files["flow"] if path.suffix.lower() == ".npz"],
        "annotation": [path for path in files["annotation"] if path.suffix.lower() in ANNOTATION_SUFFIXES],
    }
    indexes = {kind: id_index(paths, dataset_root) for kind, paths in mapping_files.items()}
    all_ids = set().union(*(item["ids"] for item in indexes.values()))
    intersection = set.intersection(*(item["ids"] for item in indexes.values())) if indexes else set()
    report["frame_id_alignment"] = {
        "intersection": sorted(intersection),
        "intersection_count": len(intersection),
        "union_count": len(all_ids),
        "missing_by_modality": {kind: sorted(all_ids - item["ids"]) for kind, item in indexes.items()},
        "duplicates_by_modality": {kind: item["duplicates"] for kind, item in indexes.items()},
        "files_without_frame_id": {kind: item["without_frame_id"] for kind, item in indexes.items()},
    }
    for kind, item in indexes.items():
        if item["duplicates"]:
            warning(report, "duplicate_frame_id", f"{kind} has duplicate frame IDs; see frame_id_alignment.duplicates_by_modality")
        if item["without_frame_id"]:
            warning(report, "missing_frame_id", f"{kind} has files for which no frame ID could be extracted")
    for kind in mapping_files:
        if not mapping_files[kind]:
            warning(report, "missing_modality", f"no mappable {kind} files were discovered")

    image_groups = {
        "rgb": mapping_files["rgb"],
        "event_preview": [path for path in files["event"] if path.suffix.lower() in IMAGE_SUFFIXES],
        "semantic": [path for path in files["semantic"] if path.suffix.lower() in IMAGE_SUFFIXES],
        "instance": [path for path in files["instance"] if path.suffix.lower() in IMAGE_SUFFIXES],
    }
    report["image_resolutions"] = {
        kind: [image_info(path, dataset_root) for path in sample_evenly(paths)]
        for kind, paths in image_groups.items()
        if paths
    }

    event_npz = sorted(mapping_files["event"], key=lambda path: (frame_id(path) is None, frame_id(path), str(path)))
    event_details: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    sampled_event_paths = set(sample_evenly(event_npz))
    for path in event_npz:
        detail, endpoint = inspect_event_npz(path, dataset_root)
        if path in sampled_event_paths:
            event_details.append(detail)
        endpoints.append(endpoint)
        if detail.get("recognized_key_mapping") is None:
            warning(report, "unknown_event_npz_fields", f"could not identify x/y/p/t fields in {relative(path, dataset_root)}")
        if detail.get("timestamp_monotonic_nondecreasing") is False:
            warning(report, "non_monotonic_event_timestamp", f"event timestamps are not monotonic in {relative(path, dataset_root)}")
    report["event_npz_samples"] = event_details
    report["event_timestamp_files_checked"] = len(event_npz)
    report["event_timestamp_files_recognized"] = sum(item["recognized"] for item in endpoints)
    report["event_timestamp_boundaries"] = []
    for previous, current in zip(endpoints, endpoints[1:]):
        nondecreasing = None
        delta = None
        if previous["last"] is not None and current["first"] is not None:
            delta = scalar(current["first"] - previous["last"])
            nondecreasing = bool(current["first"] >= previous["last"])
        boundary = {
            "previous_path": previous["path"],
            "current_path": current["path"],
            "previous_last": previous["last"],
            "current_first": current["first"],
            "boundary_delta": delta,
            "nondecreasing_across_boundary": nondecreasing,
            "continuous": "unknown_without_timestamp_unit_and_interval_convention",
        }
        report["event_timestamp_boundaries"].append(boundary)
        if nondecreasing is False:
            warning(report, "event_timestamp_boundary_overlap", f"timestamp decreases from {previous['path']} to {current['path']}")

    flow_npz = sorted(mapping_files["flow"], key=lambda path: (frame_id(path) is None, frame_id(path), str(path)))
    report["flow_npz_samples"] = [inspect_flow_npz(path, dataset_root) for path in sample_evenly(flow_npz)]
    annotation_files = sorted(mapping_files["annotation"], key=lambda path: (frame_id(path) is None, frame_id(path), str(path)))
    report["annotation_samples"] = [inspect_annotation(path, dataset_root) for path in sample_evenly(annotation_files)]
    for item in report["annotation_samples"]:
        if item.get("format") != "json" or not item.get("bbox_fields"):
            warning(report, "annotation_schema_unconfirmed", f"annotation field semantics are not fully confirmed for {item['path']}")

    report["metadata"] = summarize_metadata(files["metadata"], dataset_root)
    report["timestamp_unit_evidence"] = explicit_timestamp_unit(files["metadata"], dataset_root)
    if not report["timestamp_unit_evidence"]:
        warning(report, "timestamp_unit_unknown", "no explicit timestamp unit field was found in metadata.json")
    warning(
        report,
        "event_interval_semantics_unconfirmed",
        "static paths and matching frame IDs do not prove whether an Event file covers the causal interval ending at that RGB frame",
    )
    warning(
        report,
        "flow_semantics_unconfirmed",
        "flow direction, coordinate convention, and source/target frame relationship require metadata or visual confirmation",
    )
    warning(
        report,
        "event_coordinate_convention_unconfirmed",
        "Event previews render x as image columns and y as image rows; metadata or visual inspection must confirm orientation",
    )
    if files["unclassified"]:
        warning(report, "unclassified_files", f"{len(files['unclassified'])} files were not assigned to a modality")

    event_content_is_usable = bool(endpoints) and all(
        item["recognized"] and item["monotonic"] is not False for item in endpoints
    )
    report["mapping_preview_eligible"] = (
        mapping_is_clear(mapping_files, indexes, intersection) and event_content_is_usable
    )
    report["mapping_preview_basis"] = (
        "unique files with identical extracted frame IDs in one stream directory per modality"
        if report["mapping_preview_eligible"]
        else "not eligible; inspect missing modalities, Event NPZ/timestamps, duplicate IDs, stream-directory ambiguity, and consecutive IDs"
    )
    return report, mapping_files, recording_root


def mapping_is_clear(
    mapping_files: dict[str, list[Path]], indexes: dict[str, dict[str, Any]], intersection: set[int]
) -> bool:
    if any(not paths for paths in mapping_files.values()):
        return False
    if any(len({path.parent for path in paths}) != 1 for paths in mapping_files.values()):
        return False
    if any(item["duplicates"] or item["without_frame_id"] for item in indexes.values()):
        return False
    return len(first_consecutive_run(sorted(intersection), 20)) == 20


def first_consecutive_run(ids: list[int], length: int) -> list[int]:
    run: list[int] = []
    for identifier in ids:
        if not run or identifier == run[-1] + 1:
            run.append(identifier)
        else:
            run = [identifier]
        if len(run) == length:
            return run
    return []


def build_manifest_rows(
    recording: str, mapping_files: dict[str, list[Path]], dataset_root: Path
) -> list[dict[str, Any]]:
    indexes = {kind: id_index(paths, dataset_root) for kind, paths in mapping_files.items()}
    intersection = set.intersection(*(item["ids"] for item in indexes.values()))
    ids = first_consecutive_run(sorted(intersection), 20)
    if len(ids) != 20:
        return []
    rows = []
    for identifier in ids:
        rows.append(
            {
                "recording": recording,
                "frame_id": identifier,
                "rgb_path": indexes["rgb"]["by_id"][identifier][0],
                "event_path": indexes["event"]["by_id"][identifier][0],
                "flow_path": indexes["flow"]["by_id"][identifier][0],
                "annotation_path": indexes["annotation"]["by_id"][identifier][0],
            }
        )
    return rows


def load_event_accumulation(path: Path, size: tuple[int, int]) -> Image.Image | None:
    with np.load(path, allow_pickle=False) as archive:
        mapping = identify_event_keys(archive.files)
        if mapping is None:
            return None
        x = np.asarray(archive[mapping["x"]]).reshape(-1).astype(np.int64)
        y = np.asarray(archive[mapping["y"]]).reshape(-1).astype(np.int64)
        p = np.asarray(archive[mapping["p"]]).reshape(-1)
    width, height = size
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    positive = np.zeros((height, width), dtype=np.float32)
    negative = np.zeros((height, width), dtype=np.float32)
    np.add.at(positive, (y[valid & (p > 0)], x[valid & (p > 0)]), 1)
    np.add.at(negative, (y[valid & (p <= 0)], x[valid & (p <= 0)]), 1)
    scale = max(float(np.percentile(np.maximum(positive, negative), 99)), 1.0)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    canvas[..., 0] = np.clip(245 - 220 * negative / scale, 0, 255)
    canvas[..., 1] = np.clip(245 - 180 * np.maximum(positive, negative) / scale, 0, 255)
    canvas[..., 2] = np.clip(245 - 220 * positive / scale, 0, 255)
    return Image.fromarray(canvas)


def load_flow_magnitude(path: Path) -> Image.Image | None:
    with np.load(path, allow_pickle=False) as archive:
        candidate = None
        for key in archive.files:
            array = np.asarray(archive[key])
            if array.ndim == 3 and array.shape[-1] == 2:
                candidate = array
                break
            if array.ndim == 3 and array.shape[0] == 2:
                candidate = np.moveaxis(array, 0, -1)
                break
        if candidate is None:
            return None
    magnitude = np.linalg.norm(candidate.astype(np.float32), axis=-1)
    high = float(np.percentile(magnitude[np.isfinite(magnitude)], 99)) if np.isfinite(magnitude).any() else 0.0
    normalized = np.clip(magnitude / high, 0, 1) if high > 0 else np.zeros_like(magnitude)
    heat = np.stack(
        [255 * normalized, 255 * np.sqrt(normalized), 255 * (1 - normalized)], axis=-1
    ).astype(np.uint8)
    return Image.fromarray(heat)


def explicit_boxes(value: Any) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    if isinstance(value, dict):
        lowered = {str(key).lower().replace("-", "_"): item for key, item in value.items()}
        variants = [
            ("xmin", "ymin", "xmax", "ymax"),
            ("x_min", "y_min", "x_max", "y_max"),
        ]
        for names in variants:
            if all(name in lowered and isinstance(lowered[name], (int, float)) for name in names):
                boxes.append(tuple(float(lowered[name]) for name in names))
                break
        if all(name in lowered and isinstance(lowered[name], (int, float)) for name in ("x", "y", "width", "height")):
            x, y, width, height = (float(lowered[name]) for name in ("x", "y", "width", "height"))
            boxes.append((x, y, x + width, y + height))
        for item in value.values():
            boxes.extend(explicit_boxes(item))
    elif isinstance(value, list):
        for item in value:
            boxes.extend(explicit_boxes(item))
    return boxes


def annotation_overlay(rgb: Image.Image, path: Path) -> Image.Image | None:
    if path.suffix.lower() != ".json":
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    boxes = explicit_boxes(value)
    if not boxes:
        return None
    output = rgb.copy()
    draw = ImageDraw.Draw(output)
    for box in boxes:
        draw.rectangle(box, outline=(255, 32, 32), width=3)
    return output


def labeled_panel(image: Image.Image, label: str, cell_size: tuple[int, int]) -> Image.Image:
    width, height = cell_size
    panel = Image.new("RGB", (width, height + 24), "white")
    fitted = image.convert("RGB")
    fitted.thumbnail((width, height))
    panel.paste(fitted, ((width - fitted.width) // 2, 24 + (height - fitted.height) // 2))
    ImageDraw.Draw(panel).text((5, 5), label, fill="black")
    return panel


def create_visualization(
    rows: list[dict[str, Any]], dataset_root: Path, destination: Path
) -> tuple[bool, str | None]:
    if len(rows) < 3:
        return False, "fewer than three aligned manifest-preview rows are available"
    chosen = sample_evenly(rows, 3)
    columns: list[list[Image.Image]] = []
    for row in chosen:
        try:
            rgb = Image.open(dataset_root / row["rgb_path"]).convert("RGB")
            event = load_event_accumulation(dataset_root / row["event_path"], rgb.size)
            flow = load_flow_magnitude(dataset_root / row["flow_path"])
            boxes = annotation_overlay(rgb, dataset_root / row["annotation_path"])
        except Exception as exc:
            return False, f"frame {row['frame_id']}: {type(exc).__name__}: {exc}"
        if event is None or flow is None or boxes is None:
            return False, f"frame {row['frame_id']}: event, flow, or explicit annotation-box layout is unrecognized"
        columns.append([rgb, event, flow, boxes])
    cell = (480, 270)
    canvas = Image.new("RGB", (cell[0] * 3, (cell[1] + 24) * 4), "white")
    labels = ("RGB", "Event accumulation (x right, y down)", "Flow magnitude", "Annotation boxes")
    for column, (row, images) in enumerate(zip(chosen, columns)):
        for row_index, (label, image) in enumerate(zip(labels, images)):
            panel = labeled_panel(image, f"{label}; frame_id={row['frame_id']}", cell)
            canvas.paste(panel, (column * cell[0], row_index * (cell[1] + 24)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return True, None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="SEVD dataset root")
    parser.add_argument("--recording", required=True, action="append", help="recording name; repeat this option")
    parser.add_argument("--output", required=True, type=Path, help="JSON audit report path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.root.expanduser().resolve()
    if not dataset_root.is_dir():
        print(f"error: dataset root is not a directory: {dataset_root}", file=sys.stderr)
        return 2
    output = args.output.expanduser().resolve()
    combined: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "recordings_requested": args.recording,
        "recordings": [],
        "warnings": [],
        "scope": "static, read-only directory and file-content audit",
    }
    manifests: list[dict[str, Any]] = []
    visualizations: list[dict[str, Any]] = []
    try:
        for name in args.recording:
            report, mapping_files, _recording_root = audit_recording(dataset_root, name)
            combined["recordings"].append(report)
            if report["mapping_preview_eligible"]:
                rows = build_manifest_rows(name, mapping_files, dataset_root)
                if not manifests:
                    manifests.extend(rows)
                preview_path = output.parent / f"alignment_preview_{name}.png"
                created, reason = create_visualization(rows, dataset_root, preview_path)
                visualizations.append(
                    {
                        "recording": name,
                        "path": preview_path.name if created else None,
                        "created": created,
                        "warning": reason,
                    }
                )
                if reason:
                    warning(report, "visualization_not_created", reason)
            else:
                visualizations.append(
                    {"recording": name, "path": None, "created": False, "warning": "mapping preview is not eligible"}
                )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    combined["visualizations"] = visualizations
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = output.parent / "manifest_preview.jsonl"
    if manifests:
        manifest_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifests), encoding="utf-8"
        )
        print(f"wrote {output} and {manifest_path}")
    else:
        print(f"wrote {output}; manifest preview was not created because mapping was not confirmed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
