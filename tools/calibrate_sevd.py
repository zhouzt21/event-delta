#!/usr/bin/env python3
"""CPU-only SEVD Event/flow temporal and spatial convention calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    from tools.audit_sevd import audit_recording, event_arrays
except ModuleNotFoundError:  # Direct invocation: python tools/calibrate_sevd.py
    from audit_sevd import audit_recording, event_arrays


EVENT_TRANSFORMS = ("identity", "horizontal_flip", "vertical_flip", "horizontal_and_vertical_flip")
EVENT_TIMES = ("previous_to_current", "current_to_next")
TOP_FRACTIONS = (0.01, 0.02, 0.05)
FLOW_SCALES = ("raw", "image_size", "half_image_size")
FLOW_TIMES = EVENT_TIMES
FLOW_SIGNS = (1, -1)
BACKGROUND_TREATMENTS = ("raw", "subtract_background")
QUANTILES = (0, 1, 10, 50, 90, 95, 99, 99.9, 100)
VERSION = "1.0"
WARP_EQUATION = (
    "forward bilinear splat: for each valid source pixel (x,y), "
    "target(x + sign*sx*(u-bg_u), y + sign*sy*(v-bg_v)) += source(x,y)*bilinear_weight; "
    "photometric MAE is evaluated only where accumulated weight > 0"
)


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def unique_path(index: dict[str, Any], identifier: int) -> Path | None:
    paths = index["by_id"].get(identifier, [])
    return paths[0] if len(paths) == 1 else None


def stratified_sample(items: list[int], count: int, seed: int) -> list[int]:
    """Deterministic, timeline-wide sampling with one seeded choice per stratum."""
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    rng = random.Random(seed)
    edges = np.linspace(0, len(items), count + 1)
    selected = []
    for start, stop in zip(edges[:-1], edges[1:]):
        lo, hi = int(math.ceil(start)), int(math.ceil(stop))
        hi = max(lo + 1, min(hi, len(items)))
        selected.append(items[rng.randrange(lo, hi)])
    return selected


def distribute_samples(total: int, recordings: list[str]) -> dict[str, int]:
    base, remainder = divmod(max(0, total), len(recordings))
    return {name: base + (position < remainder) for position, name in enumerate(recordings)}


def eligible_frame_ids(state: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]]]:
    ids = state["intersection"]
    eligible, skipped = [], []
    rgb_ids = state["indexes"]["rgb_frame"]["ids"]
    for position, identifier in enumerate(ids):
        reasons = []
        if position == 0 or position == len(ids) - 1:
            reasons.append("recording_boundary")
        elif identifier - 1 not in rgb_ids or identifier + 1 not in rgb_ids:
            reasons.append("missing_adjacent_rgb")
        if any(unique_path(state["indexes"][kind], identifier) is None
               for kind in ("rgb_frame", "event_npz", "flow_npz")):
            reasons.append("non_unique_canonical_path")
        if reasons:
            skipped.append({"frame_id": identifier, "reasons": sorted(set(reasons))})
        else:
            eligible.append(identifier)
    return eligible, skipped


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def read_event_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays, _schema = event_arrays(archive)
        if arrays is None:
            raise ValueError(f"unrecognized Event schema: {path}")
        output = {name: np.asarray(value).reshape(-1) for name, value in arrays.items()}
    if len({len(value) for value in output.values()}) != 1:
        raise ValueError(f"Event fields have different lengths: {path}")
    return output


def event_count_map(events: dict[str, np.ndarray], shape: tuple[int, int], polarity: bool | None = None) -> np.ndarray:
    height, width = shape
    x = events["x"].astype(np.int64, copy=False)
    y = events["y"].astype(np.int64, copy=False)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if polarity is not None:
        valid &= events["pol"].astype(bool, copy=False) == polarity
    counts = np.zeros(shape, dtype=np.float32)
    np.add.at(counts, (y[valid], x[valid]), 1.0)
    return np.log1p(counts)


def transform_map(array: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return array
    if transform == "horizontal_flip":
        return array[:, ::-1]
    if transform == "vertical_flip":
        return array[::-1, :]
    if transform == "horizontal_and_vertical_flip":
        return array[::-1, ::-1]
    raise ValueError(f"unknown transform: {transform}")


def smooth(array: np.ndarray, radius: float = 1.5) -> np.ndarray:
    if not np.any(array):
        return array.astype(np.float32, copy=True)
    scale = float(np.max(array))
    image = Image.fromarray(np.clip(array / scale * 255, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32) / 255.0 * scale


def pearson_activity(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = (a > 0) | (b > 0)
    if np.count_nonzero(mask) < 3:
        return None
    av, bv = a[mask].astype(np.float64), b[mask].astype(np.float64)
    if np.std(av) <= 1e-12 or np.std(bv) <= 1e-12:
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def top_mask(array: np.ndarray, fraction: float) -> np.ndarray:
    flat = array.reshape(-1)
    active = np.flatnonzero(flat > 0)
    mask = np.zeros(flat.size, dtype=bool)
    if not active.size:
        return mask.reshape(array.shape)
    count = min(active.size, max(1, int(math.ceil(flat.size * fraction))))
    active_values = flat[active]
    indices = active[np.argpartition(active_values, active_values.size - count)[active_values.size - count:]]
    mask[indices] = True
    return mask.reshape(array.shape)


def overlap_metrics(a: np.ndarray, b: np.ndarray, fraction: float) -> tuple[float, float]:
    am, bm = top_mask(a, fraction), top_mask(b, fraction)
    intersection = int(np.count_nonzero(am & bm))
    union = int(np.count_nonzero(am | bm))
    return intersection / union if union else 0.0, intersection / max(1, int(np.count_nonzero(am)))


def event_sample_scores(event: np.ndarray, previous_diff: np.ndarray,
                        next_diff: np.ndarray) -> list[dict[str, Any]]:
    differences = {"previous_to_current": smooth(previous_diff), "current_to_next": smooth(next_diff)}
    rows = []
    for transform in EVENT_TRANSFORMS:
        transformed = smooth(transform_map(event, transform))
        for temporal, difference in differences.items():
            row: dict[str, Any] = {"temporal_alignment": temporal, "spatial_transform": transform,
                                   "correlation": pearson_activity(transformed, difference)}
            overlap_values = []
            for fraction in TOP_FRACTIONS:
                iou, precision = overlap_metrics(transformed, difference, fraction)
                label = str(int(fraction * 100))
                row[f"top_{label}_iou"] = iou
                row[f"top_{label}_precision"] = precision
                overlap_values.extend((iou, precision))
            row["score"] = ((row["correlation"] if row["correlation"] is not None else -1.0)
                            + float(np.mean(overlap_values))) / 2.0
            rows.append(row)
    return rows


def aggregate_event_scores(sample_rows: list[dict[str, Any]], recording: str) -> list[dict[str, Any]]:
    output = []
    for temporal in EVENT_TIMES:
        for transform in EVENT_TRANSFORMS:
            matches = [row for row in sample_rows if row["temporal_alignment"] == temporal
                       and row["spatial_transform"] == transform]
            item: dict[str, Any] = {"recording": recording, "temporal_alignment": temporal,
                                    "spatial_transform": transform, "valid_samples": len(matches)}
            for key in ("correlation", "top_1_iou", "top_1_precision", "top_2_iou",
                        "top_2_precision", "top_5_iou", "top_5_precision", "score"):
                values = [row[key] for row in matches if row.get(key) is not None and np.isfinite(row[key])]
                item[f"{key}_mean"] = float(np.mean(values)) if values else None
                item[f"{key}_median"] = float(np.median(values)) if values else None
                item[f"{key}_valid_samples"] = len(values)
            output.append(item)
    output.sort(key=lambda row: row["score_mean"] if row["score_mean"] is not None else -math.inf, reverse=True)
    for rank, row in enumerate(output, 1):
        row["rank"] = rank
    if output:
        second = output[1]["score_mean"] if len(output) > 1 else None
        output[0]["second_best_margin"] = (output[0]["score_mean"] - second
                                            if output[0]["score_mean"] is not None and second is not None else None)
    return output


def read_flow(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "flow" not in archive.files:
            raise ValueError(f"flow key missing: {path}")
        flow = np.asarray(archive["flow"], dtype=np.float32)
    if flow.ndim == 3 and flow.shape[0] == 2 and flow.shape[-1] != 2:
        flow = np.moveaxis(flow, 0, -1)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"expected HxWx2 flow, got {flow.shape}")
    return flow


def robust_background(flow: np.ndarray) -> np.ndarray:
    finite = np.isfinite(flow).all(axis=-1)
    if not np.any(finite):
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.median(flow[finite].astype(np.float64), axis=0)


def quantile_dict(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return {f"q{q:g}": None for q in QUANTILES}
    found = np.percentile(values, QUANTILES)
    return {f"q{q:g}": float(value) for q, value in zip(QUANTILES, found)}


def flow_frame_statistics(flow: np.ndarray, frame_id: int | None = None) -> dict[str, Any]:
    finite_vectors = np.isfinite(flow).all(axis=-1)
    magnitude = np.linalg.norm(flow.astype(np.float64), axis=-1)
    background = robust_background(flow)
    exact = np.all(flow == np.float32(-1 / 1024), axis=-1)
    zero = np.all(flow == 0, axis=-1)
    item: dict[str, Any] = {
        "frame_id": frame_id, "shape": list(flow.shape), "vector_count": int(finite_vectors.size),
        "finite_vector_count": int(np.count_nonzero(finite_vectors)),
        "non_finite_vector_count": int(np.count_nonzero(~finite_vectors)),
        "background_vector": background.tolist(),
        "exact_minus_1_over_1024_ratio": float(np.mean(exact)), "zero_vector_ratio": float(np.mean(zero)),
    }
    for name, values in (("u", flow[..., 0]), ("v", flow[..., 1]), ("magnitude", magnitude)):
        finite = values[np.isfinite(values)]
        item[name] = {"min": float(np.min(finite)) if finite.size else None,
                      "max": float(np.max(finite)) if finite.size else None, **quantile_dict(finite)}
    return item


def aggregate_flow_statistics(entries: list[tuple[int, np.ndarray]], sample_limit: int = 20000) -> dict[str, Any]:
    frames, samples = [], {"u": [], "v": [], "magnitude": []}
    finite_count = nonfinite_count = vector_count = exact_count = zero_count = 0
    backgrounds = []
    for identifier, flow in entries:
        stats = flow_frame_statistics(flow, identifier)
        frames.append(stats)
        vector_count += stats["vector_count"]
        finite_count += stats["finite_vector_count"]
        nonfinite_count += stats["non_finite_vector_count"]
        exact_count += int(round(stats["exact_minus_1_over_1024_ratio"] * stats["vector_count"]))
        zero_count += int(round(stats["zero_vector_ratio"] * stats["vector_count"]))
        backgrounds.append(stats["background_vector"])
        flat = flow.reshape(-1, 2)
        step = max(1, int(math.ceil(len(flat) / sample_limit)))
        subset = flat[::step]
        samples["u"].append(subset[:, 0])
        samples["v"].append(subset[:, 1])
        samples["magnitude"].append(np.linalg.norm(subset.astype(np.float64), axis=-1))
    overall = {"vector_count": vector_count, "finite_vector_count": finite_count,
               "non_finite_vector_count": nonfinite_count,
               "background_vector": np.nanmedian(np.asarray(backgrounds), axis=0).tolist() if backgrounds else [None, None],
               "exact_minus_1_over_1024_ratio": exact_count / vector_count if vector_count else None,
               "zero_vector_ratio": zero_count / vector_count if vector_count else None,
               "quantile_method": f"deterministic stride sample, at most {sample_limit} vectors per frame"}
    for name, chunks in samples.items():
        values = np.concatenate(chunks) if chunks else np.array([])
        finite = values[np.isfinite(values)]
        overall[name] = {"min": min((frame[name]["min"] for frame in frames if frame[name]["min"] is not None), default=None),
                         "max": max((frame[name]["max"] for frame in frames if frame[name]["max"] is not None), default=None),
                         **quantile_dict(finite)}
    return {"per_frame": frames, "overall": overall}


def read_kitti_objects(path: Path | None) -> dict[str, tuple[float, float, float, float]]:
    objects = {}
    if path is None or not path.is_file():
        return objects
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split()
        try:
            box = tuple(float(value) for value in tokens[4:8])
            object_id = tokens[-1]
        except (IndexError, ValueError):
            continue
        objects[object_id] = box  # Last duplicate wins deterministically.
    return objects


def match_kitti_objects(source: dict[str, tuple[float, float, float, float]],
                        target: dict[str, tuple[float, float, float, float]]) -> list[dict[str, Any]]:
    pairs = []
    for object_id in sorted(source.keys() & target.keys()):
        source_box, target_box = source[object_id], target[object_id]
        source_center = np.array([(source_box[0] + source_box[2]) / 2, (source_box[1] + source_box[3]) / 2])
        target_center = np.array([(target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2])
        pairs.append({"object_id": object_id, "source_bbox": source_box, "target_bbox": target_box,
                      "displacement": target_center - source_center})
    return pairs


def bbox_flow_median(flow: np.ndarray, box: Iterable[float], minimum_size: float = 4.0) -> np.ndarray | None:
    left, top, right, bottom = box
    if right - left < minimum_size or bottom - top < minimum_size:
        return None
    height, width = flow.shape[:2]
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half_w, half_h = (right - left) / 4, (bottom - top) / 4
    x0, x1 = max(0, int(math.floor(cx - half_w))), min(width, int(math.ceil(cx + half_w)))
    y0, y1 = max(0, int(math.floor(cy - half_h))), min(height, int(math.ceil(cy + half_h)))
    if x1 <= x0 or y1 <= y0:
        return None
    values = flow[y0:y1, x0:x1].reshape(-1, 2)
    values = values[np.isfinite(values).all(axis=1)]
    return np.median(values, axis=0) if values.size else None


def flow_scale(name: str, width: int, height: int) -> np.ndarray:
    if name == "raw":
        return np.array([1.0, 1.0])
    if name == "image_size":
        return np.array([float(width), float(height)])
    if name == "half_image_size":
        return np.array([width / 2.0, height / 2.0])
    raise ValueError(name)


def candidate_displacement(flow: np.ndarray, scale: str, sign: int, background_treatment: str,
                           background: np.ndarray | None = None) -> np.ndarray:
    height, width = flow.shape[:2]
    base = flow.astype(np.float64)
    if background_treatment == "subtract_background":
        base = base - (robust_background(flow) if background is None else background)
    return base * flow_scale(scale, width, height) * sign


def resize_array(array: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if array.ndim == 2:
        return np.asarray(Image.fromarray(array.astype(np.float32), mode="F").resize(size, resample), dtype=np.float32)
    channels = [resize_array(array[..., channel], size, nearest) for channel in range(array.shape[-1])]
    return np.stack(channels, axis=-1)


def photometric_displacement(flow: np.ndarray, original_shape: tuple[int, int], scale: str, sign: int,
                             background_treatment: str, background: np.ndarray) -> np.ndarray:
    """Convert flow into pixels of the resized photometric evaluation grid."""
    photo_height, photo_width = flow.shape[:2]
    original_height, original_width = original_shape
    base = flow.astype(np.float64)
    if background_treatment == "subtract_background":
        base -= background
    original_scale = flow_scale(scale, original_width, original_height)
    resize_scale = np.array([photo_width / original_width, photo_height / original_height])
    return base * original_scale * resize_scale * sign


def forward_splat(source: np.ndarray, displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward bilinear splat source using source-indexed displacement."""
    height, width = source.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float64)
    tx, ty = xx + displacement[..., 0], yy + displacement[..., 1]
    valid = np.isfinite(tx) & np.isfinite(ty)
    output = np.zeros_like(source, dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    channels = 1 if source.ndim == 2 else source.shape[2]
    source_values = source[..., None] if source.ndim == 2 else source
    for x_index, y_index, weight in ((np.floor(tx), np.floor(ty), (1 - (tx % 1)) * (1 - (ty % 1))),
                                     (np.ceil(tx), np.floor(ty), (tx % 1) * (1 - (ty % 1))),
                                     (np.floor(tx), np.ceil(ty), (1 - (tx % 1)) * (ty % 1)),
                                     (np.ceil(tx), np.ceil(ty), (tx % 1) * (ty % 1))):
        xi, yi = x_index.astype(np.int64), y_index.astype(np.int64)
        keep = valid & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height) & (weight > 0)
        np.add.at(weights, (yi[keep], xi[keep]), weight[keep])
        for channel in range(channels):
            target = output if channels == 1 else output[..., channel]
            np.add.at(target, (yi[keep], xi[keep]), source_values[..., channel][keep] * weight[keep])
    mask = weights > 1e-8
    if channels == 1:
        output[mask] /= weights[mask]
    else:
        output[mask] /= weights[mask, None]
    return output.astype(np.float32), mask


def photometric_metrics(source: np.ndarray, target: np.ndarray, displacement: np.ndarray,
                        motion_mask: np.ndarray) -> dict[str, Any]:
    warped, valid = forward_splat(source, displacement)
    residual = np.mean(np.abs(warped - target), axis=-1) if target.ndim == 3 else np.abs(warped - target)
    focused = valid & motion_mask
    return {"mae": float(np.mean(residual[valid])) if np.any(valid) else None,
            "motion_mae": float(np.mean(residual[focused])) if np.any(focused) else None,
            "valid_pixels": int(np.count_nonzero(valid)), "motion_valid_pixels": int(np.count_nonzero(focused)),
            "warped": warped, "valid_mask": valid, "residual": residual}


def annotation_path(state: dict[str, Any], identifier: int) -> Path | None:
    selected = state.get("annotation")
    if selected is None or selected.get("sensor") != "rgb":
        return None
    paths = selected["index"]["by_id"].get(identifier, [])
    return paths[0] if len(paths) == 1 else None


def candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return row["scale"], row["sign"], row["time_pair"], row["background_treatment"]


def score_flow_candidates(samples: list[dict[str, Any]], recording: str) -> list[dict[str, Any]]:
    accum: dict[tuple[Any, ...], dict[str, list[float] | int]] = {}
    for scale in FLOW_SCALES:
        for sign in FLOW_SIGNS:
            for time_pair in FLOW_TIMES:
                for treatment in BACKGROUND_TREATMENTS:
                    accum[(scale, sign, time_pair, treatment)] = {"object_all": [], "object_motion": [],
                                                                  "object_norm_all": [], "photo": [],
                                                                  "photo_motion": [], "photo_pixels": 0,
                                                                  "photo_motion_pixels": 0}
    for sample in samples:
        flow, background = sample["flow"], sample["background"]
        for key, values in accum.items():
            scale, sign, time_pair, treatment = key
            for pair in sample["object_pairs"][time_pair]:
                raw_estimate = bbox_flow_median(flow, pair["source_bbox"])
                if raw_estimate is None:
                    continue
                if treatment == "subtract_background":
                    raw_estimate = raw_estimate - background
                estimate = raw_estimate * flow_scale(scale, flow.shape[1], flow.shape[0]) * sign
                truth = pair["displacement"]
                error = float(np.linalg.norm(estimate - truth))
                motion = float(np.linalg.norm(truth))
                values["object_all"].append(error)  # type: ignore[union-attr]
                values["object_norm_all"].append(error / max(motion, 1.0))  # type: ignore[union-attr]
                if motion >= 1.0:
                    values["object_motion"].append(error)  # type: ignore[union-attr]
            pair_images = sample["photo_images"][time_pair]
            photo_displacement = photometric_displacement(sample["photo_flow"], flow.shape[:2], scale, sign,
                                                          treatment, background)
            photo = photometric_metrics(pair_images[0], pair_images[1], photo_displacement,
                                        sample["photo_motion_masks"][time_pair])
            if photo["mae"] is not None:
                values["photo"].append(photo["mae"])  # type: ignore[union-attr]
                values["photo_pixels"] += photo["valid_pixels"]  # type: ignore[operator]
            if photo["motion_mae"] is not None:
                values["photo_motion"].append(photo["motion_mae"])  # type: ignore[union-attr]
                values["photo_motion_pixels"] += photo["motion_valid_pixels"]  # type: ignore[operator]
    rows = []
    for key, values in accum.items():
        scale, sign, time_pair, treatment = key
        all_errors = values["object_all"]
        motion_errors = values["object_motion"]
        norm_errors = values["object_norm_all"]
        photo, photo_motion = values["photo"], values["photo_motion"]
        rows.append({"recording": recording, "scale": scale, "sign": sign, "time_pair": time_pair,
                     "background_treatment": treatment,
                     "object_all_count": len(all_errors), "object_all_median_epe": median_or_none(all_errors),
                     "object_all_mean_epe": mean_or_none(all_errors),
                     "object_normalized_mean_error": mean_or_none(norm_errors),
                     "object_motion_count": len(motion_errors), "object_motion_median_epe": median_or_none(motion_errors),
                     "object_motion_mean_epe": mean_or_none(motion_errors),
                     "photometric_sample_count": len(photo), "photometric_mae": mean_or_none(photo),
                     "photometric_motion_sample_count": len(photo_motion),
                     "photometric_motion_mae": mean_or_none(photo_motion),
                     "photometric_valid_pixels": values["photo_pixels"],
                     "photometric_motion_valid_pixels": values["photo_motion_pixels"]})
    rank_flow_rows(rows)
    return rows


def mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def median_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.median(values)) if values else None


def rank_flow_rows(rows: list[dict[str, Any]]) -> None:
    object_key = lambda row: (row["object_motion_median_epe"] if row["object_motion_count"] else
                              row["object_all_median_epe"] if row["object_all_count"] else math.inf)
    photo_key = lambda row: (row["photometric_motion_mae"] if row["photometric_motion_sample_count"] else
                             row["photometric_mae"] if row["photometric_sample_count"] else math.inf)
    for rank, row in enumerate(sorted(rows, key=object_key), 1):
        row["object_rank"] = rank
    for rank, row in enumerate(sorted(rows, key=photo_key), 1):
        row["photometric_rank"] = rank
    for row in rows:
        row["combined_rank_sum"] = row["object_rank"] + row["photometric_rank"]
    rows.sort(key=lambda row: (row["combined_rank_sum"], object_key(row), photo_key(row)))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank


def bbox_union_mask(shape: tuple[int, int], objects: dict[str, tuple[float, float, float, float]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for left, top, right, bottom in objects.values():
        x0, x1 = max(0, int(left)), min(width, int(math.ceil(right)))
        y0, y1 = max(0, int(top)), min(height, int(math.ceil(bottom)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def make_sample(state: dict[str, Any], identifier: int) -> dict[str, Any]:
    previous_id, next_id = identifier - 1, identifier + 1
    rgb_paths = {key: unique_path(state["indexes"]["rgb_frame"], value)
                 for key, value in (("previous", previous_id), ("current", identifier), ("next", next_id))}
    if any(path is None for path in rgb_paths.values()):
        raise FileNotFoundError("missing adjacent RGB")
    images = {key: load_rgb(path) for key, path in rgb_paths.items()}  # type: ignore[arg-type]
    grays = {key: np.mean(value, axis=-1) for key, value in images.items()}
    flow = read_flow(unique_path(state["indexes"]["flow_npz"], identifier))  # type: ignore[arg-type]
    if flow.shape[:2] != images["current"].shape[:2]:
        raise ValueError(f"flow/RGB shape mismatch for frame {identifier}")
    event_path = unique_path(state["indexes"]["event_npz"], identifier)
    events = read_event_npz(event_path)  # type: ignore[arg-type]
    event = event_count_map(events, flow.shape[:2])
    objects = {key: read_kitti_objects(annotation_path(state, value)) for key, value in
               (("previous", previous_id), ("current", identifier), ("next", next_id))}
    pair_names = {"previous_to_current": ("previous", "current"), "current_to_next": ("current", "next")}
    object_pairs, pair_images, motion_masks = {}, {}, {}
    for name, (source_name, target_name) in pair_names.items():
        object_pairs[name] = match_kitti_objects(objects[source_name], objects[target_name])
        pair_images[name] = (images[source_name], images[target_name])
        difference = np.mean(np.abs(images[target_name] - images[source_name]), axis=-1)
        threshold = np.percentile(difference, 90)
        motion_masks[name] = (difference >= threshold) | bbox_union_mask(flow.shape[:2], objects[source_name])
    height, width = flow.shape[:2]
    photo_ratio = min(1.0, 320.0 / max(height, width))
    photo_size = (max(1, round(width * photo_ratio)), max(1, round(height * photo_ratio)))
    photo_images = {name: (resize_array(pair[0], photo_size), resize_array(pair[1], photo_size))
                    for name, pair in pair_images.items()}
    photo_motion_masks = {name: resize_array(mask.astype(np.float32), photo_size, nearest=True).astype(bool)
                          for name, mask in motion_masks.items()}
    return {"frame_id": identifier, "neighbor_frame_ids": [previous_id, next_id], "flow": flow,
            "background": robust_background(flow), "event": event, "events": events,
            "event_scores": event_sample_scores(event, np.abs(grays["current"] - grays["previous"]),
                                                 np.abs(grays["next"] - grays["current"])),
            "images": pair_images, "all_images": images, "motion_masks": motion_masks,
            "photo_images": photo_images, "photo_motion_masks": photo_motion_masks,
            "photo_flow": resize_array(flow, photo_size), "photometric_resolution": list(reversed(photo_size)),
            "object_pairs": object_pairs, "objects": objects}


def best_and_status_event(by_recording: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    best = {name: rows[0] for name, rows in by_recording.items() if rows}
    conventions = {(row["temporal_alignment"], row["spatial_transform"]) for row in best.values()}
    margins = [row.get("second_best_margin") for row in best.values()]
    confirmed = (len(by_recording) >= 2 and len(best) == len(by_recording) and len(conventions) == 1 and
                 all(margin is not None and margin > 1e-6 for margin in margins))
    convention = next(iter(conventions)) if len(conventions) == 1 else (None, None)
    return {"status": "confirmed" if confirmed else "ambiguous", "temporal_alignment": convention[0],
            "spatial_transform": convention[1], "per_recording_best": best,
            "criterion": "at least two recordings agree and each best score has a strictly positive (>1e-6) second-best margin"}


def best_and_status_flow(by_recording: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    details, conventions = {}, set()
    confirmed = True
    for name, rows in by_recording.items():
        object_rows = [row for row in rows if row["object_all_count"]]
        photo_rows = [row for row in rows if row["photometric_sample_count"]]
        object_metric = lambda row: (row["object_motion_median_epe"] if row["object_motion_count"]
                                     else row["object_all_median_epe"])
        photo_metric = lambda row: (row["photometric_motion_mae"] if row["photometric_motion_sample_count"]
                                    else row["photometric_mae"])
        object_ordered = sorted(object_rows, key=object_metric)
        photo_ordered = sorted(photo_rows, key=photo_metric)
        object_best = object_ordered[0] if object_ordered else None
        photo_best = photo_ordered[0] if photo_ordered else None
        object_margin = (object_metric(object_ordered[1]) - object_metric(object_ordered[0])
                         if len(object_ordered) > 1 else None)
        photo_margin = (photo_metric(photo_ordered[1]) - photo_metric(photo_ordered[0])
                        if len(photo_ordered) > 1 else None)
        combined = rows[0] if rows else None
        details[name] = {"combined_best": combined, "object_best": object_best,
                         "object_second_best_margin": object_margin, "photometric_best": photo_best,
                         "photometric_second_best_margin": photo_margin}
        if combined:
            conventions.add(candidate_key(combined))
        if (object_best is None or photo_best is None or candidate_key(object_best) != candidate_key(photo_best)
                or object_margin is None or object_margin <= 1e-9
                or photo_margin is None or photo_margin <= 1e-9):
            confirmed = False
    confirmed &= len(by_recording) >= 2 and len(details) == len(by_recording) and len(conventions) == 1
    key = next(iter(conventions)) if len(conventions) == 1 else (None, None, None, None)
    return {"status": "confirmed" if confirmed else "ambiguous", "scale": key[0], "sign": key[1],
            "time_pair": key[2], "background_treatment": key[3], "per_recording": details,
            "criterion": "object and photometric winners agree with positive (>1e-9) second-best margins within every recording, and at least two recordings agree",
            "warp_equation": WARP_EQUATION,
            "limitation": "forward splatting creates holes; MAE uses only covered target pixels and object-center EPE is primary physical evidence"}


def _matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "eventdelta-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def create_event_preview(recording: str, samples: list[dict[str, Any]], best: dict[str, Any], destination: Path) -> None:
    plt = _matplotlib()
    selected = [samples[index] for index in sorted(set((0, len(samples) // 2, len(samples) - 1)))]
    fig, axes = plt.subplots(len(selected), 4, figsize=(14, 3.5 * len(selected)), squeeze=False)
    temporal = best.get("temporal_alignment") or "previous_to_current"
    transform = best.get("spatial_transform") or "identity"
    for row, sample in enumerate(selected):
        source, target = sample["images"][temporal]
        event = transform_map(sample["event"], transform)
        panels = (source, target, event, np.mean(np.abs(target - source), axis=-1))
        titles = (f"source RGB ({temporal})", "target RGB", f"Event {transform}", "RGB absolute difference")
        for axis, image, title in zip(axes[row], panels, titles):
            axis.imshow(image, cmap="magma" if image.ndim == 2 else None)
            axis.set_title(f"{title}; frame={sample['frame_id']}")
            axis.axis("off")
    fig.suptitle(f"Event candidate inspection: {recording}; status={best.get('status', 'ambiguous')}")
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def draw_rgb_boxes(image: np.ndarray, objects: dict[str, tuple[float, float, float, float]]) -> np.ndarray:
    canvas = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    for object_id, box in objects.items():
        draw.rectangle(box, outline=(255, 32, 32), width=2)
        draw.text((box[0], max(0, box[1] - 10)), object_id, fill=(255, 32, 32))
    return np.asarray(canvas)


def robust_limit(array: np.ndarray) -> float:
    finite = np.asarray(array)[np.isfinite(array)]
    active = finite[np.abs(finite) > 1e-12]
    return max(float(np.percentile(np.abs(active), 99)) if active.size else 0.0, 1e-12)


def create_flow_preview(recording: str, samples: list[dict[str, Any]], best: dict[str, Any], destination: Path) -> None:
    plt = _matplotlib()
    selected = [samples[index] for index in sorted(set((0, len(samples) // 2, len(samples) - 1)))]
    fig, axes = plt.subplots(len(selected), 6, figsize=(22, 3.5 * len(selected)), squeeze=False)
    candidate = best.get("per_recording", {}).get(recording, {}).get("combined_best") or {}
    scale, sign = candidate.get("scale", "raw"), candidate.get("sign", 1)
    treatment = candidate.get("background_treatment", "raw")
    for row, sample in enumerate(selected):
        flow, bg = sample["flow"], sample["background"]
        raw_mag = np.linalg.norm(flow, axis=-1)
        residual_mag = np.linalg.norm(flow - bg, axis=-1)
        pixel_mag = np.linalg.norm(candidate_displacement(flow, scale, sign, treatment, bg), axis=-1)
        panels = (draw_rgb_boxes(sample["all_images"]["current"], sample["objects"]["current"]),
                  flow[..., 0], flow[..., 1], raw_mag, residual_mag, pixel_mag)
        titles = ("RGB + bbox", "raw u", "raw v", "raw magnitude",
                  f"magnitude after bg subtraction\nbg=({bg[0]:.6g},{bg[1]:.6g})", f"best pixel displacement\n{scale}, sign={sign}")
        units = (None, "raw", "raw", "raw", "raw", "pixel")
        for column, (axis, image, title, unit) in enumerate(zip(axes[row], panels, titles, units)):
            if column == 0:
                axis.imshow(image)
            else:
                if column in (1, 2):
                    limit = robust_limit(image)
                    plotted = axis.imshow(image, cmap="coolwarm", vmin=-limit, vmax=limit)
                else:
                    plotted = axis.imshow(image, cmap="magma", vmin=0, vmax=robust_limit(image))
                colorbar = fig.colorbar(plotted, ax=axis, fraction=0.046, pad=0.04)
                colorbar.set_label(unit)
            axis.set_title(f"{title}; frame={sample['frame_id']}")
            axis.axis("off")
    fig.suptitle(f"Flow calibration: {recording}; robust P99 of nonzero/finite values")
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def create_warp_preview(recording: str, sample: dict[str, Any], best: dict[str, Any], destination: Path) -> None:
    plt = _matplotlib()
    candidate = best.get("per_recording", {}).get(recording, {}).get("combined_best") or {}
    time_pair = candidate.get("time_pair", "previous_to_current")
    displacement = photometric_displacement(sample["photo_flow"], sample["flow"].shape[:2],
                                            candidate.get("scale", "raw"), candidate.get("sign", 1),
                                            candidate.get("background_treatment", "raw"), sample["background"])
    source, target = sample["photo_images"][time_pair]
    photo = photometric_metrics(source, target, displacement, sample["photo_motion_masks"][time_pair])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = (source, target, photo["warped"], photo["residual"])
    titles = ("source", "target", "forward-splatted source", f"absolute residual; MAE={photo['mae']}")
    for index, (axis, image, title) in enumerate(zip(axes, panels, titles)):
        shown = axis.imshow(image, cmap="magma" if image.ndim == 2 else None)
        if index == 3:
            fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04).set_label("normalized RGB absolute error")
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{recording}; frame={sample['frame_id']}; {time_pair}\n{WARP_EQUATION}", fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_text(report: dict[str, Any]) -> str:
    event, flow = report["event_best_candidate"], report["flow_best_candidate"]
    enough = event["status"] == "confirmed" and flow["status"] == "confirmed"
    return f"""# SEVD R1.2 calibration summary

- Event status: **{event['status']}**. Most likely interval: `{event.get('temporal_alignment')}`; spatial transform: `{event.get('spatial_transform')}`.
- Flow status: **{flow['status']}**. Scale: `{flow.get('scale')}`; sign: `{flow.get('sign')}`; interval: `{flow.get('time_pair')}`; background: `{flow.get('background_treatment')}`.
- Timestamp storage-unit inference: approximately nanoseconds when the reported numeric delta agrees with `metadata.delta_seconds`; this does **not** imply 1 GHz perception. Event packets represent about 100 ms / 10 Hz.
- Confirmed here: only candidate conventions satisfying the recorded agreement and margin criteria. All candidate metrics remain in CSV/JSON.
- Not confirmed here: the Event-delta research hypothesis, vision-encoder savings, or online model quality. Flow/KITTI evidence is oracle calibration evidence.
- R2 readiness: **{'yes for convention-dependent follow-up' if enough else 'no; resolve ambiguous convention evidence first'}**.
- Planned experiment timing remains 1 Hz RGB anchors plus 10 Hz Event deltas; intermediate RGB is offline ground truth only.

## Known limitations

- Photometric flow uses forward bilinear splatting and scores only covered target pixels; disocclusion, occlusion, lighting, and interpolation holes can disagree with object-center EPE.
- KITTI center displacement is a coarse proxy for object motion and depends on matched object IDs and sufficiently large boxes.
- Overall flow quantiles use deterministic stride samples; per-frame min/max and counts inspect every pixel.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--recording", required=True, action="append")
    parser.add_argument("--samples", type=int, default=120, help="total samples, balanced across recordings")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not root.is_dir() or args.samples <= 0:
        print("error: --root must be a directory and --samples must be positive", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    allocations = distribute_samples(args.samples, args.recording)
    reports, all_samples = {}, {}
    event_by_recording, flow_by_recording, flow_stats = {}, {}, {}
    raw_event_rows: dict[str, list[dict[str, Any]]] = {}
    timestamp_inferences, skipped = {}, {}
    try:
        for recording_index, recording in enumerate(args.recording):
            audit, state = audit_recording(root, recording, args.metadata_path)
            reports[recording] = audit
            eligible, initial_skips = eligible_frame_ids(state)
            chosen = stratified_sample(eligible, allocations[recording], args.seed + recording_index)
            samples, runtime_skips = [], []
            for identifier in chosen:
                try:
                    samples.append(make_sample(state, identifier))
                except (OSError, ValueError, KeyError) as exc:
                    runtime_skips.append({"frame_id": identifier, "reasons": [f"{type(exc).__name__}: {exc}"]})
            if not samples:
                raise RuntimeError(f"recording {recording} has no usable calibration samples")
            all_samples[recording] = samples
            sample_event_rows = [row for sample in samples for row in sample["event_scores"]]
            raw_event_rows[recording] = sample_event_rows
            event_by_recording[recording] = aggregate_event_scores(sample_event_rows, recording)
            flow_by_recording[recording] = score_flow_candidates(samples, recording)
            flow_stats[recording] = aggregate_flow_statistics([(sample["frame_id"], sample["flow"]) for sample in samples])
            skipped[recording] = initial_skips + runtime_skips
            timestamp_inferences[recording] = audit["event_audit"]["timestamp_unit"]
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    event_overall = aggregate_event_scores([row for rows in raw_event_rows.values() for row in rows], "overall")
    flow_overall = score_flow_candidates([sample for samples in all_samples.values() for sample in samples], "overall")
    flow_stats_overall = aggregate_flow_statistics(
        [(sample["frame_id"], sample["flow"]) for samples in all_samples.values() for sample in samples])
    event_best = best_and_status_event(event_by_recording)
    flow_best = best_and_status_flow(flow_by_recording)
    event_rows = [row for rows in event_by_recording.values() for row in rows] + event_overall
    flow_rows = [row for rows in flow_by_recording.values() for row in rows] + flow_overall
    write_csv(output_dir / "event_candidate_scores.csv", event_rows)
    write_csv(output_dir / "flow_candidate_scores.csv", flow_rows)
    generated = ["calibration.json", "event_candidate_scores.csv", "flow_candidate_scores.csv", "summary.md"]
    for recording in args.recording:
        event_path = output_dir / f"event_alignment_{recording}.png"
        flow_path = output_dir / f"flow_calibration_{recording}.png"
        warp_path = output_dir / f"best_warp_{recording}.png"
        per_event = dict(event_best)
        per_event.update(event_best["per_recording_best"].get(recording, {}))
        create_event_preview(recording, all_samples[recording], per_event, event_path)
        create_flow_preview(recording, all_samples[recording], flow_best, flow_path)
        create_warp_preview(recording, all_samples[recording][len(all_samples[recording]) // 2], flow_best, warp_path)
        generated.extend((event_path.name, flow_path.name, warp_path.name))
    metadata = next((value.get("metadata") for value in reports.values() if value.get("metadata")), None)
    report = {
        "scope": "SEVD R1.2 temporal/spatial convention calibration; CPU; read-only dataset access",
        "dataset_root": str(root), "recordings": args.recording, "metadata": metadata,
        "sampling": {"seed": args.seed, "requested_total": args.samples, "allocation": allocations,
                     "frame_ids": {name: [sample["frame_id"] for sample in samples]
                                   for name, samples in all_samples.items()}, "skipped": skipped},
        "event_candidate_scores": {"by_recording": event_by_recording, "overall": event_overall},
        "event_best_candidate": event_best,
        "timestamp_unit_inference": timestamp_inferences,
        "timestamp_note": "integer unit approximately ns is a storage-scale inference, not 1 GHz perception; packets are about 100 ms / 10 Hz",
        "flow_raw_statistics": {"by_recording": flow_stats, "overall": flow_stats_overall},
        "flow_background_vector": {
            "by_recording": {name: stats["overall"]["background_vector"] for name, stats in flow_stats.items()},
            "overall": flow_stats_overall["overall"]["background_vector"],
        },
        "flow_candidate_scores": {"by_recording": flow_by_recording, "overall": flow_overall},
        "flow_best_candidate": flow_best,
        "warp_equation": WARP_EQUATION,
        "unresolved_assumptions": [
            "Event and flow statuses remain ambiguous unless their explicit evidence criteria pass",
            "KITTI boxes/IDs and flow are oracle calibration signals, not causal query inputs",
            "forward-splat photometric scores exclude uncovered target pixels and cannot resolve occlusion",
        ],
        "tool": {"name": "tools/calibrate_sevd.py", "version": VERSION, "python": platform.python_version(),
                 "numpy": np.__version__, "Pillow": Image.__version__,
                 "photometric_max_dimension": 320},
        "generated_files": generated,
    }
    try:
        import matplotlib
        report["tool"]["matplotlib"] = matplotlib.__version__
    except ImportError:
        pass
    (output_dir / "summary.md").write_text(summary_text(report), encoding="utf-8")
    (output_dir / "calibration.json").write_text(json.dumps(report, default=json_value, indent=2,
                                                              ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
