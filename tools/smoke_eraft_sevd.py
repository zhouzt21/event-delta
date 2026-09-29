#!/usr/bin/env python3
"""Run the frozen R3a single-transition, flow-only E-RAFT smoke test."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from event_delta.flow_warp import sevd_flow_to_pixel_displacement  # noqa: E402
from event_delta.json_io import atomic_write_json, json_dumps  # noqa: E402
from tools.audit_sevd import find_recording, frame_id  # noqa: E402
from tools.calibrate_sevd import read_flow  # noqa: E402


ERAFT_COMMIT = "c58ce0524ea0ebfa9849991caafb547f44fe9bfd"
NATIVE_HW = (960, 1280)
MODEL_HW = (480, 640)
MOTION_THRESHOLD_PIXELS = 1.0
POLARITY_MIN_EVENTS = 100
POLARITY_MIN_ABS_CORRELATION = 0.05
PIL_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
NUMPY_LEGACY_ALIASES = ("np.float", "np.int", "np.bool", "numpy.float", "numpy.int", "numpy.bool")


def index_files(directory: Path, suffix: str, filename_suffix: str | None = None) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required directory does not exist: {directory}")
    indexed: dict[int, Path] = {}
    duplicates: list[int] = []
    for path in sorted(directory.iterdir()):
        if (not path.is_file() or path.suffix.lower() != suffix
                or (filename_suffix is not None and not path.name.lower().endswith(filename_suffix))):
            continue
        identifier = frame_id(path)
        if identifier is None:
            continue
        if identifier in indexed:
            duplicates.append(identifier)
        indexed[identifier] = path.resolve()
    if duplicates:
        raise ValueError(f"duplicate frame IDs in {directory}: {sorted(set(duplicates))[:10]}")
    return indexed


def resolve_transition_paths(root: Path, recording: str, source_frame_id: int) -> dict[str, Path]:
    """Resolve Event_i, Event_i+1, flow_i, RGB_i, and RGB_i+1 uniquely."""
    recording_root = find_recording(root.resolve(), recording).resolve()
    events = index_files(recording_root / "dvs_camera", ".npz", "-xytp.npz")
    flows = index_files(recording_root / "optical_flow_camera", ".npz")
    rgbs = index_files(recording_root / "rgb_camera", ".png")
    requested = {
        "old_event": (events, source_frame_id),
        "new_event": (events, source_frame_id + 1),
        "gt_flow": (flows, source_frame_id),
        "source_rgb": (rgbs, source_frame_id),
        "target_rgb": (rgbs, source_frame_id + 1),
    }
    missing = [f"{name} frame {identifier}" for name, (index, identifier) in requested.items()
               if identifier not in index]
    if missing:
        raise FileNotFoundError("missing transition inputs: " + ", ".join(missing))
    return {"recording_root": recording_root,
            **{name: index[identifier] for name, (index, identifier) in requested.items()}}


def read_sevd_events(path: Path) -> dict[str, np.ndarray]:
    """Read only the confirmed SEVD ``dvs_events[x,y,t,pol]`` schema."""
    with np.load(path, allow_pickle=False) as archive:
        if "dvs_events" not in archive.files:
            raise ValueError(f"Event NPZ must contain key 'dvs_events': {path}")
        structured = np.asarray(archive["dvs_events"])
    fields = structured.dtype.names
    if fields is None or not {"x", "y", "t", "pol"}.issubset(fields):
        raise ValueError(f"dvs_events must be structured with fields x/y/t/pol: {path}; fields={fields}")
    return {name: np.asarray(structured[name]).reshape(-1) for name in ("x", "y", "t", "pol")}


def validate_events(events: dict[str, np.ndarray], path: Path,
                    source_hw: tuple[int, int] = NATIVE_HW) -> dict[str, Any]:
    missing = sorted({"x", "y", "t", "pol"} - set(events))
    if missing:
        raise ValueError(f"Event packet is missing confirmed fields {missing}: {path}")
    lengths = {name: np.asarray(events[name]).reshape(-1).size for name in ("x", "y", "t", "pol")}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Event fields have different lengths in {path}: {lengths}")
    count = lengths["t"]
    if count == 0:
        raise ValueError(f"empty Event packet: {path}")
    timestamps = np.asarray(events["t"]).reshape(-1)
    if not np.issubdtype(timestamps.dtype, np.integer):
        raise ValueError(f"Event timestamps must be integer before subtraction in {path}, got {timestamps.dtype}")
    timestamps = timestamps.astype(np.int64, copy=False)
    if np.any(np.diff(timestamps) < 0):
        raise ValueError(f"Event timestamps are not monotonic nondecreasing: {path}")
    if timestamps[-1] == timestamps[0]:
        raise ValueError(f"Event packet has fewer than two distinct timestamps: {path}")
    x = np.asarray(events["x"]).reshape(-1)
    y = np.asarray(events["y"]).reshape(-1)
    if not np.issubdtype(x.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"Event x/y must be integer fields: {path}; x={x.dtype}, y={y.dtype}")
    source_height, source_width = source_hw
    if np.any((x < 0) | (x >= source_width)) or np.any((y < 0) | (y >= source_height)):
        raise ValueError(f"Event coordinates exceed source bounds {source_width}x{source_height}: {path}; "
                         f"x=[{x.min()},{x.max()}], y=[{y.min()},{y.max()}]")
    raw_polarity = np.asarray(events["pol"]).reshape(-1)
    if raw_polarity.dtype != np.bool_:
        raise ValueError(f"Event pol must use the confirmed bool dtype: {path}; got {raw_polarity.dtype}")
    polarity = raw_polarity.astype(bool, copy=False)
    return {
        "event_count": count,
        "fields": {name: str(np.asarray(events[name]).dtype) for name in ("x", "y", "t", "pol")},
        "x_min": int(x.min()),
        "x_max": int(x.max()),
        "y_min": int(y.min()),
        "y_max": int(y.max()),
        "timestamp_first": int(timestamps[0]),
        "timestamp_last": int(timestamps[-1]),
        "timestamp_span": int(timestamps[-1] - timestamps[0]),
        "positive_event_count": int(np.count_nonzero(polarity)),
        "negative_event_count": int(count - np.count_nonzero(polarity)),
    }


def voxelize_events(events: dict[str, np.ndarray], bins: int = 15,
                    input_hw: tuple[int, int] = MODEL_HW,
                    source_hw: tuple[int, int] = NATIVE_HW) -> tuple[np.ndarray, dict[str, Any]]:
    """Match upstream E-RAFT spatial/temporal trilinear splatting and nonzero normalization."""
    if bins != 15:
        raise ValueError(f"R3a smoke requires exactly 15 bins, got {bins}")
    height, width = input_hw
    source_height, source_width = source_hw
    if height <= 0 or width <= 0 or source_height <= 0 or source_width <= 0:
        raise ValueError("voxel dimensions must be positive")

    timestamps_i64 = np.asarray(events["t"]).reshape(-1).astype(np.int64, copy=False)
    if timestamps_i64.size == 0:
        raise ValueError("cannot voxelize an empty Event packet")
    relative_t = timestamps_i64 - timestamps_i64[0]  # Integer subtraction is intentional.
    span = int(relative_t[-1])
    if span <= 0 or np.unique(timestamps_i64).size < 2:
        raise ValueError("Event packet must contain at least two distinct increasing timestamps")
    t = relative_t.astype(np.float64) * ((bins - 1) / span)
    x = np.asarray(events["x"]).reshape(-1).astype(np.float64) * (width / source_width)
    y = np.asarray(events["y"]).reshape(-1).astype(np.float64) * (height / source_height)
    polarity = np.asarray(events["pol"]).reshape(-1).astype(bool, copy=False)
    if not (x.size == y.size == t.size == polarity.size):
        raise ValueError("Event x/y/t/pol fields must have equal lengths")
    values = np.where(polarity, 1.0, -1.0)

    voxel = np.zeros((bins, height, width), dtype=np.float32)
    x0, y0, t0 = np.trunc(x).astype(np.int64), np.trunc(y).astype(np.int64), np.trunc(t).astype(np.int64)
    for xi in (x0, x0 + 1):
        for yi in (y0, y0 + 1):
            for ti in (t0, t0 + 1):
                valid = ((xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
                         & (ti >= 0) & (ti < bins))
                weights = values * (1.0 - np.abs(xi - x)) * (1.0 - np.abs(yi - y)) * (1.0 - np.abs(ti - t))
                np.add.at(voxel, (ti[valid], yi[valid], xi[valid]), weights[valid].astype(np.float32))

    nonzero_before = voxel != 0
    count_before = int(np.count_nonzero(nonzero_before))
    if count_before:
        selected = voxel[nonzero_before]
        mean = np.float32(selected.mean(dtype=np.float32))
        # torch.std() at the audited commit uses the sample (unbiased) standard deviation.
        std = np.float32(selected.std(ddof=1, dtype=np.float32)) if selected.size > 1 else np.float32(np.nan)
        if np.isfinite(std) and std > 0:
            voxel[nonzero_before] = (selected - mean) / std
        else:
            voxel[nonzero_before] = selected - mean
    nonzero_after = int(np.count_nonzero(voxel))
    stats = {
        "voxel_shape": [1, bins, height, width],
        "voxel_dtype": str(voxel.dtype),
        "voxel_finite": bool(np.isfinite(voxel).all()),
        "coordinate_scale_xy": [width / source_width, height / source_height],
        "coordinate_mapping": "x->width, y->height; no swap or flip",
        "polarity_mapping": {"False": -1.0, "True": 1.0},
        "event_count": int(t.size),
        "timestamp_span": span,
        "positive_event_count": int(np.count_nonzero(polarity)),
        "negative_event_count": int(polarity.size - np.count_nonzero(polarity)),
        "nonzero_voxel_count_before_normalization": count_before,
        "nonzero_voxel_ratio_before_normalization": count_before / voxel.size,
        "nonzero_voxel_count": nonzero_after,
        "nonzero_voxel_ratio": nonzero_after / voxel.size,
    }
    return voxel[np.newaxis].astype(np.float32, copy=False), stats


def normalize_state_dict_prefix(state: Mapping[str, Any], expected_keys: set[str]) -> dict[str, Any]:
    """Accept exact keys or the documented all-keys DataParallel ``module.`` prefix only."""
    keys = set(state)
    if keys == expected_keys:
        return dict(state)
    prefixed = [key.startswith("module.") for key in state]
    if prefixed and all(prefixed):
        stripped = {key[len("module."):]: value for key, value in state.items()}
        if set(stripped) == expected_keys:
            return stripped
    if any(prefixed) and not all(prefixed):
        raise ValueError("checkpoint state dict mixes prefixed and unprefixed keys")
    missing = sorted(expected_keys - keys)[:10]
    unexpected = sorted(keys - expected_keys)[:10]
    raise ValueError(f"checkpoint keys do not strictly match model; missing={missing}, unexpected={unexpected}")


def checkpoint_prefix_action(state: Mapping[str, Any], expected_keys: set[str]) -> str:
    keys = set(state)
    if keys == expected_keys:
        return "none"
    if keys and all(key.startswith("module.") for key in keys):
        stripped = {key[len("module."):] for key in keys}
        if stripped == expected_keys:
            return "stripped_uniform_module_prefix"
    return "incompatible"


def require_prediction(prediction: Any, expected: tuple[int, ...] = (1, 2, 480, 640)) -> Any:
    shape = tuple(prediction.shape)
    if shape != expected:
        raise ValueError(f"E-RAFT output must have shape {expected}, got {shape}")
    finite = (np.isfinite(prediction).all() if isinstance(prediction, np.ndarray)
              else prediction.isfinite().all().item())
    if not bool(finite):
        raise ValueError("E-RAFT output contains NaN or Inf")
    return prediction


def resize_prediction_native(prediction: np.ndarray,
                             native_hw: tuple[int, int] = NATIVE_HW) -> np.ndarray:
    """Resize [1,2,H,W] pixel flow and convert vector units to native pixels."""
    array = np.asarray(prediction, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 2:
        raise ValueError(f"expected prediction [1,2,H,W], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("prediction contains NaN or Inf")
    source_height, source_width = array.shape[2:]
    target_height, target_width = native_hw
    channels = []
    for channel in range(2):
        image = Image.fromarray(array[0, channel], mode="F")
        channels.append(np.asarray(image.resize((target_width, target_height), PIL_BILINEAR),
                                   dtype=np.float32))
    output = np.stack(channels, axis=-1)
    output[..., 0] *= target_width / source_width
    output[..., 1] *= target_height / source_height
    if output.shape != (target_height, target_width, 2) or not np.isfinite(output).all():
        raise ValueError(f"invalid native prediction shape/values: {output.shape}")
    return output.astype(np.float32, copy=False)


def flow_metrics(prediction: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(prediction).all(axis=-1) & np.isfinite(ground_truth).all(axis=-1)
    count = int(np.count_nonzero(valid))
    if not count:
        return {"pixel_count": 0, "epe": None, "median_epe": None, "angular_error_degrees": None,
                "endpoint_cosine": None, "directional_pixel_count": 0,
                "predicted_magnitude_mean": None, "gt_magnitude_mean": None,
                "predicted_to_gt_magnitude_ratio": None, "predicted_motion_coverage": None,
                "gt_motion_recall": None, "1PE": None, "2PE": None, "3PE": None}
    pred = prediction[valid].astype(np.float64)
    truth = ground_truth[valid].astype(np.float64)
    epe = np.linalg.norm(pred - truth, axis=-1)
    pred_norm, truth_norm = np.linalg.norm(pred, axis=-1), np.linalg.norm(truth, axis=-1)
    directional = (pred_norm > 1e-8) & (truth_norm > 1e-8)
    cosine = (np.sum(pred[directional] * truth[directional], axis=-1)
              / (pred_norm[directional] * truth_norm[directional])) if np.any(directional) else np.empty(0)
    cosine = np.clip(cosine, -1, 1)
    predicted_motion = pred_norm >= MOTION_THRESHOLD_PIXELS
    gt_motion = truth_norm >= MOTION_THRESHOLD_PIXELS
    gt_mean = float(np.mean(truth_norm))
    return {
        "pixel_count": count,
        "epe": float(np.mean(epe)),
        "median_epe": float(np.median(epe)),
        "angular_error_degrees": float(np.mean(np.degrees(np.arccos(cosine)))) if cosine.size else None,
        "endpoint_cosine": float(np.mean(cosine)) if cosine.size else None,
        "directional_pixel_count": int(cosine.size),
        "predicted_magnitude_mean": float(np.mean(pred_norm)),
        "gt_magnitude_mean": gt_mean,
        "predicted_to_gt_magnitude_ratio": float(np.mean(pred_norm) / gt_mean) if gt_mean > 1e-8 else None,
        "predicted_motion_coverage": float(np.mean(predicted_motion)),
        "gt_motion_recall": float(np.mean(predicted_motion[gt_motion])) if np.any(gt_motion) else None,
        "1PE": float(np.mean(epe > 1.0)),
        "2PE": float(np.mean(epe > 2.0)),
        "3PE": float(np.mean(epe > 3.0)),
    }


def prediction_statistics(prediction: np.ndarray, tensor_dtype: str | None = None,
                          tensor_device: str | None = None) -> dict[str, Any]:
    array = np.asarray(prediction)
    magnitude = np.linalg.norm(array.astype(np.float64), axis=-1)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "model_tensor_dtype": tensor_dtype,
        "model_tensor_device": tensor_device,
        "finite": bool(np.isfinite(array).all()),
        "nonzero_vector_ratio": float(np.mean(magnitude > 1e-6)),
        "dx": {"min": float(np.min(array[..., 0])), "mean": float(np.mean(array[..., 0])),
               "max": float(np.max(array[..., 0])), "std": float(np.std(array[..., 0]))},
        "dy": {"min": float(np.min(array[..., 1])), "mean": float(np.mean(array[..., 1])),
               "max": float(np.max(array[..., 1])), "std": float(np.std(array[..., 1]))},
        "magnitude": {"min": float(np.min(magnitude)), "mean": float(np.mean(magnitude)),
                      "max": float(np.max(magnitude)), "p99": float(np.percentile(magnitude, 99))},
    }


def signed_metric_report(prediction: np.ndarray, ground_truth: np.ndarray,
                         motion_mask: np.ndarray) -> dict[str, Any]:
    all_mask = np.ones(ground_truth.shape[:2], dtype=bool)
    return {
        "positive_prediction": {
            "all_pixels": flow_metrics(prediction, ground_truth, all_mask),
            "gt_motion_region": flow_metrics(prediction, ground_truth, motion_mask),
        },
        "negative_prediction": {
            "all_pixels": flow_metrics(-prediction, ground_truth, all_mask),
            "gt_motion_region": flow_metrics(-prediction, ground_truth, motion_mask),
        },
    }


def smoke_decision(correct: dict[str, Any], swapped: dict[str, Any],
                   statistics: dict[str, Any], clear_margin: float = 0.05) -> dict[str, Any]:
    """Conservative interface diagnostic; never changes the formal prediction."""
    scope = ("gt_motion_region"
             if correct["positive_prediction"]["gt_motion_region"]["pixel_count"]
             else "all_pixels")
    positive = correct["positive_prediction"][scope]
    negative = correct["negative_prediction"][scope]
    swapped_positive = swapped["positive_prediction"][scope]
    reasons = []
    if not statistics["finite"]:
        reasons.append("prediction_non_finite")
    if statistics["nonzero_vector_ratio"] == 0:
        reasons.append("prediction_all_zero")
    positive_epe = positive["epe"]
    negative_epe = negative["epe"]
    swapped_epe = swapped_positive["epe"]
    if positive_epe is None or negative_epe is None or swapped_epe is None:
        reasons.append("insufficient_valid_pixels")
    else:
        required_factor = 1.0 - clear_margin
        if not positive_epe < negative_epe * required_factor:
            reasons.append("positive_sign_not_clearly_better_than_negative")
        if not positive_epe < swapped_epe * required_factor:
            reasons.append("correct_packet_order_not_clearly_better_than_swapped")
    ratio = positive.get("predicted_to_gt_magnitude_ratio")
    if ratio is not None and (ratio < 0.01 or ratio > 100.0):
        reasons.append("prediction_scale_is_obviously_incompatible_with_gt")
    return {
        "status": "passed" if not reasons else "blocked",
        "diagnostic_scope": scope,
        "clear_improvement_margin": clear_margin,
        "reasons": reasons,
        "formal_prediction_changed": False,
        "positive_epe": positive_epe,
        "negative_epe": negative_epe,
        "swapped_positive_epe": swapped_epe,
    }


def polarity_sanity(events: dict[str, np.ndarray], source_rgb: np.ndarray,
                    target_rgb: np.ndarray) -> dict[str, Any]:
    if source_rgb.shape != target_rgb.shape or source_rgb.ndim != 3:
        raise ValueError(f"RGB pair shape mismatch: {source_rgb.shape} vs {target_rgb.shape}")
    gray_weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float64)
    intensity_change = (target_rgb.astype(np.float64) - source_rgb.astype(np.float64)) @ gray_weights
    x = np.asarray(events["x"]).reshape(-1).astype(np.int64)
    y = np.asarray(events["y"]).reshape(-1).astype(np.int64)
    signs = np.where(np.asarray(events["pol"]).reshape(-1).astype(bool), 1.0, -1.0)
    height, width = intensity_change.shape
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    samples = intensity_change[y[valid], x[valid]]
    signs = signs[valid]
    correlation = None
    if samples.size >= 2 and np.std(samples) > 0 and np.std(signs) > 0:
        correlation = float(np.corrcoef(signs, samples)[0, 1])
    positive = samples[signs > 0]
    negative = samples[signs < 0]
    reliable = (samples.size >= POLARITY_MIN_EVENTS and correlation is not None
                and abs(correlation) >= POLARITY_MIN_ABS_CORRELATION
                and positive.size > 0 and negative.size > 0)
    if not reliable:
        status = "unconfirmed"
    elif correlation > 0:
        status = "supports_original"
    else:
        status = "supports_inverted"
    return {
        "status": status,
        "selection_effect": "diagnostic_only; inference always uses False=-1 and True=+1",
        "valid_event_count": int(samples.size),
        "reliability_rule": {"minimum_events": POLARITY_MIN_EVENTS,
                             "minimum_absolute_correlation": POLARITY_MIN_ABS_CORRELATION,
                             "requires_both_polarities": True},
        "original_polarity": {
            "correlation_with_rgb_intensity_change": correlation,
            "positive_event_mean_intensity_change": float(np.mean(positive)) if positive.size else None,
            "negative_event_mean_intensity_change": float(np.mean(negative)) if negative.size else None,
        },
        "inverted_polarity": {
            "correlation_with_rgb_intensity_change": -correlation if correlation is not None else None,
            "positive_event_mean_intensity_change": float(np.mean(negative)) if negative.size else None,
            "negative_event_mean_intensity_change": float(np.mean(positive)) if positive.size else None,
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                             capture_output=True, check=False)
    if process.returncode or not process.stdout.strip():
        raise RuntimeError(f"cannot resolve git HEAD for {root}: {process.stderr.strip()}")
    return process.stdout.strip()


def select_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def numpy_compatibility_error(exc: BaseException) -> RuntimeError | None:
    text = str(exc)
    if any(alias in text for alias in NUMPY_LEGACY_ALIASES) or "has no attribute 'float'" in text \
            or "has no attribute 'int'" in text or "has no attribute 'bool'" in text:
        return RuntimeError(
            "E-RAFT import failed because third-party code uses a removed NumPy alias. "
            "Patch that external checkout locally (np.float->float/np.float64, np.int->int/np.int64, "
            "np.bool->bool/np.bool_) and record the patch; EventDelta does not monkey-patch NumPy globally. "
            f"Original error: {type(exc).__name__}: {exc}"
        )
    return None


def load_model(torch: Any, eraft_root: Path, checkpoint_path: Path, bins: int,
               mode: str, device: Any) -> tuple[Any, dict[str, Any]]:
    revision = git_head(eraft_root)
    if revision != ERAFT_COMMIT:
        raise RuntimeError(f"E-RAFT HEAD mismatch: expected {ERAFT_COMMIT}, got {revision}")
    if str(eraft_root) not in sys.path:
        sys.path.insert(0, str(eraft_root))
    try:
        module = importlib.import_module("model.eraft")
    except (AttributeError, ImportError) as exc:
        compatibility = numpy_compatibility_error(exc)
        if compatibility is not None:
            raise compatibility from exc
        raise
    module_path = Path(module.__file__).resolve()
    if eraft_root not in module_path.parents:
        raise ImportError(f"resolved model.eraft outside --eraft-root: {module_path}")
    model = module.ERAFT(config={"subtype": mode}, n_first_channels=bins)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError("checkpoint must be a mapping containing key 'model'")
    state = checkpoint["model"]
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint['model'] must be a state-dict mapping")
    expected_keys = set(model.state_dict())
    prefix_action = checkpoint_prefix_action(state, expected_keys)
    normalized = normalize_state_dict_prefix(state, expected_keys)
    model.load_state_dict(normalized, strict=True)
    metadata = {
        "strict_load": True,
        "state_dict_key_count": len(state),
        "module_prefix_action": prefix_action,
        "external_module_path": str(module_path),
        "numpy_compatibility_adapter": "none",
        "numpy_global_monkey_patch": False,
    }
    return model.to(device).eval(), metadata


def timed_forward(torch: Any, model: Any, old: Any, new: Any, iterations: int,
                  device: Any) -> tuple[Any, dict[str, Any]]:
    kwargs = {"image1": old, "image2": new, "iters": iterations, "flow_init": None}
    with torch.inference_mode():
        model(**kwargs)  # One explicit warmup for this ordering.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            _low, predictions = model(**kwargs)
            end.record()
            torch.cuda.synchronize(device)
            seconds = float(start.elapsed_time(end) / 1000.0)
            timing_method = "cuda_events_after_one_warmup"
        else:
            start_time = time.perf_counter()
            _low, predictions = model(**kwargs)
            seconds = time.perf_counter() - start_time
            timing_method = "perf_counter_after_one_warmup"
    if not isinstance(predictions, (list, tuple)) or len(predictions) != iterations:
        raise ValueError(f"E-RAFT must return {iterations} full-resolution predictions")
    final = require_prediction(predictions[-1])
    return final, {"seconds": seconds, "iterations": iterations, "warmup_forwards": 1,
                   "timing_method": timing_method}


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def flow_color(flow: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(flow.astype(np.float64), axis=-1)
    finite = magnitude[np.isfinite(magnitude)]
    scale = float(np.percentile(finite, 99)) if finite.size else 0.0
    angle = (np.arctan2(flow[..., 1], flow[..., 0]) + np.pi) / (2 * np.pi)
    saturation = np.clip(magnitude / scale, 0, 1) if scale > 0 else np.zeros_like(magnitude)
    # Compact HSV(value=1) conversion.
    h6 = angle * 6
    sector = np.floor(h6).astype(np.int64) % 6
    fraction = h6 - np.floor(h6)
    p, q, t = 1 - saturation, 1 - saturation * fraction, 1 - saturation * (1 - fraction)
    choices = ((np.ones_like(p), t, p), (q, np.ones_like(p), p), (p, np.ones_like(p), t),
               (p, q, np.ones_like(p)), (t, p, np.ones_like(p)), (np.ones_like(p), p, q))
    rgb = np.zeros((*magnitude.shape, 3), dtype=np.float64)
    for index, channels in enumerate(choices):
        mask = sector == index
        for channel, values in enumerate(channels):
            rgb[..., channel][mask] = values[mask]
    rgb[~np.isfinite(flow).all(axis=-1)] = 0
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def event_visualization(events: dict[str, np.ndarray], shape: tuple[int, int] = NATIVE_HW) -> np.ndarray:
    height, width = shape
    x = np.asarray(events["x"]).reshape(-1).astype(np.int64)
    y = np.asarray(events["y"]).reshape(-1).astype(np.int64)
    polarity = np.asarray(events["pol"]).reshape(-1).astype(bool)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    positive = np.zeros(shape, dtype=np.float32)
    negative = np.zeros(shape, dtype=np.float32)
    np.add.at(positive, (y[valid & polarity], x[valid & polarity]), 1)
    np.add.at(negative, (y[valid & ~polarity], x[valid & ~polarity]), 1)
    activity = np.maximum(positive, negative)
    nonzero = activity[activity > 0]
    scale = max(float(np.percentile(nonzero, 99)) if nonzero.size else 0.0, 1.0)
    image = np.full((height, width, 3), 245.0, dtype=np.float32)
    image[..., 0] -= 220 * negative / scale
    image[..., 1] -= 180 * activity / scale
    image[..., 2] -= 220 * positive / scale
    return np.clip(image, 0, 255).astype(np.uint8)


def error_visualization(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    error = np.linalg.norm(prediction.astype(np.float64) - ground_truth.astype(np.float64), axis=-1)
    finite = error[np.isfinite(error)]
    scale = max(float(np.percentile(finite, 99)) if finite.size else 0.0, 1e-6)
    normalized = np.clip(error / scale, 0, 1)
    return np.stack((255 * normalized, 255 * np.sqrt(normalized), 255 * (1 - normalized)), axis=-1).astype(np.uint8)


def labeled_panel(array: np.ndarray, label: str, size: tuple[int, int]) -> Image.Image:
    width, height = size
    panel_image = Image.fromarray(array.astype(np.uint8)).resize((width, height), PIL_BILINEAR)
    output = Image.new("RGB", (width, height + 24), "white")
    output.paste(panel_image, (0, 24))
    ImageDraw.Draw(output).text((5, 5), label, fill="black")
    return output


def save_qualitative(path: Path, source_rgb: np.ndarray, target_rgb: np.ndarray,
                     old_events: dict[str, np.ndarray], new_events: dict[str, np.ndarray],
                     prediction: np.ndarray, ground_truth: np.ndarray) -> None:
    panels = [
        ("RGB source", source_rgb),
        ("RGB target (evaluation only)", target_rgb),
        ("Old Event_i from raw dvs_events", event_visualization(old_events)),
        ("New Event_i+1 from raw dvs_events", event_visualization(new_events)),
        ("GT flow_i (oracle evaluation)", flow_color(ground_truth)),
        ("E-RAFT prediction i->i+1", flow_color(prediction)),
        ("Endpoint error map (P99)", error_visualization(prediction, ground_truth)),
    ]
    cell = (400, 300)
    canvas = Image.new("RGB", (cell[0] * 4, (cell[1] + 24) * 2), "white")
    for index, (label, values) in enumerate(panels):
        image = labeled_panel(values, label, cell)
        canvas.paste(image, ((index % 4) * cell[0], (index // 4) * (cell[1] + 24)))
    canvas.save(path)


def write_summary(path: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    decision = metrics["decision"]
    scope = decision["diagnostic_scope"]
    positive = metrics["correct_old_new"]["positive_prediction"][scope]
    lines = [
        "# R3a E-RAFT/SEVD smoke summary",
        "",
        f"- Status: **{decision['status']}**",
        f"- Mapping: Event_{config['source_frame_id']} + Event_{config['target_frame_id']} -> "
        f"flow_{config['source_frame_id']} evaluation.",
        f"- Formal result: positive-sign, correctly ordered prediction; diagnostic scope `{scope}`.",
        f"- EPE: {positive['epe']}",
        f"- Angular error (degrees): {positive['angular_error_degrees']}",
        f"- Endpoint cosine: {positive['endpoint_cosine']}",
        f"- Predicted/GT magnitude ratio: {positive['predicted_to_gt_magnitude_ratio']}",
        f"- Block reasons: {decision['reasons'] or 'none'}",
        "",
        "Negative-sign and swapped-event results are diagnostics only and never replace the formal output.",
        "GT flow is oracle evaluation data and is not an E-RAFT input.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--source-frame-id", required=True, type=int)
    parser.add_argument("--eraft-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--input-height", type=int, default=480)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--mode", choices=("standard",), default="standard")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root, eraft_root = args.root.expanduser().resolve(), args.eraft_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.bins != 15 or (args.input_height, args.input_width) != MODEL_HW or args.iterations <= 0:
        print("error: frozen R3a contract requires bins=15, input=480x640, iterations>0", file=sys.stderr)
        return 2
    if not root.is_dir() or not eraft_root.is_dir() or not checkpoint.is_file():
        print("error: --root/--eraft-root must be directories and --checkpoint must be a file", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "status": "starting",
        "scope": "R3a E-RAFT flow-only single-transition smoke",
        "recording": args.recording,
        "source_frame_id": args.source_frame_id,
        "target_frame_id": args.source_frame_id + 1,
        "old_event_frame_id": args.source_frame_id,
        "new_event_frame_id": args.source_frame_id + 1,
        "gt_flow_frame_id": args.source_frame_id,
        "eraft_root": str(eraft_root),
        "checkpoint": str(checkpoint),
        "bins": args.bins,
        "model_input_hw": [args.input_height, args.input_width],
        "native_hw": list(NATIVE_HW),
        "coordinate_scale_xy": [args.input_width / NATIVE_HW[1], args.input_height / NATIVE_HW[0]],
        "native_prediction_scale_xy": [NATIVE_HW[1] / args.input_width, NATIVE_HW[0] / args.input_height],
        "iterations": args.iterations,
        "mode": args.mode,
        "flow_init": None,
        "warm_start": False,
        "requested_device": args.device,
        "gt_motion_threshold_pixels": MOTION_THRESHOLD_PIXELS,
        "formal_prediction": "correct old/new, source-to-target positive sign",
        "diagnostics_do_not_select_sign_order_or_polarity": True,
    }
    atomic_write_json(output_dir / "config.json", config)
    try:
        paths = resolve_transition_paths(root, args.recording, args.source_frame_id)
        resolved_paths = {name: str(path) for name, path in paths.items()}
        print(json_dumps({"resolved_paths": resolved_paths}), flush=True)
        config["resolved_paths"] = resolved_paths
        config["transition_mapping"] = {
            "source_rgb": {"frame_id": args.source_frame_id, "path": resolved_paths["source_rgb"]},
            "target_rgb": {"frame_id": args.source_frame_id + 1, "path": resolved_paths["target_rgb"]},
            "old_event_packet": {"frame_id": args.source_frame_id, "path": resolved_paths["old_event"]},
            "new_event_packet": {"frame_id": args.source_frame_id + 1, "path": resolved_paths["new_event"]},
            "gt_evaluation_flow": {"frame_id": args.source_frame_id, "path": resolved_paths["gt_flow"]},
        }
        atomic_write_json(output_dir / "config.json", config)

        decode_start = time.perf_counter()
        old_events = read_sevd_events(paths["old_event"])
        new_events = read_sevd_events(paths["new_event"])
        raw_gt = read_flow(paths["gt_flow"])
        npz_decode_seconds = time.perf_counter() - decode_start
        old_validation = validate_events(old_events, paths["old_event"])
        new_validation = validate_events(new_events, paths["new_event"])
        if raw_gt.shape != (*NATIVE_HW, 2) or not np.isfinite(raw_gt).all():
            raise ValueError(f"GT flow must be finite with shape (960,1280,2), got {raw_gt.shape}")

        voxel_start = time.perf_counter()
        old_voxel, old_stats = voxelize_events(old_events, args.bins, MODEL_HW, NATIVE_HW)
        new_voxel, new_stats = voxelize_events(new_events, args.bins, MODEL_HW, NATIVE_HW)
        voxel_seconds = time.perf_counter() - voxel_start
        expected_input_shape = (1, 15, 480, 640)
        if old_voxel.shape != expected_input_shape or new_voxel.shape != expected_input_shape:
            raise ValueError(f"voxel tensor contract failed: old={old_voxel.shape}, new={new_voxel.shape}")
        print(json_dumps({"old_tensor_shape": list(old_voxel.shape),
                          "new_tensor_shape": list(new_voxel.shape)}), flush=True)

        source_rgb, target_rgb = read_rgb(paths["source_rgb"]), read_rgb(paths["target_rgb"])
        if source_rgb.shape != (*NATIVE_HW, 3) or target_rgb.shape != (*NATIVE_HW, 3):
            raise ValueError(f"SEVD RGB inputs must have shape (960,1280,3), got "
                             f"{source_rgb.shape} and {target_rgb.shape}")
        polarity = polarity_sanity(new_events, source_rgb, target_rgb)

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required in the isolated E-RAFT environment") from exc
        device = select_device(torch, args.device)
        model, checkpoint_load = load_model(torch, eraft_root, checkpoint, args.bins, args.mode, device)
        old_tensor = torch.from_numpy(old_voxel).to(device=device, dtype=torch.float32)
        new_tensor = torch.from_numpy(new_voxel).to(device=device, dtype=torch.float32)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        correct, correct_timing = timed_forward(torch, model, old_tensor, new_tensor,
                                                 args.iterations, device)
        swapped, swapped_timing = timed_forward(torch, model, new_tensor, old_tensor,
                                                 args.iterations, device)
        print(json_dumps({"correct_output_shape": list(correct.shape),
                          "swapped_output_shape": list(swapped.shape)}), flush=True)

        resize_start = time.perf_counter()
        correct_native = resize_prediction_native(correct.detach().cpu().numpy())
        correct_resize_seconds = time.perf_counter() - resize_start
        resize_start = time.perf_counter()
        swapped_native = resize_prediction_native(swapped.detach().cpu().numpy())
        swapped_resize_seconds = time.perf_counter() - resize_start
        ground_truth = np.asarray(sevd_flow_to_pixel_displacement(raw_gt, NATIVE_HW), dtype=np.float32)
        valid_mask = np.isfinite(ground_truth).all(axis=-1)
        motion_mask = valid_mask & (np.linalg.norm(ground_truth.astype(np.float64), axis=-1)
                                    >= MOTION_THRESHOLD_PIXELS)
        correct_report = signed_metric_report(correct_native, ground_truth, motion_mask)
        swapped_report = signed_metric_report(swapped_native, ground_truth, motion_mask)
        correct_stats = prediction_statistics(correct_native, str(correct.dtype), str(correct.device))
        swapped_stats = prediction_statistics(swapped_native, str(swapped.dtype), str(swapped.device))
        decision = smoke_decision(correct_report, swapped_report, correct_stats)
        metrics = {
            "formal_prediction": "correct_old_new.positive_prediction",
            "sign_and_order_are_diagnostics_only": True,
            "gt_motion_threshold_pixels": MOTION_THRESHOLD_PIXELS,
            "valid_pixel_count": int(np.count_nonzero(valid_mask)),
            "motion_pixel_count": int(np.count_nonzero(motion_mask)),
            "correct_old_new": correct_report,
            "swapped_old_new": swapped_report,
            "prediction_statistics": {"correct_old_new": correct_stats, "swapped_old_new": swapped_stats},
            "model_output_contract": {
                "correct": {"shape": list(correct.shape), "dtype": str(correct.dtype),
                            "device": str(correct.device), "finite": bool(correct.isfinite().all().item())},
                "swapped": {"shape": list(swapped.shape), "dtype": str(swapped.dtype),
                            "device": str(swapped.device), "finite": bool(swapped.isfinite().all().item())},
            },
            "prediction_magnitude": {
                "background_mean": float(np.mean(np.linalg.norm(correct_native[~motion_mask].astype(np.float64), axis=-1)))
                if np.any(~motion_mask) else None,
                "motion_region_mean": float(np.mean(np.linalg.norm(correct_native[motion_mask].astype(np.float64), axis=-1)))
                if np.any(motion_mask) else None,
                "background_pixel_count": int(np.count_nonzero(~motion_mask)),
                "motion_region_pixel_count": int(np.count_nonzero(motion_mask)),
            },
            "polarity_sanity": polarity,
            "packet_statistics": {
                "old": {**old_validation, **old_stats},
                "new": {**new_validation, **new_stats},
            },
            "efficiency_seconds": {
                "npz_decode": npz_decode_seconds,
                "voxelization": voxel_seconds,
                "network_forward_correct": correct_timing,
                "network_forward_swapped": swapped_timing,
                "resize_and_unit_conversion_correct": correct_resize_seconds,
                "resize_and_unit_conversion_swapped": swapped_resize_seconds,
            },
        }
        peak_cuda = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        } if device.type == "cuda" else {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
        metrics["peak_cuda_memory"] = peak_cuda
        metrics["decision"] = decision

        np.savez_compressed(output_dir / "prediction.npz",
                            flow=correct_native.astype(np.float32),
                            swapped_flow=swapped_native.astype(np.float32),
                            valid_mask=valid_mask,
                            motion_mask=motion_mask)
        save_qualitative(output_dir / "qualitative.png", source_rgb, target_rgb,
                         old_events, new_events, correct_native, ground_truth)
        eventdelta_revision = git_head(REPOSITORY_ROOT)
        cudnn_version = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
        gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
        environment = {
            "eventdelta_commit": eventdelta_revision,
            "eventdelta_worktree_dirty": bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT, text=True,
                capture_output=True, check=False).stdout.strip()),
            "eraft_commit": git_head(eraft_root),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_load": checkpoint_load,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": cudnn_version,
            "gpu": gpu_name,
            "selected_device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "input_shapes": {"old": list(old_tensor.shape), "new": list(new_tensor.shape)},
            "model_output_shapes": {"correct": list(correct.shape), "swapped": list(swapped.shape)},
            "saved_output_shape": list(correct_native.shape),
            "peak_cuda_memory": peak_cuda,
        }
        try:
            import torchvision
            environment["torchvision"] = torchvision.__version__
        except ImportError:
            environment["torchvision"] = None
        config["status"] = decision["status"]
        atomic_write_json(output_dir / "config.json", config)
        atomic_write_json(output_dir / "environment.json", environment)
        atomic_write_json(output_dir / "metrics.json", metrics)
        write_summary(output_dir / "summary.md", config, metrics)
        print(json_dumps({"status": decision["status"], "metrics": metrics,
                          "environment": environment}), flush=True)
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        config["status"] = "blocked"
        config["blocked_error"] = error
        atomic_write_json(output_dir / "config.json", config)
        atomic_write_json(output_dir / "metrics.json", {"decision": {"status": "blocked",
                                                                       "reasons": ["runtime_or_interface_error"]},
                                                          "error": error})
        environment_path = output_dir / "environment.json"
        if not environment_path.exists():
            blocked_environment: dict[str, Any] = {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "eraft_commit": None,
                "checkpoint_sha256": None,
                "numpy_global_monkey_patch": False,
                "status": "blocked_before_environment_capture_completed",
            }
            try:
                blocked_environment["eraft_commit"] = git_head(eraft_root)
            except (OSError, RuntimeError):
                pass
            try:
                blocked_environment["checkpoint_sha256"] = sha256_file(checkpoint)
            except OSError:
                pass
            atomic_write_json(environment_path, blocked_environment)
        (output_dir / "summary.md").write_text(
            "# R3a E-RAFT/SEVD smoke summary\n\n"
            f"- Status: **blocked**\n- Error: `{type(exc).__name__}: {exc}`\n",
            encoding="utf-8",
        )
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
