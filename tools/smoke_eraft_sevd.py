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
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from event_delta.flow_warp import sevd_flow_to_pixel_displacement  # noqa: E402
from event_delta.json_io import atomic_write_json, json_dumps  # noqa: E402
from tools.audit_sevd import find_recording, frame_id  # noqa: E402
from tools.calibrate_sevd import read_event_npz, read_flow  # noqa: E402


ERAFT_COMMIT = "c58ce0524ea0ebfa9849991caafb547f44fe9bfd"
NATIVE_HW = (960, 1280)
MODEL_HW = (480, 640)
MOTION_THRESHOLD_PIXELS = 1.0
POLARITY_MIN_EVENTS = 100
POLARITY_MIN_ABS_CORRELATION = 0.05
PIL_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def index_files(directory: Path, suffix: str) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required directory does not exist: {directory}")
    indexed: dict[int, Path] = {}
    duplicates: list[int] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() != suffix:
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
    events = index_files(recording_root / "dvs_camera", ".npz")
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


def validate_events(events: dict[str, np.ndarray], path: Path) -> dict[str, Any]:
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
    polarity = np.asarray(events["pol"]).reshape(-1).astype(bool, copy=False)
    return {
        "event_count": count,
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
        return {"pixel_count": 0, "epe": None, "median_epe": None, "direction_cosine": None,
                "direction_cosine_pixel_count": 0, "1PE": None, "2PE": None, "3PE": None}
    pred = prediction[valid].astype(np.float64)
    truth = ground_truth[valid].astype(np.float64)
    epe = np.linalg.norm(pred - truth, axis=-1)
    pred_norm, truth_norm = np.linalg.norm(pred, axis=-1), np.linalg.norm(truth, axis=-1)
    directional = (pred_norm > 1e-8) & (truth_norm > 1e-8)
    cosine = (np.sum(pred[directional] * truth[directional], axis=-1)
              / (pred_norm[directional] * truth_norm[directional])) if np.any(directional) else np.empty(0)
    return {
        "pixel_count": count,
        "epe": float(np.mean(epe)),
        "median_epe": float(np.median(epe)),
        "direction_cosine": float(np.mean(np.clip(cosine, -1, 1))) if cosine.size else None,
        "direction_cosine_pixel_count": int(cosine.size),
        "1PE": float(np.mean(epe > 1.0)),
        "2PE": float(np.mean(epe > 2.0)),
        "3PE": float(np.mean(epe > 3.0)),
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


def load_model(torch: Any, eraft_root: Path, checkpoint_path: Path, bins: int,
               mode: str, device: Any) -> Any:
    revision = git_head(eraft_root)
    if revision != ERAFT_COMMIT:
        raise RuntimeError(f"E-RAFT HEAD mismatch: expected {ERAFT_COMMIT}, got {revision}")
    if str(eraft_root) not in sys.path:
        sys.path.insert(0, str(eraft_root))
    module = importlib.import_module("model.eraft")
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
    normalized = normalize_state_dict_prefix(state, set(model.state_dict()))
    model.load_state_dict(normalized, strict=True)
    return model.to(device).eval()


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


def save_qualitative(path: Path, source_rgb: np.ndarray, target_rgb: np.ndarray,
                     prediction: np.ndarray, ground_truth: np.ndarray) -> None:
    panels = [source_rgb, target_rgb, flow_color(prediction), flow_color(ground_truth)]
    canvas = Image.new("RGB", (NATIVE_HW[1] * 2, NATIVE_HW[0] * 2))
    for index, panel in enumerate(panels):
        image = Image.fromarray(panel).resize((NATIVE_HW[1], NATIVE_HW[0]), PIL_BILINEAR)
        canvas.paste(image, ((index % 2) * NATIVE_HW[1], (index // 2) * NATIVE_HW[0]))
    canvas.save(path)


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
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
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

    try:
        paths = resolve_transition_paths(root, args.recording, args.source_frame_id)
        resolved_paths = {name: str(path) for name, path in paths.items()}
        print(json_dumps({"resolved_paths": resolved_paths}), flush=True)
        config = {
            "status": "starting",
            "scope": "R3a E-RAFT flow-only single-transition smoke",
            "recording": args.recording,
            "source_frame_id": args.source_frame_id,
            "target_frame_id": args.source_frame_id + 1,
            "resolved_paths": resolved_paths,
            "eraft_root": str(eraft_root),
            "checkpoint": str(checkpoint),
            "bins": args.bins,
            "model_input_hw": list(MODEL_HW),
            "native_hw": list(NATIVE_HW),
            "iterations": args.iterations,
            "mode": args.mode,
            "flow_init": None,
            "requested_device": args.device,
            "gt_motion_threshold_pixels": MOTION_THRESHOLD_PIXELS,
            "formal_prediction": "correct old/new, source-to-target positive sign",
            "diagnostics_do_not_select_sign_order_or_polarity": True,
        }
        atomic_write_json(output_dir / "config.json", config)

        decode_start = time.perf_counter()
        old_events = read_event_npz(paths["old_event"])
        new_events = read_event_npz(paths["new_event"])
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
        model = load_model(torch, eraft_root, checkpoint, args.bins, args.mode, device)
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
        motion_mask = np.linalg.norm(ground_truth.astype(np.float64), axis=-1) >= MOTION_THRESHOLD_PIXELS
        metrics = {
            "formal_prediction": "correct_old_new.positive_prediction",
            "sign_and_order_are_diagnostics_only": True,
            "gt_motion_threshold_pixels": MOTION_THRESHOLD_PIXELS,
            "correct_old_new": signed_metric_report(correct_native, ground_truth, motion_mask),
            "swapped_old_new": signed_metric_report(swapped_native, ground_truth, motion_mask),
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

        np.savez_compressed(output_dir / "predicted_flow_native.npz", flow=correct_native.astype(np.float32))
        save_qualitative(output_dir / "flow_qualitative.png", source_rgb, target_rgb,
                         correct_native, ground_truth)
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
            "python": platform.python_version(),
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
        config["status"] = "complete"
        atomic_write_json(output_dir / "config.json", config)
        atomic_write_json(output_dir / "environment.json", environment)
        atomic_write_json(output_dir / "metrics.json", metrics)
        print(json_dumps({"status": "complete", "metrics": metrics,
                          "environment": environment}), flush=True)
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
