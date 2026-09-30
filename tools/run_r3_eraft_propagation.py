#!/usr/bin/env python3
"""Run the fixed R3b E-RAFT -> Qwen recurrent feature-propagation smoke."""

from __future__ import annotations

import argparse
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from event_delta.flow_warp import (  # noqa: E402
    forward_splat_torch,
    resize_flow_to_grid,
    sevd_flow_to_feature_displacement,
    sevd_flow_to_pixel_displacement,
)
from event_delta.json_io import atomic_write_json, json_dumps  # noqa: E402
from event_delta.qwen_features import (  # noqa: E402
    ProbeError,
    QwenVisionExtractor,
    TokenOrderError,
    require_runtime,
    select_device_name,
)
from event_delta.r3_propagation import (  # noqa: E402
    LATENCY_FIELDS,
    compare_feature_variants,
    complete_online_delta_latency,
    event_support_mask,
    expanded_decision,
    predicted_flow_to_feature_displacement,
    transition_frame_ids,
    validate_latency_fields,
)
from tools.audit_sevd import find_recording  # noqa: E402
from tools.run_r2_oracle import (  # noqa: E402
    allocate_window_counts,
    index_files,
    read_flow,
    read_rgb,
    select_activity_windows,
    select_qualitative_anchors,
    validate_window_files,
    write_csv,
)
from tools.smoke_eraft_sevd import (  # noqa: E402
    ERAFT_COMMIT,
    NATIVE_HW,
    flow_metrics,
    index_files as index_event_files,
    load_model,
    read_sevd_events,
    require_prediction,
    resize_prediction_native,
    validate_events,
    voxelize_events,
)


def validate_r3b_design(args: argparse.Namespace) -> None:
    if args.recording != ["fixed-3", "fixed-4"]:
        raise ValueError("R3b first round requires --recording fixed-3 --recording fixed-4 in that order")
    expected = {"windows": 6, "anchor_stride": 10, "max_offset": 9,
                "seed": 20260929, "feature_stage": "post_merger", "bins": 15,
                "input_height": 480, "input_width": 640, "iterations": 12,
                "mode": "standard", "motion_threshold_pixels": 1.0}
    mismatches = {name: (getattr(args, name), value) for name, value in expected.items()
                  if getattr(args, name) != value}
    if mismatches:
        raise ValueError(f"R3b first-round protocol is fixed; mismatches={mismatches}")


def validate_recurrent_files(rgb: dict[int, Path], flow: dict[int, Path],
                             events: dict[int, Path], anchor: int, max_offset: int) -> None:
    validate_window_files(rgb, flow, anchor, max_offset)
    missing = [identifier for identifier in range(anchor, anchor + max_offset + 1)
               if identifier not in events]
    if missing:
        raise FileNotFoundError(f"missing Event frames for anchor {anchor}: {missing}")


def timed_call(torch: Any, device: Any, operation: Callable[[], Any]) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        torch.cuda.synchronize(device)
        return result, float(start.elapsed_time(end))
    start = time.perf_counter()
    result = operation()
    return result, (time.perf_counter() - start) * 1000.0


def eraft_forward_once(torch: Any, model: Any, old: Any, new: Any, iterations: int,
                       device: Any) -> tuple[Any, float]:
    def operation() -> Any:
        with torch.inference_mode():
            return model(image1=old, image2=new, iters=iterations, flow_init=None)
    output, elapsed = timed_call(torch, device, operation)
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise ValueError("E-RAFT must return (low_resolution_flow, prediction_sequence)")
    predictions = output[1]
    if not isinstance(predictions, (tuple, list)) or len(predictions) != iterations:
        raise ValueError(f"E-RAFT must return {iterations} full-resolution predictions")
    return require_prediction(predictions[-1]), elapsed


def warmup_eraft(torch: Any, model: Any, old: Any, new: Any, iterations: int) -> None:
    with torch.inference_mode():
        model(image1=old, image2=new, iters=iterations, flow_init=None)


def flow_quality(prediction: np.ndarray, gt: np.ndarray, motion_threshold: float) -> dict[str, Any]:
    finite = np.isfinite(prediction).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    gt_magnitude = np.linalg.norm(gt.astype(np.float64), axis=-1)
    motion = finite & (gt_magnitude >= motion_threshold)
    background = finite & ~motion
    pred_magnitude = np.linalg.norm(prediction.astype(np.float64), axis=-1)
    output: dict[str, Any] = {}
    for scope, mask in (("all", finite), ("motion", motion)):
        for key, value in flow_metrics(prediction, gt, mask).items():
            output[f"eraft_flow_{scope}_{key}"] = value
    output["predicted_background_flow_magnitude"] = (
        float(np.mean(pred_magnitude[background])) if np.any(background) else None)
    output["gt_motion_pixel_ratio"] = float(np.mean(motion))
    return output


def aggregate_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    excluded = {"anchor_frame_id", "source_frame_id", "target_frame_id", "flow_frame_id", "offset"}
    for identity, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        item: dict[str, Any] = dict(zip(group_keys, identity))
        item["observation_count"] = len(group)
        numeric_keys = list(dict.fromkeys(key for row in group for key, value in row.items()
                                          if key not in excluded and isinstance(value, (int, float))))
        for key in numeric_keys:
            values = [float(row[key]) for row in group if row.get(key) is not None
                      and math.isfinite(float(row[key]))]
            item[f"{key}_mean"] = float(np.mean(values)) if values else None
        output.append(item)
    return output


def efficiency_report(rows: list[dict[str, Any]], peak_cuda_memory: int | None) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for field in LATENCY_FIELDS:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        stages[field] = {"count": len(values), "mean_ms": float(np.mean(values)) if values else None,
                         "p50_ms": float(np.percentile(values, 50)) if values else None,
                         "p95_ms": float(np.percentile(values, 95)) if values else None,
                         "sum_ms": float(np.sum(values)) if values else None}
    return {
        "timing_scope": {
            "dataset_npz_decode_ms": "old/new Event NPZ decode; GT-flow decode excluded from online path",
            "voxelization_ms": "two independent packet voxels",
            "eraft_forward_ms": "standard zero-init forward, no warm start; one startup warmup excluded",
            "flow_resize_grid_conversion_ms": "prediction upsample plus raw/masked/GT feature-grid conversion",
            "feature_warp_ms": "three recurrent forward splats (raw, masked, GT)",
            "qwen_vit_ms": "dense target vision feature extraction, offline reference only",
            "complete_online_delta_path_ms": "Event decode + voxelization + E-RAFT + flow conversion + raw feature warp",
        },
        "stages": stages,
        "peak_cuda_memory_bytes": peak_cuda_memory,
        "warning": "Network forward latency alone is not end-to-end online latency.",
    }


def _matplotlib() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "eventdelta-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_recovery(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt = _matplotlib()
    offsets = sorted({int(row["offset"]) for row in rows})
    for suffix, key, ylabel in (
        ("recovery_vs_offset.png", "recovery", "relative cosine-error recovery"),
        ("oracle_gap_closed.png", "oracle_gap_closed", "fraction of GT-oracle gap closed"),
    ):
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for scope, linestyle in (("all_valid", "-"), ("motion", "--")):
            for method, marker in (("raw_eraft", "o"), ("masked_eraft", "s")):
                values = []
                for offset in offsets:
                    samples = [row.get(f"{scope}_{method}_cosine_distance_{key}") for row in rows
                               if int(row["offset"]) == offset]
                    samples = [float(value) for value in samples if value is not None]
                    values.append(float(np.mean(samples)) if samples else np.nan)
                axis.plot(offsets, values, linestyle=linestyle, marker=marker,
                          label=f"{method}/{scope}")
        axis.axhline(0, color="black", linewidth=1)
        if key == "oracle_gap_closed":
            axis.axhline(0.2, color="tab:red", linewidth=1, linestyle=":", label="expanded threshold")
        axis.set(xlabel="offset from anchor", ylabel=ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / suffix, dpi=140)
        plt.close(fig)


def plot_qualitative(example: dict[str, Any], path: Path) -> None:
    plt = _matplotlib()
    target = example["target"]
    methods = ("stale", "raw_eraft", "masked_eraft", "gt_oracle")
    residuals = {method: np.linalg.norm(example[method] - target, axis=-1) for method in methods}
    common = example["common_valid"]
    selected = np.concatenate([values[common] for values in residuals.values()])
    limit = max(float(np.percentile(selected, 99)) if selected.size else 0.0, 1e-8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.reshape(-1)
    axes[0].imshow(example["anchor_rgb"]); axes[0].set_title("anchor RGB")
    axes[1].imshow(example["target_rgb"]); axes[1].set_title("dense target RGB (offline)")
    for axis, method in zip(axes[2:6], methods):
        shown = axis.imshow(np.where(common, residuals[method], np.nan), cmap="magma", vmin=0, vmax=limit)
        axis.set_title(f"{method} feature L2")
        fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    axes[6].imshow(common, cmap="gray", vmin=0, vmax=1); axes[6].set_title("shared evaluation mask")
    axes[7].imshow(common & example["motion"], cmap="gray", vmin=0, vmax=1)
    axes[7].set_title("shared motion-token mask")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"{example['recording']} anchor={example['anchor']} offset={example['offset']}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def summary_markdown(decision: dict[str, Any], rows: list[dict[str, Any]],
                     efficiency: dict[str, Any], r3a_passed: bool) -> str:
    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None
    return f"""# R3b E-RAFT Qwen feature-propagation smoke

## Integration status

- R3a integration passed in this run: **{str(r3a_passed).lower()}** (pinned repo, strict checkpoint load, finite shaped output).
- R3b scientific result: **{decision['status']}**.
- This decision is based on all six fixed windows, not on a single transition direction check.

## Scientific comparison

- Motion-token raw E-RAFT recovery: {mean('motion_raw_eraft_cosine_distance_recovery')}.
- Motion-token masked E-RAFT recovery: {mean('motion_masked_eraft_cosine_distance_recovery')}.
- Motion-token GT oracle recovery: {mean('motion_gt_oracle_cosine_distance_recovery')}.
- Raw E-RAFT oracle gap closed: {mean('motion_raw_eraft_cosine_distance_oracle_gap_closed')}.
- Masked E-RAFT oracle gap closed: {mean('motion_masked_eraft_cosine_distance_oracle_gap_closed')}.
- Decision criteria: `{json_dumps(decision['criteria'])}`.

## Efficiency boundary

- Complete online delta path P50: {efficiency['stages']['complete_online_delta_path_ms']['p50_ms']} ms.
- E-RAFT network forward P50: {efficiency['stages']['eraft_forward_ms']['p50_ms']} ms.
- Qwen dense target ViT P50 (offline reference): {efficiency['stages']['qwen_vit_ms']['p50_ms']} ms.
- The complete path includes Event NPZ decode, two-packet voxelization, E-RAFT, flow conversion, and raw feature warp. It excludes GT decode and dense target Qwen.

## Boundary

- Event_i + Event_i+1 predicts i->i+1. GT flow_i is used only for oracle propagation, motion masks, and flow-quality evaluation.
- `stale`, raw E-RAFT, fixed event-support-masked E-RAFT, and GT oracle use the identical intersection token mask.
- The mask is a fixed union of Event_i/Event_i+1 coordinate support with the configured square dilation; no parameter search is performed.
- Failure of the expanded gate means `stop/diagnostic`; this tool does not tune parameters automatically.
- This is a feature-propagation validation, not VLM accuracy evaluation and not evidence that the vision encoder has already been saved in a production pipeline.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--recording", action="append", default=None)
    parser.add_argument("--eraft-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--input-height", type=int, default=480)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--mode", default="standard")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--anchor-stride", type=int, default=10)
    parser.add_argument("--max-offset", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--feature-stage", default="post_merger")
    parser.add_argument("--motion-threshold-pixels", type=float, default=1.0)
    parser.add_argument("--support-dilation-radius", type=int, default=3)
    parser.add_argument("--maximum-coverage-drop", type=float, default=0.10)
    args = parser.parse_args(argv)
    if args.recording is None:
        args.recording = ["fixed-3", "fixed-4"]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_r3b_design(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not root.is_dir() or args.support_dilation_radius < 0:
        print("error: dataset root is missing or dilation radius is invalid", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        torch, transformers = require_runtime()
        device_name = select_device_name(args.device, bool(torch.cuda.is_available()))
        device = torch.device(device_name)
    except ProbeError as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
                              text=True, capture_output=True, check=False).stdout.strip() or None
    config: dict[str, Any] = {
        "status": "starting", "scope": "R3b small-scale end-to-end validation",
        "git_commit": revision, "dataset_root": str(root), "recordings": args.recording,
        "model": args.model, "feature_stage": args.feature_stage, "windows": args.windows,
        "windows_per_recording": allocate_window_counts(args.recording, args.windows),
        "anchor_stride": args.anchor_stride, "offsets": list(range(1, args.max_offset + 1)),
        "seed": args.seed, "activity_sampling": "exact R2 low/mid/high GT-flow activity ranking",
        "eraft": {"root": str(args.eraft_root.expanduser().resolve()),
                  "checkpoint": str(args.checkpoint.expanduser().resolve()), "bins": args.bins,
                  "input_hw": [args.input_height, args.input_width], "iterations": args.iterations,
                  "mode": args.mode, "warm_start": False,
                  "mapping": "Event_i + Event_i+1 -> predicted flow i->i+1; flow_i evaluation only"},
        "event_support_mask": {"packets": "union(Event_i, Event_i+1)", "binary": True,
                               "dilation": "square MaxFilter", "dilation_radius_pixels": args.support_dilation_radius,
                               "parameter_search": False, "outside_flow": 0.0},
        "evaluation_mask": "intersection(raw E-RAFT valid, masked E-RAFT valid, GT-oracle valid)",
        "motion_threshold_pixels": args.motion_threshold_pixels,
        "device": {"requested": args.device, "selected": device_name},
    }
    atomic_write_json(output_dir / "config.json", config)

    indexes: dict[str, Any] = {}
    selected: list[dict[str, Any]] = []
    try:
        counts = config["windows_per_recording"]
        for position, recording in enumerate(args.recording):
            recording_root = find_recording(root, recording).resolve()
            rgb = index_files(recording_root / "rgb_camera", ".png")
            flow = index_files(recording_root / "optical_flow_camera", ".npz")
            events = index_event_files(recording_root / "dvs_camera", ".npz", "-xytp.npz")
            indexes[recording] = {"root": recording_root, "rgb": rgb, "flow": flow, "events": events}
            chosen = select_activity_windows(rgb, flow, counts[recording], args.anchor_stride,
                                             args.max_offset, args.seed + position)
            for window in chosen:
                validate_recurrent_files(rgb, flow, events, window["anchor_frame_id"], args.max_offset)
                selected.append({"recording": recording, **window})
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error resolving R3b inputs: {exc}", file=sys.stderr)
        return 2
    config["selected_windows"] = selected
    atomic_write_json(output_dir / "config.json", config)

    try:
        eraft, checkpoint_metadata = load_model(torch, args.eraft_root.expanduser().resolve(),
                                                args.checkpoint.expanduser().resolve(), args.bins,
                                                args.mode, device)
        extractor = QwenVisionExtractor(args.model, args.local_files_only, args.device)
        first = selected[0]
        first_image = indexes[first["recording"]]["rgb"][first["anchor_frame_id"]]
        probe_inputs = extractor.preprocess(first_image)
        probe_feature, probe = extractor.encode_inputs(probe_inputs, args.feature_stage)
        if probe["token_order"]["status"] != "confirmed_from_installed_source":
            raise TokenOrderError(probe["token_order"]["warning"])
    except (ProbeError, OSError, ValueError, RuntimeError, ImportError) as exc:
        config.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(output_dir / "config.json", config)
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    config["checkpoint"] = checkpoint_metadata
    config["qwen_probe"] = probe
    config["runtime"] = extractor.runtime_diagnostics(probe_inputs, probe_feature, args.feature_stage)
    atomic_write_json(output_dir / "config.json", config)
    environment = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device), "checkpoint": checkpoint_metadata,
        "eraft_commit": ERAFT_COMMIT,
        "eraft_module_path": checkpoint_metadata.get("external_module_path"),
    }
    atomic_write_json(output_dir / "environment.json", environment)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows: list[dict[str, Any]] = []
    qualitative: dict[str, Any] = {}
    qualitative_anchors = select_qualitative_anchors(selected, args.recording)
    eraft_warmed = False
    r3a_output_valid = True
    try:
        for window_number, window in enumerate(selected, 1):
            recording = window["recording"]
            anchor = int(window["anchor_frame_id"])
            current = indexes[recording]
            print(f"R3b window {window_number}/{len(selected)}: {recording} anchor={anchor}", flush=True)
            (anchor_feature, _anchor_vit_ms) = timed_call(
                torch, extractor.device,
                lambda: extractor.feature_grid_from_image(current["rgb"][anchor], args.feature_stage).detach())
            feature_hw = tuple(int(value) for value in anchor_feature.shape[:2])
            raw_feature = masked_feature = gt_feature = anchor_feature
            initial_confidence = torch.ones(feature_hw, dtype=torch.float32, device=anchor_feature.device)
            raw_confidence = masked_confidence = gt_confidence = initial_confidence
            for offset in range(1, args.max_offset + 1):
                source_id, target_id, flow_id = transition_frame_ids(anchor, offset)
                old_path, new_path = current["events"][source_id], current["events"][target_id]
                decode_start = time.perf_counter()
                old_events, new_events = read_sevd_events(old_path), read_sevd_events(new_path)
                event_decode_ms = (time.perf_counter() - decode_start) * 1000.0
                old_stats = validate_events(old_events, old_path, NATIVE_HW)
                new_stats = validate_events(new_events, new_path, NATIVE_HW)
                voxel_start = time.perf_counter()
                old_voxel, old_voxel_stats = voxelize_events(
                    old_events, args.bins, (args.input_height, args.input_width), NATIVE_HW)
                new_voxel, new_voxel_stats = voxelize_events(
                    new_events, args.bins, (args.input_height, args.input_width), NATIVE_HW)
                voxel_ms = (time.perf_counter() - voxel_start) * 1000.0
                old_tensor = torch.from_numpy(old_voxel).to(device)
                new_tensor = torch.from_numpy(new_voxel).to(device)
                if not eraft_warmed:
                    warmup_eraft(torch, eraft, old_tensor, new_tensor, args.iterations)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    eraft_warmed = True
                prediction_tensor, eraft_ms = eraft_forward_once(
                    torch, eraft, old_tensor, new_tensor, args.iterations, device)
                prediction_model = prediction_tensor.detach().float().cpu().numpy()
                if not np.any(np.linalg.norm(prediction_model[0].transpose(1, 2, 0), axis=-1) > 1e-6):
                    r3a_output_valid = False
                    raise ValueError("E-RAFT prediction is all zero")

                gt_decode_start = time.perf_counter()
                gt_raw_numpy = read_flow(current["flow"][flow_id])
                gt_decode_ms = (time.perf_counter() - gt_decode_start) * 1000.0
                native_resize_start = time.perf_counter()
                prediction_native = resize_prediction_native(prediction_model, NATIVE_HW)
                native_resize_ms = (time.perf_counter() - native_resize_start) * 1000.0
                raw_pred_tensor = torch.from_numpy(prediction_native).to(extractor.device)
                raw_displacement, raw_grid_ms = timed_call(
                    torch, extractor.device,
                    lambda: predicted_flow_to_feature_displacement(raw_pred_tensor, feature_hw, NATIVE_HW))
                conversion_ms = native_resize_ms + raw_grid_ms
                diagnostic_conversion_start = time.perf_counter()
                support = event_support_mask([old_events, new_events], NATIVE_HW, args.support_dilation_radius)
                masked_native = prediction_native * support[..., None]
                masked_pred_tensor = torch.from_numpy(masked_native).to(extractor.device)
                masked_displacement = predicted_flow_to_feature_displacement(masked_pred_tensor, feature_hw, NATIVE_HW)
                gt_raw = torch.from_numpy(gt_raw_numpy).to(extractor.device)
                gt_displacement = sevd_flow_to_feature_displacement(gt_raw, feature_hw, NATIVE_HW)
                sampled_gt = resize_flow_to_grid(gt_raw, feature_hw)
                gt_pixel_grid = sevd_flow_to_pixel_displacement(sampled_gt, NATIVE_HW)
                motion_mask = torch.linalg.vector_norm(gt_pixel_grid.float(), dim=-1) >= args.motion_threshold_pixels
                if extractor.device.type == "cuda":
                    torch.cuda.synchronize(extractor.device)
                diagnostic_conversion_ms = (time.perf_counter() - diagnostic_conversion_start) * 1000.0

                (raw_feature, raw_valid, raw_confidence), warp_ms = timed_call(
                    torch, extractor.device,
                    lambda: forward_splat_torch(raw_feature, raw_displacement, raw_confidence))
                (masked_feature, masked_valid, masked_confidence), masked_warp_ms = timed_call(
                    torch, extractor.device,
                    lambda: forward_splat_torch(masked_feature, masked_displacement, masked_confidence))
                (gt_feature, gt_valid, gt_confidence), gt_warp_ms = timed_call(
                    torch, extractor.device,
                    lambda: forward_splat_torch(gt_feature, gt_displacement, gt_confidence))
                target_feature, target_vit_ms = timed_call(
                    torch, extractor.device,
                    lambda: extractor.feature_grid_from_image(current["rgb"][target_id], args.feature_stage))
                if tuple(target_feature.shape) != tuple(anchor_feature.shape):
                    raise ValueError(f"feature-grid shape changed: {anchor_feature.shape} -> {target_feature.shape}")

                arrays = [value.detach().float().cpu().numpy() for value in
                          (anchor_feature, raw_feature, masked_feature, gt_feature, target_feature)]
                masks = [value.detach().cpu().numpy() for value in
                         (raw_valid, masked_valid, gt_valid, motion_mask)]
                feature_metrics = compare_feature_variants(*arrays, *masks)
                gt_native = sevd_flow_to_pixel_displacement(gt_raw_numpy, NATIVE_HW)
                quality = flow_quality(prediction_native, gt_native, args.motion_threshold_pixels)
                latencies = {
                    "dataset_npz_decode_ms": event_decode_ms,
                    "voxelization_ms": voxel_ms,
                    "eraft_forward_ms": eraft_ms,
                    "flow_resize_grid_conversion_ms": conversion_ms,
                    "feature_warp_ms": warp_ms,
                    "qwen_vit_ms": target_vit_ms,
                }
                latencies["complete_online_delta_path_ms"] = complete_online_delta_latency(latencies)
                row = {
                    "recording": recording, "anchor_frame_id": anchor, "offset": offset,
                    "source_frame_id": source_id, "target_frame_id": target_id, "flow_frame_id": flow_id,
                    "activity_stratum": window["activity_stratum"], "activity_group": window["activity_group"],
                    "flow_activity_mean_pixels": window["flow_activity_mean_pixels"],
                    "old_event_path": str(old_path.relative_to(root)),
                    "new_event_path": str(new_path.relative_to(root)),
                    "gt_flow_path": str(current["flow"][flow_id].relative_to(root)),
                    "event_old_count": old_stats["event_count"], "event_new_count": new_stats["event_count"],
                    "old_voxel_nonzero_ratio": old_voxel_stats["nonzero_voxel_ratio"],
                    "new_voxel_nonzero_ratio": new_voxel_stats["nonzero_voxel_ratio"],
                    "event_support_coverage_ratio": float(np.mean(support)),
                    "event_support_dilation_radius": args.support_dilation_radius,
                    "gt_oracle_npz_decode_ms": gt_decode_ms,
                    "diagnostic_masked_feature_warp_ms": masked_warp_ms,
                    "gt_oracle_feature_warp_ms": gt_warp_ms,
                    "diagnostic_mask_and_oracle_flow_conversion_ms": diagnostic_conversion_ms,
                    **quality, **feature_metrics, **latencies,
                }
                validate_latency_fields(row)
                rows.append(row)
                if anchor == qualitative_anchors[recording] and offset == args.max_offset:
                    common = masks[0] & masks[1] & masks[2]
                    qualitative[recording] = {
                        "recording": recording, "anchor": anchor, "offset": offset,
                        "anchor_rgb": read_rgb(current["rgb"][anchor]),
                        "target_rgb": read_rgb(current["rgb"][target_id]),
                        "stale": arrays[0], "raw_eraft": arrays[1], "masked_eraft": arrays[2],
                        "gt_oracle": arrays[3], "target": arrays[4], "common_valid": common,
                        "motion": masks[3],
                    }
    except (OSError, ValueError, RuntimeError) as exc:
        config.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(output_dir / "config.json", config)
        print(f"blocked during R3b evaluation: {exc}", file=sys.stderr)
        return 2

    windows = aggregate_rows(rows, ("recording", "anchor_frame_id", "activity_stratum"))
    aggregates = [{"group_type": "recording_offset", **item}
                  for item in aggregate_rows(rows, ("recording", "offset"))]
    aggregates.extend({"group_type": "offset", **item} for item in aggregate_rows(rows, ("offset",)))
    aggregates.extend({"group_type": "overall", **item} for item in aggregate_rows(rows, ()))
    write_csv(output_dir / "per_offset_metrics.csv", rows)
    write_csv(output_dir / "window_summary.csv", windows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregates)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    efficiency = efficiency_report(rows, peak_memory)
    atomic_write_json(output_dir / "efficiency.json", efficiency)
    decision = expanded_decision(rows, args.recording, maximum_coverage_drop=args.maximum_coverage_drop)
    r3a_passed = bool(r3a_output_valid and checkpoint_metadata.get("strict_load"))
    config.update(status="complete", r3a_integration_passed=r3a_passed,
                  r3b_scientific_result=decision, row_count=len(rows))
    atomic_write_json(output_dir / "config.json", config)
    plot_recovery(rows, output_dir)
    for recording in args.recording:
        if recording not in qualitative:
            raise RuntimeError(f"missing qualitative example for {recording}")
        plot_qualitative(qualitative[recording], output_dir / f"qualitative_features_{recording}.png")
    (output_dir / "summary.md").write_text(
        summary_markdown(decision, rows, efficiency, r3a_passed), encoding="utf-8")
    print(f"wrote {output_dir}; R3b={decision['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
