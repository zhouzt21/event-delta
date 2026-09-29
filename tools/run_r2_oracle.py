#!/usr/bin/env python3
"""Run R2 Qwen vision-feature + SEVD GT-flow oracle validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

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
from event_delta.metrics import (  # noqa: E402
    bootstrap_window_reductions,
    compare_stale_and_propagated,
    grouped_motion_aggregates,
    summarize_by_offset,
)
from event_delta.qwen_features import (  # noqa: E402
    ProbeError,
    QwenVisionExtractor,
    TokenOrderError,
    require_runtime,
    select_device_name,
)
from tools.audit_sevd import find_recording, frame_id  # noqa: E402


def index_files(directory: Path, suffix: str) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required directory does not exist: {directory}")
    output: dict[int, Path] = {}
    duplicates = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() != suffix:
            continue
        identifier = frame_id(path)
        if identifier is None:
            continue
        if identifier in output:
            duplicates.append(identifier)
        output[identifier] = path
    if duplicates:
        raise ValueError(f"duplicate frame IDs in {directory}: {sorted(set(duplicates))[:10]}")
    return output


def validate_window_files(rgb: dict[int, Path], flow: dict[int, Path], anchor: int, max_offset: int) -> None:
    for identifier in range(anchor, anchor + max_offset + 1):
        if identifier not in rgb:
            raise FileNotFoundError(f"missing RGB frame {identifier} for anchor {anchor}")
    for identifier in range(anchor, anchor + max_offset):
        if identifier not in flow:
            raise FileNotFoundError(f"missing flow frame {identifier} for transition {identifier}->{identifier + 1}")


def read_flow(path: Path) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "flow" not in archive.files:
                raise ValueError(f"flow key missing in {path}")
            flow = np.asarray(archive["flow"], dtype=np.float32)
    except OSError as exc:
        raise OSError(f"failed to read flow file {path}: {exc}") from exc
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"expected HxWx2 flow in {path}, got {flow.shape}")
    return flow


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def window_flow_activity(flow: dict[int, Path], anchor: int, max_offset: int) -> float:
    values = []
    for identifier in range(anchor, anchor + max_offset):
        raw = read_flow(flow[identifier])
        displacement = sevd_flow_to_pixel_displacement(raw, raw.shape[:2])
        magnitude = np.linalg.norm(displacement.astype(np.float64), axis=-1)
        finite = magnitude[np.isfinite(magnitude)]
        values.append(float(np.mean(finite)) if finite.size else math.nan)
    finite_values = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else math.nan


def candidate_anchors(rgb: dict[int, Path], flow: dict[int, Path], anchor_stride: int,
                      max_offset: int) -> list[int]:
    if not rgb:
        return []
    first = min(rgb)
    candidates = []
    for anchor in sorted(rgb):
        if (anchor - first) % anchor_stride:
            continue
        try:
            validate_window_files(rgb, flow, anchor, max_offset)
        except FileNotFoundError:
            continue
        candidates.append(anchor)
    return candidates


def select_activity_windows(rgb: dict[int, Path], flow: dict[int, Path], count: int,
                            anchor_stride: int, max_offset: int, seed: int) -> list[dict[str, Any]]:
    if count == 0:
        return []
    anchors = candidate_anchors(rgb, flow, anchor_stride, max_offset)
    if len(anchors) < count:
        raise ValueError(f"only {len(anchors)} complete stride-aligned windows are available; requested {count}")
    rng = random.Random(seed)
    tie_breakers = {anchor: rng.random() for anchor in anchors}
    scored = [(window_flow_activity(flow, anchor, max_offset), tie_breakers[anchor], anchor) for anchor in anchors]
    if any(not math.isfinite(score) for score, _tie, _anchor in scored):
        raise ValueError("non-finite flow activity prevents deterministic window selection")
    scored.sort()
    if count == 1:
        positions = [len(scored) // 2]
    else:
        positions = [round(index * (len(scored) - 1) / (count - 1)) for index in range(count)]
    quantiles = [index / max(count - 1, 1) if count > 1 else 0.5 for index in range(count)]
    labels = (["low", "mid", "high"] if count == 3
              else [f"activity_quantile_{quantile:.2f}" for quantile in quantiles])
    activity_groups = [("low" if quantile < 1 / 3 else "mid" if quantile < 2 / 3 else "high")
                       for quantile in quantiles]
    return [{"anchor_frame_id": scored[position][2], "flow_activity_mean_pixels": scored[position][0],
             "activity_stratum": label, "activity_group": activity_group,
             "activity_quantile": quantile, "activity_rank": position + 1,
             "candidate_window_count": len(scored)}
            for position, label, activity_group, quantile
            in zip(positions, labels, activity_groups, quantiles)]


def allocate_window_counts(recordings: list[str], total_windows: int) -> dict[str, int]:
    """Allocate a total window budget evenly; CLI-order recordings receive any remainder."""
    if not recordings or len(set(recordings)) != len(recordings):
        raise ValueError("--recording values must be non-empty and unique")
    if total_windows < len(recordings):
        raise ValueError("--windows is the total across recordings and must allow at least one per recording")
    base, remainder = divmod(total_windows, len(recordings))
    return {recording: base + (index < remainder) for index, recording in enumerate(recordings)}


def select_qualitative_anchors(selected: list[dict[str, Any]], recordings: list[str]) -> dict[str, int]:
    """Select the highest-activity sampled window independently for each recording."""
    return {
        recording: int(max((window for window in selected if window["recording"] == recording),
                           key=lambda window: window["flow_activity_mean_pixels"])["anchor_frame_id"])
        for recording in recordings
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def device_timings(torch: Any, device_type: str, operation: Callable[[], Any],
                   warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    timings = []
    if device_type == "cuda":
        torch.cuda.synchronize()
        for _ in range(repeats):
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            torch.cuda.synchronize()
            timings.append(float(start.elapsed_time(end)))
        method = "CUDA events with synchronize before and after each measurement"
    else:
        for _ in range(repeats):
            start = time.perf_counter()
            operation()
            timings.append((time.perf_counter() - start) * 1000.0)
        method = "CPU perf_counter; no torch.cuda calls"
    return {"warmup": warmup, "repeats": repeats, "milliseconds": timings,
            "method": method, "p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95))}


def profile_efficiency(extractor: QwenVisionExtractor, image_path: Path, stage: str,
                       feature: Any, displacement: Any, warmup: int, repeats: int) -> dict[str, Any]:
    torch = extractor.torch
    inputs = extractor.preprocess(image_path)
    device_type = extractor.device.type
    vision = device_timings(torch, device_type, lambda: extractor.feature_grid_from_inputs(inputs, stage),
                            warmup, repeats)
    confidence = torch.ones(feature.shape[:2], dtype=torch.float32, device=feature.device)
    warp = device_timings(torch, device_type,
                          lambda: forward_splat_torch(feature, displacement, confidence), warmup, repeats)

    def nine_steps() -> Any:
        current, current_confidence = feature, confidence
        for _ in range(9):
            current, _valid, current_confidence = forward_splat_torch(current, displacement, current_confidence)
        return current

    recurrent = device_timings(torch, device_type, nine_steps, warmup, repeats)
    dense_cost = 10 * vision["p50_ms"]
    propagation_cost = vision["p50_ms"] + 9 * warp["p50_ms"]
    return {"timing_method": vision["method"], "selected_device": str(extractor.device),
            "timing_scope": "vision-module GPU forward excludes image decode/processor and disk I/O; warp excludes flow disk I/O",
            "vision_encoder": vision, "single_feature_warp": warp, "recurrent_nine_step_warp": recurrent,
            "cuda_memory_allocated_bytes": (int(torch.cuda.memory_allocated()) if device_type == "cuda" else None),
            "peak_cuda_memory_bytes": (int(torch.cuda.max_memory_allocated()) if device_type == "cuda" else None),
            "feature_shape": list(feature.shape), "visual_token_count": int(feature.shape[0] * feature.shape[1]),
            "dense_10hz_cost_ms": dense_cost, "oracle_1hz_plus_9warp_cost_ms": propagation_cost,
            "theoretical_speedup": dense_cost / propagation_cost if propagation_cost > 0 else None,
            "warp_to_vision_p50_ratio": warp["p50_ms"] / vision["p50_ms"] if vision["p50_ms"] > 0 else None,
            "scope_warning": "GT-flow reading/warping is not the final Event method; this only tests whether avoiding repeated ViT work could save compute"}


def _matplotlib() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "eventdelta-matplotlib"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create R2 smoke plots") from exc
    return plt


def plot_metrics(offset_rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt = _matplotlib()
    offsets = [row["offset"] for row in offset_rows]
    fig, axis = plt.subplots(figsize=(7, 4))
    for scope, linestyle in (("all_valid", "-"), ("motion", "--")):
        axis.plot(offsets, [row.get(f"{scope}_stale_cosine_distance_mean") for row in offset_rows],
                  linestyle, marker="o", label=f"stale/{scope}")
        axis.plot(offsets, [row.get(f"{scope}_propagated_cosine_distance_mean") for row in offset_rows],
                  linestyle, marker="s", label=f"GT-flow/{scope}")
    axis.set(xlabel="offset from 1 Hz RGB anchor (100 ms steps)", ylabel="cosine distance",
             title="Feature distance to dense-RGB oracle")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "distance_vs_offset.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    for scope in ("all_valid", "motion"):
        axis.plot(offsets, [row.get(f"{scope}_cosine_distance_relative_error_reduction_mean")
                           for row in offset_rows], marker="o", label=scope)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="offset from anchor", ylabel="relative cosine-error reduction",
             title="GT-flow recovery over stale features")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "recovery_vs_offset.png", dpi=140)
    plt.close(fig)


def plot_qualitative(example: dict[str, Any], output: Path) -> None:
    plt = _matplotlib()
    stale, propagated, target = example["stale"], example["propagated"], example["target"]
    stale_residual = np.linalg.norm(stale - target, axis=-1)
    propagated_residual = np.linalg.norm(propagated - target, axis=-1)
    improvement = stale_residual - propagated_residual
    valid = np.asarray(example["valid"], dtype=bool)
    residual_values = np.concatenate((stale_residual[valid], propagated_residual[valid]))
    residual_limit = float(np.percentile(residual_values, 99)) if residual_values.size else 1.0
    residual_limit = max(residual_limit, 1e-12)
    flow = np.asarray(example["flow_magnitude_pixels"])
    finite_flow = flow[np.isfinite(flow) & (flow > 0)]
    flow_limit = float(np.percentile(finite_flow, 99)) if finite_flow.size else 1.0
    flow_limit = max(flow_limit, 1e-12)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.reshape(-1)
    axes[0].imshow(example["anchor_rgb"])
    axes[0].set_title("anchor RGB (method input)")
    axes[1].imshow(example["target_rgb"])
    axes[1].set_title("target RGB (offline GT only)")
    shown = axes[2].imshow(flow, cmap="magma", vmin=0, vmax=flow_limit)
    axes[2].set_title("current GT-flow magnitude [pixel]")
    fig.colorbar(shown, ax=axes[2], fraction=0.046, pad=0.04)
    shown = axes[3].imshow(np.where(valid, stale_residual, np.nan),
                           cmap="magma", vmin=0, vmax=residual_limit)
    axes[3].set_title("stale residual L2 (valid mask)")
    fig.colorbar(shown, ax=axes[3], fraction=0.046, pad=0.04)
    shown = axes[4].imshow(np.where(valid, propagated_residual, np.nan),
                           cmap="magma", vmin=0, vmax=residual_limit)
    axes[4].set_title("propagated residual L2 (valid mask)")
    fig.colorbar(shown, ax=axes[4], fraction=0.046, pad=0.04)
    shown = axes[5].imshow(np.where(valid, improvement, np.nan),
                           cmap="coolwarm", vmin=-residual_limit, vmax=residual_limit)
    axes[5].set_title("stale - propagated residual (positive=better)")
    fig.colorbar(shown, ax=axes[5], fraction=0.046, pad=0.04)
    axes[6].imshow(valid, cmap="gray", vmin=0, vmax=1)
    axes[6].set_title("valid/coverage mask")
    axes[7].imshow(valid & np.asarray(example["motion"], dtype=bool), cmap="gray", vmin=0, vmax=1)
    axes[7].set_title("motion & valid mask")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"{example['recording']} anchor={example['anchor']} offset={example['offset']} "
                 f"activity={example['activity_stratum']} ({example['activity_group']}) | "
                 f"valid tokens={example['all_valid_token_count']}, motion tokens={example['motion_token_count']}")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def window_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["recording"], row["anchor_frame_id"]), []).append(row)
    output = []
    for (recording, anchor), group in groups.items():
        item: dict[str, Any] = {"recording": recording, "anchor_frame_id": anchor,
                                "activity_stratum": group[0]["activity_stratum"],
                                "activity_group": group[0]["activity_group"],
                                "flow_activity_mean_pixels": group[0]["flow_activity_mean_pixels"],
                                "offset_count": len(group)}
        for scope in ("all_valid", "motion"):
            for method in ("stale", "propagated"):
                key = f"{scope}_{method}_cosine_distance"
                values = [row[key] for row in group if row.get(key) is not None]
                item[f"{key}_mean"] = float(np.mean(values)) if values else None
            wins = [
                row[f"{scope}_propagated_cosine_distance"] < row[f"{scope}_stale_cosine_distance"]
                for row in group if row.get(f"{scope}_propagated_cosine_distance") is not None
                and row.get(f"{scope}_stale_cosine_distance") is not None]
            item[f"{scope}_win_rate"] = float(np.mean(wins)) if wins else None
        item["motion_offset_count"] = sum(row.get("motion_stale_cosine_distance") is not None for row in group)
        item["motion_token_count"] = sum(int(row.get("motion_token_count", 0)) for row in group)
        item["coverage_ratio_mean"] = float(np.mean([row["valid_coverage_ratio"] for row in group]))
        output.append(item)
    return output


def evaluate_criteria(rows: list[dict[str, Any]], offset_summary: list[dict[str, Any]],
                      efficiency: dict[str, Any], minimum_coverage: float,
                      maximum_warp_vision_ratio: float,
                      bootstrap: dict[str, Any] | None = None) -> dict[str, Any]:
    recordings = sorted({row["recording"] for row in rows})
    motion_pairs = [(row["motion_stale_cosine_distance"], row["motion_propagated_cosine_distance"])
                    for row in rows if row.get("motion_stale_cosine_distance") is not None
                    and row.get("motion_propagated_cosine_distance") is not None]
    motion_improvement = (bool(np.mean([pair[1] for pair in motion_pairs]) < np.mean(
        [pair[0] for pair in motion_pairs])) if motion_pairs else None)
    critical = [row for row in offset_summary if row["offset"] in {3, 5, 9}]
    critical_wins = [(row["offset"], row.get("motion_cosine_win_rate")) for row in critical]
    available_critical = [value for _offset, value in critical_wins if value is not None]
    majority_critical = len(available_critical) == 3 and sum(value > 0.5 for value in available_critical) >= 2
    per_recording = {}
    for recording in recordings:
        pairs = [(row["motion_stale_cosine_distance"], row["motion_propagated_cosine_distance"])
                 for row in rows if row["recording"] == recording
                 and row.get("motion_stale_cosine_distance") is not None
                 and row.get("motion_propagated_cosine_distance") is not None]
        per_recording[recording] = (bool(np.mean([b for _a, b in pairs]) < np.mean([a for a, _b in pairs]))
                                    if pairs else None)
    coverage = float(np.mean([row["valid_coverage_ratio"] for row in rows]))
    enough_coverage = coverage >= minimum_coverage
    warp_fast = efficiency["warp_to_vision_p50_ratio"] < maximum_warp_vision_ratio
    bootstrap_groups = {item["group"]: item for item in (bootstrap or {}).get("groups", [])}
    required_bootstrap_groups = ["overall", *recordings]
    def bootstrap_group_is_positive(name: str) -> bool:
        group = bootstrap_groups.get(name, {})
        macro_low = group.get("window_macro", {}).get("ci95_low")
        micro_low = group.get("motion_token_weighted_micro", {}).get("ci95_low")
        return macro_low is not None and micro_low is not None and macro_low > 0 and micro_low > 0

    bootstrap_ci_positive = bool(bootstrap_groups) and all(
        bootstrap_group_is_positive(name) for name in required_bootstrap_groups)
    criteria = {"two_recordings_present": len(recordings) >= 2,
                "motion_cosine_improves_aggregated": motion_improvement,
                "critical_offsets_3_5_9_majority_win": bool(majority_critical),
                "critical_offset_motion_win_rates": {str(offset): value for offset, value in critical_wins},
                "improvement_consistent_by_recording": per_recording,
                "not_background_only": bool(motion_improvement is True and motion_pairs),
                "bootstrap_macro_and_micro_ci_positive_overall_and_recordings": bootstrap_ci_positive,
                "mean_coverage_ratio": coverage, "minimum_coverage_required": minimum_coverage,
                "coverage_sufficient": enough_coverage,
                "warp_vision_p50_ratio": efficiency["warp_to_vision_p50_ratio"],
                "maximum_ratio_for_significantly_faster": maximum_warp_vision_ratio,
                "warp_significantly_faster": warp_fast}
    promising = (criteria["two_recordings_present"] and motion_improvement is True and majority_critical
                 and all(value is True for value in per_recording.values())
                 and bootstrap_ci_positive and enough_coverage and warp_fast)
    return {"status": "promising" if promising else "stop", "criteria": criteria,
            "next_step": ("eligible for a separately scoped Event-estimated-flow test" if promising else
                          "do not start E-RAFT/Event; first audit feature stage, grid ordering, and warp validity; if correct, record a kill result")}


def summary_markdown(rows: list[dict[str, Any]], offset_summary: list[dict[str, Any]],
                     efficiency: dict[str, Any], decision: dict[str, Any],
                     aggregates: list[dict[str, Any]], bootstrap: dict[str, Any],
                     total_windows: int, windows_per_recording: dict[str, int]) -> str:
    def mean(key: str, subset: list[dict[str, Any]] = rows) -> float | None:
        values = [row[key] for row in subset if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    first = [row for row in rows if row["offset"] == 1]
    last = [row for row in rows if row["offset"] == max(item["offset"] for item in rows)]
    stale_worsens = (mean("all_valid_stale_cosine_distance", last) or 0) > (
        mean("all_valid_stale_cosine_distance", first) or 0)
    overall = next(item for item in aggregates if item["group_type"] == "overall")
    bootstrap_groups = {item["group"]: item for item in bootstrap["groups"]}
    return f"""# R2b expanded Qwen GT-flow oracle validation

Decision: **{decision['status']}**

Windows: total={total_windows}; per recording={json_dumps(windows_per_recording)}. Motion windows={overall['motion_window_count']}/{overall['window_count']}; motion tokens={overall['motion_token_count']}.

1. Stale feature 随 anchor age 变差：**{'yes' if stale_worsens else 'no/unclear'}**（offset 1 mean={mean('all_valid_stale_cosine_distance', first)}, final mean={mean('all_valid_stale_cosine_distance', last)}）。
2. GT-flow propagation 比 stale 更接近 dense target：all-valid cosine stale={mean('all_valid_stale_cosine_distance')}，propagated={mean('all_valid_propagated_cosine_distance')}。
3. Motion tokens：window-macro reduction={overall['window_macro_cosine_error_reduction']}；token-weighted micro reduction={overall['motion_token_micro_cosine_error_reduction']}；无 motion 的 window/offset 记为 N/A。
4. fixed-3/fixed-4 一致性：`{json_dumps(decision['criteria']['improvement_consistent_by_recording'])}`。
5. Warp coverage：mean={decision['criteria']['mean_coverage_ratio']}，门槛={decision['criteria']['minimum_coverage_required']}，足够={json_dumps(decision['criteria']['coverage_sufficient'])}。
6. Latency：ViT P50={efficiency['vision_encoder']['p50_ms']} ms，warp P50={efficiency['single_feature_warp']['p50_ms']} ms，比例={efficiency['warp_to_vision_p50_ratio']}，理论 dense/oracle speedup={efficiency['theoretical_speedup']}。
7. Window bootstrap（seed={bootstrap['seed']}，replicates={bootstrap['replicates']}）：overall macro 95% CI={bootstrap_groups['overall']['window_macro']['ci95_low']}..{bootstrap_groups['overall']['window_macro']['ci95_high']}；overall micro 95% CI={bootstrap_groups['overall']['motion_token_weighted_micro']['ci95_low']}..{bootstrap_groups['overall']['motion_token_weighted_micro']['ci95_high']}；overall/recording CI 全部为正={json_dumps(decision['criteria']['bootstrap_macro_and_micro_ci_positive_overall_and_recordings'])}。
8. 是否进入 Event-estimated flow：**{decision['next_step']}**。

## Boundary

- Dense target RGB 只用于离线 feature ground truth；stale/propagated method path 不读取 target feature 来填洞。
- GT flow 是 oracle。其读取与 warp 不是最终 Event 方法，当前效率只回答“避免重复 ViT 是否可能省计算”。
- 本实验不运行 LLM generation/QA，不训练模型，也不包含 Event encoder 或 EventFlash。
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--recording", required=True, action="append")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--windows", type=int, default=6,
                        help="total number of windows across all recordings")
    parser.add_argument("--anchor-stride", type=int, default=10)
    parser.add_argument("--max-offset", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--feature-stage", choices=("pre_merger", "post_merger"), default="post_merger")
    parser.add_argument("--motion-threshold-pixels", type=float, default=1.0)
    parser.add_argument("--minimum-coverage", type=float, default=0.5)
    parser.add_argument("--maximum-warp-vision-ratio", type=float, default=0.25)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--profile-warmup", type=int, default=3)
    parser.add_argument("--profile-repeats", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root, output_dir = args.root.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if (not root.is_dir() or args.windows <= 0 or args.max_offset <= 0 or args.anchor_stride <= 0
            or args.bootstrap_replicates <= 0):
        print("error: invalid root/windows/stride/offset", file=sys.stderr)
        return 2
    try:
        window_counts = allocate_window_counts(args.recording, args.windows)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        startup_torch, _transformers = require_runtime()
        startup_device = select_device_name(args.device, bool(startup_torch.cuda.is_available()))
    except ProbeError as exc:
        print(f"blocked: {exc}", file=sys.stderr, flush=True)
        return 2
    startup_runtime = {
        "requested_device": args.device,
        "selected_device": startup_device,
        "torch_cuda_is_available": bool(startup_torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_parameter_device": None,
        "input_tensor_device": None,
        "extracted_feature_device": None,
        "feature_stage": args.feature_stage,
        "feature_grid": None,
        "feature_shape": None,
    }
    print(json_dumps(startup_runtime), flush=True)
    atomic_write_json(output_dir / "config.json", {"status": "starting", "runtime": startup_runtime})
    indexes, selected = {}, []
    phase_start = time.perf_counter()
    print("resolving recordings", flush=True)
    try:
        for recording in args.recording:
            recording_root = find_recording(root, recording)
            rgb = index_files(recording_root / "rgb_camera", ".png")
            flow = index_files(recording_root / "optical_flow_camera", ".npz")
            indexes[recording] = {"rgb": rgb, "flow": flow, "root": recording_root}
        print(f"resolving recordings done in {time.perf_counter() - phase_start:.1f}s", flush=True)
        phase_start = time.perf_counter()
        print("sampling windows (reads and decompresses candidate flow NPZ files on CPU)", flush=True)
        for position, recording in enumerate(args.recording):
            rgb, flow = indexes[recording]["rgb"], indexes[recording]["flow"]
            chosen = select_activity_windows(rgb, flow, window_counts[recording], args.anchor_stride,
                                             args.max_offset, args.seed + position)
            selected.extend({"recording": recording, **item} for item in chosen)
        print(f"sampling windows done in {time.perf_counter() - phase_start:.1f}s", flush=True)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
                              text=True, capture_output=True, check=False).stdout.strip() or None
    data_ranges = {name: {"rgb_count": len(value["rgb"]), "rgb_first": min(value["rgb"]),
                          "rgb_last": max(value["rgb"]), "flow_count": len(value["flow"]),
                          "flow_first": min(value["flow"]), "flow_last": max(value["flow"])}
                   for name, value in indexes.items()}
    config = {"scope": "R2b expanded Qwen vision-feature + GT-flow oracle validation",
              "dataset_root": str(root), "recordings": args.recording, "model": args.model,
              "git_commit": revision, "data_ranges": data_ranges,
              "requested_device": args.device,
              "runtime": startup_runtime,
              "seed": args.seed, "windows": args.windows, "total_windows": args.windows,
              "windows_per_recording": window_counts,
              "window_allocation_rule": "balanced across recordings in CLI order; first total_windows % recording_count recordings receive one extra",
              "anchor_stride": args.anchor_stride, "max_offset": args.max_offset,
              "feature_stage": args.feature_stage, "motion_threshold_pixels": args.motion_threshold_pixels,
              "minimum_coverage": args.minimum_coverage,
              "maximum_warp_vision_ratio": args.maximum_warp_vision_ratio,
              "selection_rule": "complete stride-aligned windows ranked by mean per-pixel GT displacement across transitions; evenly select activity ranks (3 => low/mid/high); seeded tie breaking",
              "selected_windows": selected,
              "confirmed_flow_convention": {"flow_index": "flow_i is RGB_i -> RGB_i+1", "sign": 1,
                                            "background": [-1 / 1024, -1 / 1024],
                                            "scale": "half_image_size",
                                            "pixel_formula": "dx=(u-bg_u)*original_width/2; dy=(v-bg_v)*original_height/2",
                                            "feature_formula": "dx_grid=dx_pixel*feature_width/original_width; dy_grid=dy_pixel*feature_height/original_height"},
              "causality": {"method_inputs": "anchor RGB and flows up to current transition",
                            "target_rgb": "offline dense-feature ground truth only",
                            "target_feature_hole_filling": False},
              "bootstrap": {"sampling_unit": "window", "seed": args.seed,
                            "replicates": args.bootstrap_replicates},
              "profiling": {"warmup": args.profile_warmup, "repeats": args.profile_repeats}}
    atomic_write_json(output_dir / "config.json", config)

    first_window = selected[0]
    first_image = indexes[first_window["recording"]]["rgb"][first_window["anchor_frame_id"]]
    probe_path = output_dir / "probe.json"
    try:
        phase_start = time.perf_counter()
        print("loading processor/model", flush=True)
        extractor = QwenVisionExtractor(args.model, args.local_files_only, args.device)
        print(f"loading processor/model done in {time.perf_counter() - phase_start:.1f}s", flush=True)
        probe_inputs = extractor.preprocess(first_image)
        probe_feature, probe = extractor.encode_inputs(probe_inputs, args.feature_stage)
        runtime = extractor.runtime_diagnostics(probe_inputs, probe_feature, args.feature_stage)
        config["runtime"] = runtime
        atomic_write_json(output_dir / "config.json", config)
        print(json_dumps(runtime), flush=True)
        probe.update(model=args.model, image=str(first_image), local_files_only=args.local_files_only,
                     git_commit=revision, runtime=runtime)
        atomic_write_json(probe_path, probe)
        if probe["token_order"]["status"] != "confirmed_from_installed_source":
            raise TokenOrderError(probe["token_order"]["warning"])
    except (ProbeError, OSError, ValueError) as exc:
        blocked = {"status": "blocked", "model": args.model, "error": f"{type(exc).__name__}: {exc}",
                   "stop_reason": "oracle propagation was not attempted because the vision probe was not safe"}
        if probe_path.exists():
            existing = json.loads(probe_path.read_text(encoding="utf-8"))
            existing.update(blocked)
            blocked = existing
        atomic_write_json(probe_path, blocked)
        print(f"blocked: {exc}; wrote {probe_path}", file=sys.stderr)
        return 2

    torch = extractor.torch
    if extractor.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    rows, qualitative_examples = [], {}
    qualitative_anchors = select_qualitative_anchors(selected, args.recording)
    representative_feature = representative_displacement = None
    try:
        for window_index, window in enumerate(selected, 1):
            window_start = time.perf_counter()
            recording, anchor = window["recording"], window["anchor_frame_id"]
            rgb, flows = indexes[recording]["rgb"], indexes[recording]["flow"]
            validate_window_files(rgb, flows, anchor, args.max_offset)
            print(f"encoding anchor/target: window {window_index}/{len(selected)} recording={recording} "
                  f"anchor={anchor}", flush=True)
            anchor_feature = extractor.feature_grid_from_image(rgb[anchor], args.feature_stage).detach()
            extractor.runtime_diagnostics(feature=anchor_feature, stage=args.feature_stage)
            propagated = anchor_feature
            confidence = torch.ones(anchor_feature.shape[:2], dtype=torch.float32, device=anchor_feature.device)
            coverage_history = []
            print(f"recurrent warp: window {window_index}/{len(selected)} offsets=1..{args.max_offset}",
                  flush=True)
            for offset in range(1, args.max_offset + 1):
                transition = anchor + offset - 1
                raw_numpy = read_flow(flows[transition])
                raw = torch.from_numpy(raw_numpy).to(extractor.device)
                displacement = sevd_flow_to_feature_displacement(raw, anchor_feature.shape[:2], raw_numpy.shape[:2])
                sampled_flow = resize_flow_to_grid(raw, anchor_feature.shape[:2])
                pixel_displacement = sevd_flow_to_pixel_displacement(sampled_flow, raw_numpy.shape[:2])
                motion_mask = torch.linalg.vector_norm(pixel_displacement.float(), dim=-1) >= args.motion_threshold_pixels
                propagated, valid, confidence = forward_splat_torch(propagated, displacement, confidence)
                coverage = float(valid.float().mean().item())
                coverage_history.append(coverage)
                target_feature = extractor.feature_grid_from_image(rgb[anchor + offset], args.feature_stage)
                if tuple(target_feature.shape) != tuple(anchor_feature.shape):
                    raise ValueError(f"feature-grid shape changed within window {recording}/{anchor}: "
                                     f"anchor={tuple(anchor_feature.shape)}, target={tuple(target_feature.shape)}")
                stale_numpy = anchor_feature.detach().float().cpu().numpy()
                propagated_numpy = propagated.detach().float().cpu().numpy()
                target_numpy = target_feature.detach().float().cpu().numpy()
                valid_numpy = valid.detach().cpu().numpy()
                motion_numpy = motion_mask.detach().cpu().numpy()
                metrics = compare_stale_and_propagated(stale_numpy, propagated_numpy, target_numpy,
                                                       valid_numpy, motion_numpy)
                rows.append({"recording": recording, "anchor_frame_id": anchor,
                             "target_frame_id": anchor + offset, "offset": offset,
                             "activity_stratum": window["activity_stratum"],
                             "activity_group": window["activity_group"],
                             "activity_quantile": window["activity_quantile"],
                             "flow_activity_mean_pixels": window["flow_activity_mean_pixels"],
                             "current_transition_flow_frame_id": transition,
                             "flow_original_height": raw_numpy.shape[0],
                             "flow_original_width": raw_numpy.shape[1],
                             "step_coverage_ratio": coverage,
                             "coverage_history": json_dumps(coverage_history), **metrics})
                if anchor == qualitative_anchors[recording] and offset == args.max_offset:
                    original_pixel_displacement = sevd_flow_to_pixel_displacement(raw_numpy, raw_numpy.shape[:2])
                    qualitative_examples[recording] = {
                        "recording": recording, "anchor": anchor, "offset": offset,
                        "activity_stratum": window["activity_stratum"],
                        "activity_group": window["activity_group"],
                        "stale": stale_numpy, "propagated": propagated_numpy,
                        "target": target_numpy, "valid": valid_numpy, "motion": motion_numpy,
                        "all_valid_token_count": metrics["all_valid_token_count"],
                        "motion_token_count": metrics["motion_token_count"],
                        "anchor_rgb": read_rgb(rgb[anchor]),
                        "target_rgb": read_rgb(rgb[anchor + offset]),
                        "flow_magnitude_pixels": np.linalg.norm(original_pixel_displacement, axis=-1),
                    }
                if representative_feature is None:
                    representative_feature, representative_displacement = anchor_feature, displacement
            print(f"window done: recording={recording} window={window_index}/{len(selected)} anchor={anchor} "
                  f"elapsed={time.perf_counter() - window_start:.1f}s", flush=True)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"error during oracle evaluation: {exc}", file=sys.stderr)
        return 2

    phase_start = time.perf_counter()
    print("aggregating/writing results", flush=True)
    offset_summary = summarize_by_offset(rows)
    windows = window_summaries(rows)
    aggregates = grouped_motion_aggregates(rows)
    bootstrap = bootstrap_window_reductions(rows, args.bootstrap_replicates, args.seed)
    write_csv(output_dir / "per_offset_metrics.csv", rows)
    write_csv(output_dir / "window_summary.csv", windows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregates)
    atomic_write_json(output_dir / "bootstrap.json", bootstrap)
    efficiency = profile_efficiency(extractor, first_image, args.feature_stage, representative_feature,
                                    representative_displacement, args.profile_warmup, args.profile_repeats)
    atomic_write_json(output_dir / "efficiency.json", efficiency)
    decision = evaluate_criteria(rows, offset_summary, efficiency, args.minimum_coverage,
                                 args.maximum_warp_vision_ratio, bootstrap)
    results = {"offset_summary": offset_summary, "motion_aggregates": aggregates,
               "bootstrap": bootstrap, "decision": decision}
    config["results"] = results
    atomic_write_json(output_dir / "config.json", config)
    plot_metrics(offset_summary, output_dir)
    missing_qualitative = sorted(set(args.recording) - set(qualitative_examples))
    if missing_qualitative:
        raise RuntimeError(f"no high-activity qualitative example for: {missing_qualitative}")
    for recording, example in qualitative_examples.items():
        plot_qualitative(example, output_dir / f"qualitative_features_{recording}_high.png")
    plot_qualitative(qualitative_examples[args.recording[0]], output_dir / "qualitative_features.png")
    (output_dir / "summary.md").write_text(
        summary_markdown(rows, offset_summary, efficiency, decision, aggregates, bootstrap,
                         args.windows, window_counts), encoding="utf-8")
    print(f"aggregating/writing results done in {time.perf_counter() - phase_start:.1f}s", flush=True)
    print(f"wrote {output_dir}; decision={decision['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
