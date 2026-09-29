#!/usr/bin/env python3
"""Run the minimal Qwen vision-feature + SEVD GT-flow oracle R2a smoke test."""

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
from event_delta.metrics import compare_stale_and_propagated, summarize_by_offset  # noqa: E402
from event_delta.qwen_features import ProbeError, QwenVisionExtractor, TokenOrderError  # noqa: E402
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
    labels = (["low", "mid", "high"] if count == 3
              else [f"activity_quantile_{index / max(count - 1, 1):.2f}" for index in range(count)])
    return [{"anchor_frame_id": scored[position][2], "flow_activity_mean_pixels": scored[position][0],
             "activity_stratum": label, "activity_rank": position + 1,
             "candidate_window_count": len(scored)} for position, label in zip(positions, labels)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cuda_timings(torch: Any, operation: Callable[[], Any], warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return {"warmup": warmup, "repeats": repeats, "milliseconds": timings,
            "p50_ms": float(np.percentile(timings, 50)), "p95_ms": float(np.percentile(timings, 95))}


def profile_efficiency(extractor: QwenVisionExtractor, image_path: Path, stage: str,
                       feature: Any, displacement: Any, warmup: int, repeats: int) -> dict[str, Any]:
    torch = extractor.torch
    inputs = extractor.preprocess(image_path)
    vision = cuda_timings(torch, lambda: extractor.feature_grid_from_inputs(inputs, stage), warmup, repeats)
    confidence = torch.ones(feature.shape[:2], dtype=torch.float32, device=feature.device)
    warp = cuda_timings(torch, lambda: forward_splat_torch(feature, displacement, confidence), warmup, repeats)

    def nine_steps() -> Any:
        current, current_confidence = feature, confidence
        for _ in range(9):
            current, _valid, current_confidence = forward_splat_torch(current, displacement, current_confidence)
        return current

    recurrent = cuda_timings(torch, nine_steps, warmup, repeats)
    dense_cost = 10 * vision["p50_ms"]
    propagation_cost = vision["p50_ms"] + 9 * warp["p50_ms"]
    return {"timing_method": "CUDA events after warmup; torch.cuda.synchronize after every measurement",
            "timing_scope": "vision-module GPU forward excludes image decode/processor and disk I/O; warp excludes flow disk I/O",
            "vision_encoder": vision, "single_feature_warp": warp, "recurrent_nine_step_warp": recurrent,
            "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
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
    panels = (np.linalg.norm(stale, axis=-1), np.linalg.norm(propagated, axis=-1),
              np.linalg.norm(target, axis=-1), np.linalg.norm(propagated - target, axis=-1), example["valid"],
              example["motion"])
    titles = ("stale feature norm", "propagated feature norm", "target feature norm",
              "propagated residual L2", "valid/coverage mask", "motion-token mask")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for axis, panel, title in zip(axes.reshape(-1), panels, titles):
        shown = axis.imshow(panel, cmap="magma")
        fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{example['recording']} anchor={example['anchor']} offset={example['offset']}")
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
        item["coverage_ratio_mean"] = float(np.mean([row["valid_coverage_ratio"] for row in group]))
        output.append(item)
    return output


def evaluate_criteria(rows: list[dict[str, Any]], offset_summary: list[dict[str, Any]],
                      efficiency: dict[str, Any], minimum_coverage: float,
                      maximum_warp_vision_ratio: float) -> dict[str, Any]:
    recordings = sorted({row["recording"] for row in rows})
    motion_pairs = [(row["motion_stale_cosine_distance"], row["motion_propagated_cosine_distance"])
                    for row in rows if row.get("motion_stale_cosine_distance") is not None
                    and row.get("motion_propagated_cosine_distance") is not None]
    motion_improvement = bool(motion_pairs) and np.mean([pair[1] for pair in motion_pairs]) < np.mean(
        [pair[0] for pair in motion_pairs])
    critical = [row for row in offset_summary if row["offset"] in {3, 5, 9}]
    critical_wins = [row.get("motion_cosine_win_rate", 0) > 0.5 for row in critical]
    majority_critical = len(critical) == 3 and sum(critical_wins) >= 2
    per_recording = {}
    for recording in recordings:
        pairs = [(row["motion_stale_cosine_distance"], row["motion_propagated_cosine_distance"])
                 for row in rows if row["recording"] == recording
                 and row.get("motion_stale_cosine_distance") is not None
                 and row.get("motion_propagated_cosine_distance") is not None]
        per_recording[recording] = bool(pairs) and np.mean([b for _a, b in pairs]) < np.mean([a for a, _b in pairs])
    coverage = float(np.mean([row["valid_coverage_ratio"] for row in rows]))
    enough_coverage = coverage >= minimum_coverage
    warp_fast = efficiency["warp_to_vision_p50_ratio"] < maximum_warp_vision_ratio
    criteria = {"two_recordings_present": len(recordings) >= 2,
                "motion_cosine_improves_aggregated": bool(motion_improvement),
                "critical_offsets_3_5_9_majority_win": bool(majority_critical),
                "improvement_consistent_by_recording": per_recording,
                "not_background_only": bool(motion_improvement and motion_pairs),
                "mean_coverage_ratio": coverage, "minimum_coverage_required": minimum_coverage,
                "coverage_sufficient": enough_coverage,
                "warp_vision_p50_ratio": efficiency["warp_to_vision_p50_ratio"],
                "maximum_ratio_for_significantly_faster": maximum_warp_vision_ratio,
                "warp_significantly_faster": warp_fast}
    promising = (criteria["two_recordings_present"] and motion_improvement and majority_critical
                 and all(per_recording.values()) and enough_coverage and warp_fast)
    return {"status": "promising" if promising else "stop", "criteria": criteria,
            "next_step": ("eligible for a separately scoped Event-estimated-flow test" if promising else
                          "do not start E-RAFT/Event; first audit feature stage, grid ordering, and warp validity; if correct, record a kill result")}


def summary_markdown(rows: list[dict[str, Any]], offset_summary: list[dict[str, Any]],
                     efficiency: dict[str, Any], decision: dict[str, Any]) -> str:
    def mean(key: str, subset: list[dict[str, Any]] = rows) -> float | None:
        values = [row[key] for row in subset if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    first = [row for row in rows if row["offset"] == 1]
    last = [row for row in rows if row["offset"] == max(item["offset"] for item in rows)]
    stale_worsens = (mean("all_valid_stale_cosine_distance", last) or 0) > (
        mean("all_valid_stale_cosine_distance", first) or 0)
    return f"""# R2a Qwen GT-flow oracle propagation smoke test

Decision: **{decision['status']}**

1. Stale feature 随 anchor age 变差：**{'yes' if stale_worsens else 'no/unclear'}**（offset 1 mean={mean('all_valid_stale_cosine_distance', first)}, final mean={mean('all_valid_stale_cosine_distance', last)}）。
2. GT-flow propagation 比 stale 更接近 dense target：all-valid cosine stale={mean('all_valid_stale_cosine_distance')}，propagated={mean('all_valid_propagated_cosine_distance')}。
3. Motion tokens：stale={mean('motion_stale_cosine_distance')}，propagated={mean('motion_propagated_cosine_distance')}；聚合改善={decision['criteria']['motion_cosine_improves_aggregated']}。
4. fixed-3/fixed-4 一致性：`{decision['criteria']['improvement_consistent_by_recording']}`。
5. Warp coverage：mean={decision['criteria']['mean_coverage_ratio']}，门槛={decision['criteria']['minimum_coverage_required']}，足够={decision['criteria']['coverage_sufficient']}。
6. Latency：ViT P50={efficiency['vision_encoder']['p50_ms']} ms，warp P50={efficiency['single_feature_warp']['p50_ms']} ms，比例={efficiency['warp_to_vision_p50_ratio']}，理论 dense/oracle speedup={efficiency['theoretical_speedup']}。
7. 是否进入 Event-estimated flow：**{decision['next_step']}**。

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
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--anchor-stride", type=int, default=10)
    parser.add_argument("--max-offset", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--feature-stage", choices=("pre_merger", "post_merger"), default="post_merger")
    parser.add_argument("--motion-threshold-pixels", type=float, default=1.0)
    parser.add_argument("--minimum-coverage", type=float, default=0.5)
    parser.add_argument("--maximum-warp-vision-ratio", type=float, default=0.25)
    parser.add_argument("--profile-warmup", type=int, default=3)
    parser.add_argument("--profile-repeats", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root, output_dir = args.root.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if not root.is_dir() or args.windows <= 0 or args.max_offset <= 0 or args.anchor_stride <= 0:
        print("error: invalid root/windows/stride/offset", file=sys.stderr)
        return 2
    if args.windows % len(args.recording):
        print("error: --windows must divide evenly across recordings", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    per_recording_count = args.windows // len(args.recording)
    indexes, selected = {}, []
    try:
        for position, recording in enumerate(args.recording):
            recording_root = find_recording(root, recording)
            rgb = index_files(recording_root / "rgb_camera", ".png")
            flow = index_files(recording_root / "optical_flow_camera", ".npz")
            indexes[recording] = {"rgb": rgb, "flow": flow, "root": recording_root}
            chosen = select_activity_windows(rgb, flow, per_recording_count, args.anchor_stride,
                                             args.max_offset, args.seed + position)
            selected.extend({"recording": recording, **item} for item in chosen)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
                              text=True, capture_output=True, check=False).stdout.strip() or None
    data_ranges = {name: {"rgb_count": len(value["rgb"]), "rgb_first": min(value["rgb"]),
                          "rgb_last": max(value["rgb"]), "flow_count": len(value["flow"]),
                          "flow_first": min(value["flow"]), "flow_last": max(value["flow"])}
                   for name, value in indexes.items()}
    config = {"scope": "R2a Qwen vision-feature + GT-flow oracle propagation smoke test",
              "dataset_root": str(root), "recordings": args.recording, "model": args.model,
              "git_commit": revision, "data_ranges": data_ranges,
              "seed": args.seed, "windows": args.windows, "windows_per_recording": per_recording_count,
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
              "profiling": {"warmup": args.profile_warmup, "repeats": args.profile_repeats}}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    first_window = selected[0]
    first_image = indexes[first_window["recording"]]["rgb"][first_window["anchor_frame_id"]]
    probe_path = output_dir / "probe.json"
    try:
        extractor = QwenVisionExtractor(args.model, args.local_files_only, "cuda")
        probe = extractor.probe_image(first_image)
        probe.update(model=args.model, image=str(first_image), local_files_only=args.local_files_only,
                     git_commit=revision)
        probe_path.write_text(json.dumps(probe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if probe["token_order"]["status"] != "confirmed_from_installed_source":
            raise TokenOrderError(probe["token_order"]["warning"])
    except (ProbeError, OSError, ValueError) as exc:
        blocked = {"status": "blocked", "model": args.model, "error": f"{type(exc).__name__}: {exc}",
                   "stop_reason": "oracle propagation was not attempted because the vision probe was not safe"}
        if probe_path.exists():
            existing = json.loads(probe_path.read_text(encoding="utf-8"))
            existing.update(blocked)
            blocked = existing
        probe_path.write_text(json.dumps(blocked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"blocked: {exc}; wrote {probe_path}", file=sys.stderr)
        return 2

    torch = extractor.torch
    torch.cuda.reset_peak_memory_stats()
    rows, qualitative = [], None
    representative_feature = representative_displacement = None
    try:
        for window in selected:
            recording, anchor = window["recording"], window["anchor_frame_id"]
            rgb, flows = indexes[recording]["rgb"], indexes[recording]["flow"]
            validate_window_files(rgb, flows, anchor, args.max_offset)
            anchor_feature = extractor.feature_grid_from_image(rgb[anchor], args.feature_stage).detach()
            propagated = anchor_feature
            confidence = torch.ones(anchor_feature.shape[:2], dtype=torch.float32, device=anchor_feature.device)
            coverage_history = []
            for offset in range(1, args.max_offset + 1):
                transition = anchor + offset - 1
                raw_numpy = read_flow(flows[transition])
                raw = torch.from_numpy(raw_numpy).to(anchor_feature.device)
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
                             "flow_activity_mean_pixels": window["flow_activity_mean_pixels"],
                             "current_transition_flow_frame_id": transition,
                             "flow_original_height": raw_numpy.shape[0],
                             "flow_original_width": raw_numpy.shape[1],
                             "step_coverage_ratio": coverage,
                             "coverage_history": json.dumps(coverage_history), **metrics})
                if qualitative is None and offset == args.max_offset:
                    qualitative = {"recording": recording, "anchor": anchor, "offset": offset,
                                   "stale": stale_numpy, "propagated": propagated_numpy,
                                   "target": target_numpy, "valid": valid_numpy, "motion": motion_numpy}
                if representative_feature is None:
                    representative_feature, representative_displacement = anchor_feature, displacement
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"error during oracle evaluation: {exc}", file=sys.stderr)
        return 2

    offset_summary = summarize_by_offset(rows)
    windows = window_summaries(rows)
    write_csv(output_dir / "per_offset_metrics.csv", rows)
    write_csv(output_dir / "window_summary.csv", windows)
    efficiency = profile_efficiency(extractor, first_image, args.feature_stage, representative_feature,
                                    representative_displacement, args.profile_warmup, args.profile_repeats)
    (output_dir / "efficiency.json").write_text(json.dumps(efficiency, indent=2) + "\n", encoding="utf-8")
    decision = evaluate_criteria(rows, offset_summary, efficiency, args.minimum_coverage,
                                 args.maximum_warp_vision_ratio)
    results = {"offset_summary": offset_summary, "decision": decision}
    config["results"] = results
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_metrics(offset_summary, output_dir)
    if qualitative is None:
        raise RuntimeError("no qualitative example was produced")
    plot_qualitative(qualitative, output_dir / "qualitative_features.png")
    (output_dir / "summary.md").write_text(summary_markdown(rows, offset_summary, efficiency, decision),
                                             encoding="utf-8")
    print(f"wrote {output_dir}; decision={decision['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
