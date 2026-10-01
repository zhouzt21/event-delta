#!/usr/bin/env python3
"""Profile one R4a window and one prompt without changing the main benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import event_delta.qwen_injection as injection_module  # noqa: E402
from event_delta.json_io import atomic_write_json, atomic_write_jsonl  # noqa: E402
from event_delta.qwen_features import ProbeError, QwenVisionExtractor  # noqa: E402
from event_delta.qwen_injection import (  # noqa: E402
    InjectionBlocked, QwenVisualInjectionAdapter, audit_qwen_interface,
    contract_failure_message, fresh_query_inputs,
)
from event_delta.r3_propagation import event_support_mask  # noqa: E402
from event_delta.r4_diagnostics import (  # noqa: E402
    event_read_reuse_evidence, query_latency_record, repeated_work_evidence, stage_shares,
)
from event_delta.r4_efficiency import measured_stage  # noqa: E402
from tools.audit_sevd import find_recording  # noqa: E402
from tools.probe_qwen_injection import prepare_inputs  # noqa: E402
from tools.run_r2_oracle import write_csv  # noqa: E402
from tools.run_r4_vlm_efficiency import (  # noqa: E402
    eraft_once, timed_model_forward, timed_vision_forward, vision_call,
)
from tools.smoke_eraft_sevd import (  # noqa: E402
    NATIVE_HW, index_files, load_model, read_sevd_events, resize_prediction_native,
    validate_events, voxelize_events,
)


DEFAULT_PROMPT = "Describe the current traffic scene in one sentence."
QUERY_MODES = ("dense_native", "dense_cached", "stale_cached", "event_masked_cached")
TRANSITION_STAGE_KEYS = (
    "event_read_decompress_cpu_wall_ms", "voxelization_cpu_wall_ms",
    "event_cpu_to_gpu_cpu_wall_ms", "eraft_forward_cpu_wall_ms",
    "flow_gpu_to_cpu_resize_cpu_wall_ms", "support_mask_cpu_wall_ms",
    "flow_cpu_to_gpu_cpu_wall_ms", "feature_grid_conversion_cpu_wall_ms",
    "branch_forward_splat_cpu_wall_ms", "holes_fallback_cpu_wall_ms",
    "clone_copy_cpu_wall_ms", "state_object_update_cpu_wall_ms",
    "propagation_other_cpu_wall_ms",
)
DENSE_STAGE_KEYS = (
    "image_read_cpu_wall_ms", "processor_cpu_wall_ms", "rgb_cpu_to_gpu_cpu_wall_ms",
    "vision_tower_cpu_wall_ms", "state_cache_cpu_wall_ms",
)
STAGE_DEFINITIONS = {
    "event_read_decompress": "np.load of Event_i/Event_i+1 plus structured-array validation",
    "voxelization": "two independent 15-bin CPU voxel builds and normalization",
    "event_cpu_to_gpu": "torch.from_numpy plus transfer of both E-RAFT input voxels",
    "eraft_forward": "official standard zero-init E-RAFT forward, 12 iterations",
    "flow_gpu_to_cpu_resize": "prediction detach/float/device-to-host/numpy plus 640x480→1280x960 resize",
    "support_mask": "CPU Event_i/Event_i+1 support union and fixed radius-3 dilation",
    "flow_cpu_to_gpu": "native predicted flow numpy→torch transfer",
    "feature_grid_conversion": "per-visual-branch flow resize and pixel→feature-grid displacement",
    "branch_forward_splat": "per-visual-branch forward splat",
    "holes_fallback": "per-visual-branch causal previous-state hole fill",
    "clone_copy": "vision-output tree reconstruction with propagated tensors",
    "state_object_update": "VisualState allocation and metadata attachment",
    "propagation_other": "support application, reshape/loop bookkeeping, and synchronized residual",
    "image_read": "PIL file open, RGB conversion, and owned image copy",
    "processor": "chat template plus official processor on CPU",
    "rgb_cpu_to_gpu": "processor tensor transfer to selected device",
    "vision_tower": "pixel dtype conversion plus complete native vision forward",
    "state_cache": "complete output-tree inspection and VisualState construction",
    "query_input_clone": "fresh tensor clone/deepcopy and KV/rope reset before each query",
    "ttft": "query-input clone plus first multimodal prefill forward; preprocessing reported separately",
    "decode_ms_per_token": "sum of post-prefill decode forwards divided by decode forward count",
    "complete_answer": "fresh prepared inputs through normal EOS or fixed-length generation completion",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--recording", default="fixed-3")
    parser.add_argument("--anchor-frame-id", type=int, default=5581)
    parser.add_argument("--eraft-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--injection-equivalence", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", choices=("auto", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--expected-transformers-version", default="4.57.6")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def ensure_new_output_directory(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty diagnostic directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def prepare_inputs_profiled(extractor: QwenVisionExtractor, image_path: Path, prompt: str,
                            row: dict[str, Any]) -> dict[str, Any]:
    torch, device = extractor.torch, extractor.device
    with measured_stage(torch, device, row, "image_read"):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB").copy()
    with measured_stage(torch, device, row, "processor"):
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                     {"type": "text", "text": prompt}]}]
        text = extractor.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        cpu_inputs = dict(extractor.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"))
    with measured_stage(torch, device, row, "rgb_cpu_to_gpu"):
        inputs = extractor.move_inputs(cpu_inputs)
    return inputs


def dense_profile(extractor: QwenVisionExtractor, adapter: QwenVisualInjectionAdapter,
                  image_path: Path, prompt: str, timestamp: int, anchor: int,
                  audit: dict[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    torch, device = extractor.torch, extractor.device
    row: dict[str, Any] = {"frame_id": timestamp}
    inputs = prepare_inputs_profiled(extractor, image_path, prompt, row)
    with adapter.capture() as captured:
        with measured_stage(torch, device, row, "vision_tower"):
            vision_call(extractor, inputs)
    with measured_stage(torch, device, row, "state_cache"):
        state = adapter.make_state(
            captured["output"], inputs["image_grid_thw"], timestamp, anchor, audit,
            require_spatial_layout=True)

    normal_row: dict[str, Any] = {"frame_id": timestamp}
    with measured_stage(torch, device, normal_row, "normal_complete_dense"):
        normal_inputs = prepare_inputs(extractor, image_path, prompt)
        with adapter.capture() as normal_capture:
            vision_call(extractor, normal_inputs)
        adapter.make_state(normal_capture["output"], normal_inputs["image_grid_thw"],
                           timestamp, anchor, audit, require_spatial_layout=True)
    row["normal_complete_dense_cpu_wall_ms"] = normal_row["normal_complete_dense_cpu_wall_ms"]
    row["normal_complete_dense_gpu_ms"] = normal_row["normal_complete_dense_gpu_ms"]
    row["diagnostic_stage_sync_changes_execution"] = True
    return state, inputs, row


@contextmanager
def profiled_propagation_functions(torch: Any, device: Any, state: Any,
                                   record: dict[str, Any]) -> Iterator[None]:
    originals = {name: getattr(injection_module, name) for name in
                 ("predicted_flow_to_feature_displacement", "forward_splat_torch",
                  "causal_hole_fill", "_replace_tree", "VisualState")}
    branch_rows: list[dict[str, Any]] = []
    counters = {"grid": 0, "splat": 0, "holes": 0}
    replace_active = False

    def timed_branch(kind: str, original: Any, *args: Any, **kwargs: Any) -> Any:
        index = counters[kind]
        counters[kind] += 1
        path = state.spatial_tensor_paths[index] if index < len(state.spatial_tensor_paths) else None
        timing: dict[str, Any] = {"branch_path": path, "operation": kind}
        with measured_stage(torch, device, timing, kind):
            result = original(*args, **kwargs)
        branch_rows.append(timing)
        return result

    def grid_wrapper(*args: Any, **kwargs: Any) -> Any:
        return timed_branch("grid", originals["predicted_flow_to_feature_displacement"], *args, **kwargs)

    def splat_wrapper(*args: Any, **kwargs: Any) -> Any:
        return timed_branch("splat", originals["forward_splat_torch"], *args, **kwargs)

    def holes_wrapper(*args: Any, **kwargs: Any) -> Any:
        return timed_branch("holes", originals["causal_hole_fill"], *args, **kwargs)

    def replace_wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal replace_active
        if replace_active:
            return originals["_replace_tree"](*args, **kwargs)
        replace_active = True
        try:
            with measured_stage(torch, device, record, "clone_copy"):
                return originals["_replace_tree"](*args, **kwargs)
        finally:
            replace_active = False

    def state_wrapper(*args: Any, **kwargs: Any) -> Any:
        with measured_stage(torch, device, record, "state_object_update"):
            return originals["VisualState"](*args, **kwargs)

    injection_module.predicted_flow_to_feature_displacement = grid_wrapper
    injection_module.forward_splat_torch = splat_wrapper
    injection_module.causal_hole_fill = holes_wrapper
    injection_module._replace_tree = replace_wrapper
    injection_module.VisualState = state_wrapper
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(injection_module, name, original)
        record["branch_timings"] = branch_rows
        mapping = {"grid": "feature_grid_conversion", "splat": "branch_forward_splat",
                   "holes": "holes_fallback"}
        for kind, output_name in mapping.items():
            rows = [row for row in branch_rows if row["operation"] == kind]
            record[f"{output_name}_cpu_wall_ms"] = sum(row[f"{kind}_cpu_wall_ms"] for row in rows)
            gpu = [row[f"{kind}_gpu_ms"] for row in rows if row[f"{kind}_gpu_ms"] is not None]
            record[f"{output_name}_gpu_ms"] = sum(gpu) if gpu else None


def transition_profile(extractor: QwenVisionExtractor, eraft: Any, state: Any,
                       old_path: Path, new_path: Path, target: int,
                       bins: int, iterations: int) -> tuple[Any, dict[str, Any]]:
    torch, device = extractor.torch, extractor.device
    row: dict[str, Any] = {"source_frame_id": target - 1, "target_frame_id": target}
    with measured_stage(torch, device, row, "event_read_decompress"):
        old, new = read_sevd_events(old_path), read_sevd_events(new_path)
        validate_events(old, old_path, NATIVE_HW)
        validate_events(new, new_path, NATIVE_HW)
    with measured_stage(torch, device, row, "voxelization"):
        old_voxel, _ = voxelize_events(old, bins, (480, 640), NATIVE_HW)
        new_voxel, _ = voxelize_events(new, bins, (480, 640), NATIVE_HW)
    with measured_stage(torch, device, row, "event_cpu_to_gpu"):
        old_tensor = torch.from_numpy(old_voxel).to(device)
        new_tensor = torch.from_numpy(new_voxel).to(device)
    with measured_stage(torch, device, row, "eraft_forward"):
        prediction = eraft_once(torch, eraft, old_tensor, new_tensor, iterations)
    with measured_stage(torch, device, row, "flow_gpu_to_cpu_resize"):
        native = resize_prediction_native(prediction.detach().float().cpu().numpy(), NATIVE_HW)
    with measured_stage(torch, device, row, "support_mask"):
        support = event_support_mask([old, new], NATIVE_HW, dilation_radius=3)
    with measured_stage(torch, device, row, "flow_cpu_to_gpu"):
        native_tensor = torch.from_numpy(native).to(device)
    with profiled_propagation_functions(torch, device, state, row):
        with measured_stage(torch, device, row, "profiled_visual_state_update"):
            profiled_state = injection_module.propagate_visual_state(
                state, native_tensor, target, support)
    attributed = sum(float(row.get(key) or 0.0) for key in (
        "feature_grid_conversion_cpu_wall_ms", "branch_forward_splat_cpu_wall_ms",
        "holes_fallback_cpu_wall_ms", "clone_copy_cpu_wall_ms",
        "state_object_update_cpu_wall_ms"))
    row["propagation_other_cpu_wall_ms"] = max(
        0.0, row["profiled_visual_state_update_cpu_wall_ms"] - attributed)
    row["propagation_other_gpu_ms"] = None
    row["event_support_coverage"] = float(np.mean(support))
    return profiled_state, row


def transition_normal(extractor: QwenVisionExtractor, eraft: Any, state: Any,
                      old_path: Path, new_path: Path, target: int,
                      bins: int, iterations: int) -> tuple[Any, dict[str, Any]]:
    torch, device = extractor.torch, extractor.device
    row: dict[str, Any] = {"source_frame_id": target - 1, "target_frame_id": target}
    with measured_stage(torch, device, row, "normal_complete_transition"):
        old, new = read_sevd_events(old_path), read_sevd_events(new_path)
        validate_events(old, old_path, NATIVE_HW)
        validate_events(new, new_path, NATIVE_HW)
        old_voxel, _ = voxelize_events(old, bins, (480, 640), NATIVE_HW)
        new_voxel, _ = voxelize_events(new, bins, (480, 640), NATIVE_HW)
        old_tensor, new_tensor = torch.from_numpy(old_voxel).to(device), torch.from_numpy(new_voxel).to(device)
        prediction = eraft_once(torch, eraft, old_tensor, new_tensor, iterations)
        native = resize_prediction_native(prediction.detach().float().cpu().numpy(), NATIVE_HW)
        support = event_support_mask([old, new], NATIVE_HW, dilation_radius=3)
        native_tensor = torch.from_numpy(native).to(device)
        next_state = injection_module.propagate_visual_state(state, native_tensor, target, support)
    return next_state, row


def query_profile(extractor: QwenVisionExtractor, adapter: QwenVisualInjectionAdapter,
                  inputs: dict[str, Any], mode: str, state: Any | None,
                  max_new_tokens: int, fixed_length: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    torch, device = extractor.torch, extractor.device
    row: dict[str, Any] = {"mode": mode, "generation_kind": "fixed" if fixed_length else "natural"}
    calls: list[dict[str, Any]] = []
    context = adapter.capture() if mode == "dense_native" else adapter.replace(state)
    vision_context = (timed_vision_forward(extractor.structure["vision"], torch, device, row)
                      if mode == "dense_native" else nullcontext())
    with measured_stage(torch, device, row, "complete_answer"):
        with measured_stage(torch, device, row, "query_input_clone"):
            query_inputs = fresh_query_inputs(inputs)
        kwargs = {"do_sample": False, "max_new_tokens": max_new_tokens, "use_cache": True}
        if fixed_length:
            kwargs["min_new_tokens"] = max_new_tokens
        with vision_context:
            with context:
                with timed_model_forward(extractor.model, torch, device, calls):
                    with torch.inference_mode():
                        generated = extractor.model.generate(**query_inputs, **kwargs)
    prefill = [call for call in calls if call["stage"] == "multimodal_prefill"]
    decode = [call for call in calls if call["stage"] == "decode"]
    if len(prefill) != 1:
        raise RuntimeError(f"expected one prefill, got {len(prefill)}")
    prefill_ms = prefill[0]["multimodal_prefill_cpu_wall_ms"]
    decode_ms = sum(call["decode_cpu_wall_ms"] for call in decode)
    output_tokens = int(generated.shape[-1] - inputs["input_ids"].shape[-1])
    latency = query_latency_record(
        row["query_input_clone_cpu_wall_ms"] + prefill_ms, decode_ms, len(decode),
        row["complete_answer_cpu_wall_ms"], output_tokens)
    row.update(latency)
    row["prefill_cpu_wall_ms"] = prefill_ms
    row["prefill_gpu_ms"] = prefill[0]["multimodal_prefill_gpu_ms"]
    row["decode_gpu_ms"] = sum(call["decode_gpu_ms"] for call in decode)
    row["same_output_token_target"] = max_new_tokens if fixed_length else None
    generation = {"mode": mode, "generation_kind": row["generation_kind"],
                  "token_ids": generated[0].tolist(), "output_token_count": output_tokens,
                  "text": extractor.processor.decode(generated[0], skip_special_tokens=True)}
    return row, generation


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = ensure_new_output_directory(args.output_dir)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
                              text=True, capture_output=True, check=False).stdout.strip() or None
    config = {
        "scope": "R4a one-window efficiency diagnostic", "git_commit": revision,
        "recording": args.recording, "anchor_frame_id": args.anchor_frame_id,
        "window_offsets": list(range(1, 10)), "prompt": args.prompt,
        "natural_max_new_tokens": args.max_new_tokens,
        "fixed_new_tokens": args.max_new_tokens, "diagnostic_replays_per_transition": 2,
        "warmup_transition_replays": 1,
        "main_benchmark_definition_changed": False, "status": "starting",
    }
    atomic_write_json(output / "config.json", config)
    try:
        if (args.bins, args.iterations, args.max_new_tokens) != (15, 12, 32):
            raise ValueError("R4a diagnostic preserves bins=15, iterations=12, max_new_tokens=32")
        equivalence = json.loads(args.injection_equivalence.read_text(encoding="utf-8"))
        if equivalence.get("status") != "passed" or equivalence.get("model") != args.model:
            raise InjectionBlocked("matching injection equivalence artifact must be passed")
        extractor = QwenVisionExtractor(args.model, args.local_files_only, args.device)
        torch, device = extractor.torch, extractor.device
        if extractor.transformers.__version__ != args.expected_transformers_version:
            raise InjectionBlocked("Transformers version mismatch")
        if device.type != "cuda":
            raise RuntimeError("the synchronized efficiency diagnostic requires CUDA")
        audit = audit_qwen_interface(extractor.model, extractor.structure["vision"],
                                     extractor.processor, extractor.order,
                                     extractor.transformers.__version__)
        if audit["status"] != "ready":
            raise InjectionBlocked(contract_failure_message(audit))
        eraft, checkpoint_meta = load_model(torch, args.eraft_root.resolve(),
                                            args.checkpoint.resolve(), args.bins, "standard", device)
        recording_root = find_recording(args.root.resolve(), args.recording).resolve()
        rgb = index_files(recording_root / "rgb_camera", ".png")
        events = index_files(recording_root / "dvs_camera", ".npz", "-xytp.npz")
        required = range(args.anchor_frame_id, args.anchor_frame_id + 10)
        if any(frame not in rgb or frame not in events for frame in required):
            raise FileNotFoundError("diagnostic window lacks RGB or Event frames")
        adapter = QwenVisualInjectionAdapter(extractor.model, extractor.structure["vision"],
                                             extractor.merge_size, extractor.order)

        anchor_state, anchor_inputs, anchor_dense = dense_profile(
            extractor, adapter, rgb[args.anchor_frame_id], args.prompt,
            args.anchor_frame_id, args.anchor_frame_id, audit)
        target_id = args.anchor_frame_id + 9
        dense_target_state, target_inputs, target_dense = dense_profile(
            extractor, adapter, rgb[target_id], args.prompt, target_id,
            args.anchor_frame_id, audit)

        state = anchor_state
        transition_rows = []
        _warm_state, _warm_timing = transition_normal(
            extractor, eraft, anchor_state, events[args.anchor_frame_id],
            events[args.anchor_frame_id + 1], args.anchor_frame_id + 1,
            args.bins, args.iterations)
        for offset in range(1, 10):
            source, target = args.anchor_frame_id + offset - 1, args.anchor_frame_id + offset
            previous_state = state
            state, normal = transition_normal(
                extractor, eraft, previous_state, events[source], events[target], target,
                args.bins, args.iterations)
            _profiled_state, profile = transition_profile(
                extractor, eraft, previous_state, events[source], events[target], target,
                args.bins, args.iterations)
            profile.update(offset=offset,
                           normal_complete_transition_cpu_wall_ms=normal[
                               "normal_complete_transition_cpu_wall_ms"],
                           normal_complete_transition_gpu_ms=normal[
                               "normal_complete_transition_gpu_ms"],
                           diagnostic_stage_sync_changes_execution=True)
            transition_rows.append(profile)
        event_state = state

        cached_anchor_inputs = prepare_inputs(extractor, rgb[args.anchor_frame_id], args.prompt)
        query_rows, generations = [], []
        for fixed in (False, True):
            for mode in QUERY_MODES:
                if mode == "dense_native":
                    inputs, query_state = target_inputs, None
                elif mode == "dense_cached":
                    inputs, query_state = target_inputs, dense_target_state
                elif mode == "stale_cached":
                    inputs, query_state = cached_anchor_inputs, anchor_state
                else:
                    inputs, query_state = cached_anchor_inputs, event_state
                row, generation = query_profile(
                    extractor, adapter, inputs, mode, query_state,
                    args.max_new_tokens, fixed)
                query_rows.append(row)
                generations.append(generation)

        for row in transition_rows:
            row["stage_shares"] = stage_shares(
                row, TRANSITION_STAGE_KEYS, "normal_complete_transition_cpu_wall_ms")
        for row in (anchor_dense, target_dense):
            row["stage_shares"] = stage_shares(
                row, DENSE_STAGE_KEYS, "normal_complete_dense_cpu_wall_ms")
        transition_aggregate = {
            "normal_complete_transition_cpu_wall_ms": sum(
                row["normal_complete_transition_cpu_wall_ms"] for row in transition_rows),
            **{key: sum(float(row.get(key) or 0.0) for row in transition_rows)
               for key in TRANSITION_STAGE_KEYS},
        }
        transition_aggregate["stage_shares"] = stage_shares(
            transition_aggregate, TRANSITION_STAGE_KEYS,
            "normal_complete_transition_cpu_wall_ms")
        dense_aggregate = {
            "normal_complete_dense_cpu_wall_ms": sum(
                row["normal_complete_dense_cpu_wall_ms"] for row in (anchor_dense, target_dense)),
            **{key: sum(float(row.get(key) or 0.0) for row in (anchor_dense, target_dense))
               for key in DENSE_STAGE_KEYS},
        }
        dense_aggregate["stage_shares"] = stage_shares(
            dense_aggregate, DENSE_STAGE_KEYS, "normal_complete_dense_cpu_wall_ms")
        fixed_counts = [row["output_token_count"] for row in query_rows
                        if row["generation_kind"] == "fixed"]
        reuse = event_read_reuse_evidence(args.anchor_frame_id, 9)
        reuse["voxelization_repeated_for_shared_adjacent_packets"] = True
        repeated_queries = repeated_work_evidence(0, 0, True)
        largest = sorted(
            ({"offset": row["offset"], **item} for row in transition_rows
             for item in row["stage_shares"]),
            key=lambda item: item["cpu_wall_ms"], reverse=True)[:10]
        diagnostic = {
            "cost_boundaries": {
                "normal_complete_transition": "single uninterrupted causal transition; one outer synchronization",
                "profiled_transition_stages": "separate replay with synchronization around each stage; not summed as the primary total",
                "warmup": "one discarded replay of the first transition before all reported transition timings",
                "normal_complete_dense": "image open through cached VisualState construction",
                "query": "prepared processor tensors through complete generation; RGB preprocessing is reported separately",
                "model_and_checkpoint_loading_excluded": True,
            },
            "transition_stage_keys": list(TRANSITION_STAGE_KEYS),
            "dense_stage_keys": list(DENSE_STAGE_KEYS),
            "stage_definitions": STAGE_DEFINITIONS,
            "transition_aggregate": transition_aggregate,
            "dense_aggregate": dense_aggregate,
            "largest_observed_profiled_stages": largest,
            "event_packet_reuse": reuse,
            "repeated_query_work": repeated_queries,
            "fixed_output_token_count_matches": len(set(fixed_counts)) == 1,
            "fixed_output_token_counts": fixed_counts,
            "recommendations_not_applied": [
                "Cache the shared Event packet/voxel between adjacent transitions and remeasure.",
                "Evaluate a torch-native flow resize/support path to remove the GPU→CPU→GPU round trip.",
                "Use branch timings to determine whether serial forward splats or hole fallback dominate before optimizing.",
                "Retain fresh per-query KV/rope state; only optimize tensor cloning after proving it is material.",
            ],
            "experiment_definition_changed": False,
        }
        branch_rows = [{"offset": row["offset"], **branch}
                       for row in transition_rows for branch in row["branch_timings"]]
        write_csv(output / "transition_timings.csv", transition_rows)
        write_csv(output / "branch_propagation_timings.csv", branch_rows)
        write_csv(output / "dense_timings.csv", [anchor_dense, target_dense])
        write_csv(output / "query_timings.csv", query_rows)
        atomic_write_jsonl(output / "generations.jsonl", generations)
        atomic_write_json(output / "efficiency_diagnostic.json", diagnostic)
        atomic_write_json(output / "environment.json", {
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "transformers": extractor.transformers.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(device), "checkpoint": checkpoint_meta,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        })
        config["status"] = "completed"
        atomic_write_json(output / "config.json", config)
        top = largest[0] if largest else None
        (output / "summary.md").write_text(
            "# R4a one-window efficiency diagnostic\n\n"
            f"- Window: {args.recording}/{args.anchor_frame_id}, offsets 1..9; one prompt.\n"
            "- Primary transition totals are independently measured normal paths; synchronized stage "
            "timings come from separate diagnostic replays.\n"
            f"- Largest profiled stage sample: {top}.\n"
            f"- Event packet reads per nine-transition replay: {reuse['total_packet_reads_without_cache']}; "
            f"unique packets: {reuse['unique_event_packets']}; repeated reads: {reuse['repeated_packet_reads']}.\n"
            "- Repeated queries reuse completed visual states and perform no Event reads/rebuilds, but clone "
            "fresh query tensors to isolate KV/rope state.\n"
            f"- Fixed-length output counts: {fixed_counts}; identical={len(set(fixed_counts)) == 1}.\n"
            "- No optimization was applied. Recommendations in efficiency_diagnostic.json are hypotheses "
            "for a later controlled change.\n",
            encoding="utf-8")
    except (OSError, ValueError, RuntimeError, ProbeError, InjectionBlocked, KeyError) as exc:
        config.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(output / "config.json", config)
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
