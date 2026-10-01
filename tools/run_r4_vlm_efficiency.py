#!/usr/bin/env python3
"""Run the fixed R4a full-VLM visual-state injection and efficiency probe."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from event_delta.json_io import atomic_write_json, atomic_write_jsonl  # noqa: E402
from event_delta.qwen_features import ProbeError, QwenVisionExtractor  # noqa: E402
from event_delta.qwen_injection import (  # noqa: E402
    InjectionBlocked, QwenVisualInjectionAdapter, audit_qwen_interface,
    fresh_query_inputs, propagate_visual_state,
)
from event_delta.r3_propagation import event_support_mask  # noqa: E402
from event_delta.r4_efficiency import (  # noqa: E402
    REQUIRED_STAGES, interleaved_schedule, measured_stage, percentile_summary,
    r4_decision, validate_complete_timing,
)
from tools.audit_sevd import find_recording  # noqa: E402
from tools.probe_qwen_injection import prepare_inputs  # noqa: E402
from tools.run_r2_oracle import write_csv  # noqa: E402
from tools.smoke_eraft_sevd import (  # noqa: E402
    NATIVE_HW, index_files, load_model, read_sevd_events, require_prediction,
    resize_prediction_native, validate_events, voxelize_events,
)


PROMPTS = (
    "Describe the current traffic scene in one sentence.",
    "Where are the visible vehicles located?",
    "Is a motorcycle visible in the scene?",
)
MODES = ("dense_native", "dense_cached", "stale_cached", "event_masked_cached")
OFFSETS = (1, 5, 9)


def blank_stage_row() -> dict[str, Any]:
    return {f"{stage}_{kind}_ms": 0.0 if kind == "cpu_wall" else None
            for stage in REQUIRED_STAGES for kind in ("cpu_wall", "gpu")}


def add_timing(total: dict[str, Any], current: dict[str, Any], stage: str) -> None:
    for kind in ("cpu_wall", "gpu"):
        key = f"{stage}_{kind}_ms"
        if current.get(key) is not None:
            total[key] = float(total.get(key) or 0.0) + float(current[key])


def vision_call(extractor: QwenVisionExtractor, inputs: dict[str, Any]) -> Any:
    vision = extractor.structure["vision"]
    parameters = inspect.signature(vision.forward).parameters
    kwargs: dict[str, Any] = {}
    pixels = inputs["pixel_values"].to(dtype=extractor.torch.bfloat16)
    if "pixel_values" in parameters:
        kwargs["pixel_values"] = pixels
    elif "hidden_states" in parameters:
        kwargs["hidden_states"] = pixels
    else:
        raise InjectionBlocked(f"unsupported vision input signature: {list(parameters)}")
    if "grid_thw" in parameters:
        kwargs["grid_thw"] = inputs["image_grid_thw"]
    elif "image_grid_thw" in parameters:
        kwargs["image_grid_thw"] = inputs["image_grid_thw"]
    else:
        raise InjectionBlocked("vision forward has no grid argument")
    with extractor.torch.inference_mode():
        return vision(**kwargs)


def capture_state(extractor: QwenVisionExtractor, adapter: QwenVisualInjectionAdapter,
                  inputs: dict[str, Any], timestamp: int, anchor: int,
                  audit: dict[str, Any], timings: dict[str, Any] | None = None) -> Any:
    target = timings if timings is not None else {}
    with adapter.capture() as captured:
        with measured_stage(extractor.torch, extractor.device, target, "anchor_vision_encoding"):
            vision_call(extractor, inputs)
    return adapter.make_state(captured["output"], inputs["image_grid_thw"], timestamp, anchor, audit)


def eraft_once(torch: Any, model: Any, old: Any, new: Any, iterations: int) -> Any:
    with torch.inference_mode():
        _low, predictions = model(image1=old, image2=new, iters=iterations, flow_init=None)
    if not isinstance(predictions, (list, tuple)) or len(predictions) != iterations:
        raise ValueError(f"E-RAFT must return {iterations} predictions")
    return require_prediction(predictions[-1])


def build_event_states(extractor: QwenVisionExtractor, eraft: Any, anchor_state: Any,
                       events: dict[int, Path], anchor: int, max_offset: int,
                       bins: int, iterations: int) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    torch, device = extractor.torch, extractor.device
    state, selected, timing_rows = anchor_state, {}, []
    for offset in range(1, max_offset + 1):
        source, target = anchor + offset - 1, anchor + offset
        timing = blank_stage_row()
        with measured_stage(torch, device, timing, "event_npz_decode"):
            old, new = read_sevd_events(events[source]), read_sevd_events(events[target])
            validate_events(old, events[source], NATIVE_HW)
            validate_events(new, events[target], NATIVE_HW)
        with measured_stage(torch, device, timing, "voxelization"):
            old_voxel, _ = voxelize_events(old, bins, (480, 640), NATIVE_HW)
            new_voxel, _ = voxelize_events(new, bins, (480, 640), NATIVE_HW)
            old_tensor = torch.from_numpy(old_voxel).to(device)
            new_tensor = torch.from_numpy(new_voxel).to(device)
        with measured_stage(torch, device, timing, "eraft_forward"):
            prediction = eraft_once(torch, eraft, old_tensor, new_tensor, iterations)
        with measured_stage(torch, device, timing, "support_mask_grid_conversion"):
            native = resize_prediction_native(prediction.detach().float().cpu().numpy(), NATIVE_HW)
            support = event_support_mask([old, new], NATIVE_HW, dilation_radius=3)
            native_tensor = torch.from_numpy(native).to(device)
        with measured_stage(torch, device, timing, "visual_branch_propagation"):
            state = propagate_visual_state(state, native_tensor, target, support)
        timing.update(offset=offset, source_frame_id=source, target_frame_id=target,
                      state_age_frames=0, event_support_coverage=float(np.mean(support)))
        timing_rows.append(timing)
        if offset in OFFSETS:
            selected[offset] = state
    return selected, timing_rows


@contextmanager
def timed_model_forward(model: Any, torch: Any, device: Any,
                        calls: list[dict[str, Any]]) -> Iterator[None]:
    original = model.forward

    def replacement(*args: Any, **kwargs: Any) -> Any:
        record: dict[str, Any] = {}
        stage = "multimodal_prefill" if not calls else "decode"
        with measured_stage(torch, device, record, stage):
            result = original(*args, **kwargs)
        calls.append({"stage": stage, **record})
        return result

    replacement.__signature__ = inspect.signature(original)  # type: ignore[attr-defined]
    model.forward = replacement
    try:
        yield
    finally:
        model.forward = original


@contextmanager
def timed_vision_forward(vision: Any, torch: Any, device: Any,
                         row: dict[str, Any]) -> Iterator[None]:
    original = vision.forward

    def replacement(*args: Any, **kwargs: Any) -> Any:
        with measured_stage(torch, device, row, "anchor_vision_encoding"):
            return original(*args, **kwargs)

    replacement.__signature__ = inspect.signature(original)  # type: ignore[attr-defined]
    vision.forward = replacement
    try:
        yield
    finally:
        vision.forward = original


def run_query(extractor: QwenVisionExtractor, adapter: QwenVisualInjectionAdapter,
              image: Path, prompt: str, mode: str, state: Any | None,
              max_new_tokens: int, prepared_inputs: dict[str, Any] | None = None,
              ) -> tuple[dict[str, Any], dict[str, Any]]:
    torch, device = extractor.torch, extractor.device
    row = blank_stage_row()
    with measured_stage(torch, device, row, "complete_query"):
        with measured_stage(torch, device, row, "rgb_decode_preprocess"):
            inputs = (fresh_query_inputs(prepared_inputs) if prepared_inputs is not None
                      else prepare_inputs(extractor, image, prompt))
        calls: list[dict[str, Any]] = []
        before = dict(adapter.counters)
        context = adapter.capture() if mode == "dense_native" else adapter.replace(state)
        vision_context = (timed_vision_forward(extractor.structure["vision"], torch, device, row)
                          if mode == "dense_native" else nullcontext())
        with vision_context:
            with context:
                with timed_model_forward(extractor.model, torch, device, calls):
                    with torch.inference_mode():
                        generated = extractor.model.generate(
                            **fresh_query_inputs(inputs), do_sample=False,
                            max_new_tokens=max_new_tokens, use_cache=True)
        after = dict(adapter.counters)
    prefill = [call for call in calls if call["stage"] == "multimodal_prefill"]
    decode = [call for call in calls if call["stage"] == "decode"]
    if len(prefill) != 1:
        raise RuntimeError(f"expected exactly one prefill forward, got {len(prefill)}")
    for stage_calls, stage in ((prefill, "multimodal_prefill"), (decode, "decode")):
        row[f"{stage}_cpu_wall_ms"] = sum(call[f"{stage}_cpu_wall_ms"] for call in stage_calls)
        gpu = [call[f"{stage}_gpu_ms"] for call in stage_calls if call[f"{stage}_gpu_ms"] is not None]
        row[f"{stage}_gpu_ms"] = sum(gpu) if gpu else None
    input_length = int(inputs["input_ids"].shape[-1])
    output_tokens = max(0, int(generated.shape[-1]) - input_length)
    row["visual_injection_cpu_wall_ms"] = (after["cached_replacement_cpu_wall_ms"]
                                             - before["cached_replacement_cpu_wall_ms"])
    row["visual_injection_gpu_ms"] = None
    row.update(actual_output_token_count=output_tokens,
               decode_tokens_per_second=(output_tokens / (row["decode_cpu_wall_ms"] / 1000.0)
                                         if row["decode_cpu_wall_ms"] > 0 else None),
               ttft_excluding_visual_update_ms=(row["rgb_decode_preprocess_cpu_wall_ms"]
                                                + row["multimodal_prefill_cpu_wall_ms"]),
               vision_forward_calls=after["vision_forward_call_count"] - before["vision_forward_call_count"],
               actual_vision_computations=(after["actual_vision_computation_count"]
                                           - before["actual_vision_computation_count"]),
               cached_replacements=after["cached_replacement_count"] - before["cached_replacement_count"],
               complete_query_derived_from_stage_percentiles=False,
               processor_image_tensors_cached=prepared_inputs is not None,
               fresh_past_key_values=True)
    validate_complete_timing(row)
    generation = {"mode": mode, "prompt": prompt, "token_ids": generated[0].tolist(),
                  "text": extractor.processor.decode(generated[0], skip_special_tokens=True),
                  "actual_output_token_count": output_tokens}
    return row, generation


def summarize_efficiency(rows: list[dict[str, Any]], maintenance: list[dict[str, Any]],
                         dense_video_rows: list[dict[str, Any]], anchor_timings: list[dict[str, Any]],
                         torch: Any, device: Any) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for mode in MODES:
        subset = [row for row in rows if row["mode"] == mode and not row["warmup"]]
        groups[mode] = {key: percentile_summary(subset, key) for key in
                        ("complete_query_cpu_wall_ms", "multimodal_prefill_cpu_wall_ms",
                         "decode_cpu_wall_ms", "ttft_including_available_update_ms")}
    transition_totals = [sum(row[f"{stage}_cpu_wall_ms"] for stage in
                             ("event_npz_decode", "voxelization", "eraft_forward",
                              "support_mask_grid_conversion", "visual_branch_propagation"))
                         for row in maintenance]
    dense_by_repeat = [sum(row["rgb_decode_preprocess_cpu_wall_ms"]
                           + row["anchor_vision_encoding_cpu_wall_ms"]
                           for row in dense_video_rows if row["repeat"] == repeat)
                       for repeat in sorted({row["repeat"] for row in dense_video_rows})]
    event_by_repeat = []
    for repeat in sorted({row["repeat"] for row in maintenance}):
        anchor_row = next(row for row in anchor_timings if row["repeat"] == repeat)
        transitions = [row for row in maintenance if row["repeat"] == repeat]
        event_by_repeat.append(anchor_row["rgb_decode_preprocess_cpu_wall_ms"]
                               + anchor_row["anchor_vision_encoding_cpu_wall_ms"]
                               + sum(sum(row[f"{stage}_cpu_wall_ms"] for stage in
                                         ("event_npz_decode", "voxelization", "eraft_forward",
                                          "support_mask_grid_conversion", "visual_branch_propagation"))
                                     for row in transitions))
    dense_video_total = float(np.mean(dense_by_repeat))
    event_video_total = float(np.mean(event_by_repeat))
    return {"query": groups, "maintenance_transition_ms": {
                "count": len(transition_totals), "sum": float(np.sum(transition_totals)),
                "mean": float(np.mean(transition_totals)) if transition_totals else None,
                "p50": float(np.percentile(transition_totals, 50)) if transition_totals else None,
                "p95": float(np.percentile(transition_totals, 95)) if transition_totals else None},
            "ten_hz_cadence_ms": 100.0,
            "keeps_up_observed_transitions": bool(transition_totals and max(transition_totals) <= 100.0),
            "transition_lag_ms": [max(0.0, value - 100.0) for value in transition_totals],
            "maximum_transition_lag_ms": (max(max(0.0, value - 100.0) for value in transition_totals)
                                          if transition_totals else None),
            "slow_samples_dropped": False,
            "async_scheduler_implemented": False, "real_concurrency_claim": False,
            "one_second_video_cost": {
                "dense_10_rgb_actual_accumulated_ms": dense_video_total,
                "event_1_anchor_plus_9_transitions_actual_accumulated_ms": event_video_total,
                "assembled_from_stage_percentiles": False,
                "steady_state_repeats": len(dense_by_repeat),
                "dense_rgb_encode_count_per_repeat": 10,
                "event_transition_count_per_repeat": 9,
                "dense_per_repeat_totals_ms": dense_by_repeat,
                "event_per_repeat_totals_ms": event_by_repeat},
            "peak_gpu_allocated_bytes": (int(torch.cuda.max_memory_allocated(device))
                                         if device.type == "cuda" else None),
            "peak_gpu_reserved_bytes": (int(torch.cuda.max_memory_reserved(device))
                                        if device.type == "cuda" else None)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--recording", default="fixed-3")
    parser.add_argument("--anchor-frame-id", type=int, default=5581)
    parser.add_argument("--eraft-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--injection-equivalence", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--expected-transformers-version", default="4.57.6")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
                              text=True, capture_output=True, check=False).stdout.strip() or None
    config = {"scope": "R4a Qwen VLM visual-state efficiency probe", "git_commit": revision,
              "model": args.model, "recording": args.recording, "anchor_frame_id": args.anchor_frame_id,
              "offsets": list(OFFSETS), "prompts": list(PROMPTS), "modes": list(MODES),
              "warmup": args.warmup, "repeats": args.repeats, "batch_size": 1,
              "expected_transformers_version": args.expected_transformers_version,
              "generation": {"greedy": True, "max_new_tokens": args.max_new_tokens},
              "event_support": {"packets": "Event_i union Event_i+1", "dilation_radius": 3},
              "fresh_multimodal_prefill_per_query": True, "incremental_kv_update": False,
              "status": "starting"}
    atomic_write_json(output / "config.json", config)
    try:
        equivalence = json.loads(args.injection_equivalence.read_text(encoding="utf-8"))
        if equivalence.get("status") != "passed":
            raise InjectionBlocked("injection correctness did not pass; Event benchmark is forbidden")
        if equivalence.get("model") != args.model:
            raise InjectionBlocked("equivalence artifact model does not match benchmark model")
        if (args.recording, args.anchor_frame_id, args.warmup, args.repeats,
                args.bins, args.iterations, args.max_new_tokens) != ("fixed-3", 5581, 3, 20, 15, 12, 32):
            raise ValueError("R4a protocol is fixed to fixed-3/5581, bins=15, iterations=12, "
                             "warmup=3, repeats=20, max_new_tokens=32")
        extractor = QwenVisionExtractor(args.model, args.local_files_only, args.device)
        torch, device = extractor.torch, extractor.device
        if extractor.transformers.__version__ != args.expected_transformers_version:
            raise InjectionBlocked(f"Transformers version mismatch: expected "
                                   f"{args.expected_transformers_version}, got "
                                   f"{extractor.transformers.__version__}")
        if device.type != "cuda":
            raise RuntimeError("R4a efficiency benchmark requires CUDA for separate GPU timing")
        eraft, checkpoint_meta = load_model(torch, args.eraft_root.resolve(), args.checkpoint.resolve(),
                                            args.bins, "standard", device)
        recording_root = find_recording(args.root.resolve(), args.recording).resolve()
        rgb = index_files(recording_root / "rgb_camera", ".png")
        events = index_files(recording_root / "dvs_camera", ".npz", "-xytp.npz")
        required_ids = range(args.anchor_frame_id, args.anchor_frame_id + 10)
        if any(frame not in rgb or frame not in events for frame in required_ids):
            raise FileNotFoundError("R4a anchor window lacks RGB or Event frames")
        audit = audit_qwen_interface(extractor.model, extractor.structure["vision"],
                                     extractor.processor, extractor.order)
        if audit["status"] != "ready":
            raise InjectionBlocked(audit.get("warning", "installed interface audit blocked"))
        adapter = QwenVisualInjectionAdapter(extractor.model, extractor.structure["vision"],
                                             extractor.merge_size, extractor.order)
        event_states, maintenance, anchor_timings = {}, [], []
        dense_states, dense_video_rows = {}, []
        for cycle in range(args.warmup + args.repeats):
            warmup_cycle = cycle < args.warmup
            repeat = cycle - args.warmup
            anchor_timing = blank_stage_row()
            with measured_stage(torch, device, anchor_timing, "rgb_decode_preprocess"):
                anchor_inputs = prepare_inputs(extractor, rgb[args.anchor_frame_id], PROMPTS[0])
            anchor_state = capture_state(extractor, adapter, anchor_inputs, args.anchor_frame_id,
                                         args.anchor_frame_id, audit, anchor_timing)
            adapter.validate_placeholder_contract(anchor_inputs, anchor_state)
            current_event_states, current_maintenance = build_event_states(
                extractor, eraft, anchor_state, events, args.anchor_frame_id, 9,
                args.bins, args.iterations)
            event_states = current_event_states
            if not warmup_cycle:
                anchor_timing["repeat"] = repeat
                anchor_timings.append(anchor_timing)
                for item in current_maintenance:
                    item["repeat"] = repeat
                    maintenance.append(item)
            for dense_offset in range(10):
                dense_timing = blank_stage_row()
                with measured_stage(torch, device, dense_timing, "rgb_decode_preprocess"):
                    inputs = prepare_inputs(extractor, rgb[args.anchor_frame_id + dense_offset], PROMPTS[0])
                dense_state = capture_state(extractor, adapter, inputs,
                                            args.anchor_frame_id + dense_offset,
                                            args.anchor_frame_id, audit, dense_timing)
                dense_timing.update(offset=dense_offset, repeat=repeat)
                if not warmup_cycle:
                    dense_video_rows.append(dense_timing)
                if dense_offset in OFFSETS:
                    dense_states[dense_offset] = dense_state
        if any(state.deepstack_paths != anchor_state.deepstack_paths for state in dense_states.values()):
            raise InjectionBlocked("DeepStack return structure changed across frames")
        rows, generations = [], []
        total_cycles = args.warmup + args.repeats
        for offset in OFFSETS:
            target = args.anchor_frame_id + offset
            for prompt_index, prompt in enumerate(PROMPTS):
                cached_target_inputs = prepare_inputs(extractor, rgb[target], prompt)
                cached_anchor_inputs = prepare_inputs(extractor, rgb[args.anchor_frame_id], prompt)
                schedule = interleaved_schedule(list(MODES), total_cycles)
                for sequence_index, mode in enumerate(schedule):
                    repeat = sequence_index // len(MODES)
                    warmup = repeat < args.warmup
                    steady_repeat = None if warmup else repeat - args.warmup
                    if mode == "dense_native":
                        image, state, prepared = rgb[target], None, None
                    elif mode == "dense_cached":
                        image, state, prepared = rgb[target], dense_states[offset], cached_target_inputs
                    elif mode == "stale_cached":
                        image, state, prepared = rgb[args.anchor_frame_id], anchor_state, cached_anchor_inputs
                    else:
                        image, state, prepared = rgb[args.anchor_frame_id], event_states[offset], cached_anchor_inputs
                    row, generation = run_query(extractor, adapter, image, prompt, mode, state,
                                                args.max_new_tokens, prepared)
                    maintenance_repeat = 0 if steady_repeat is None else steady_repeat
                    update_rows = ([item for item in maintenance
                                    if item["repeat"] == maintenance_repeat and item["offset"] <= offset]
                                   if mode == "event_masked_cached" else [])
                    update_ms = sum(sum(item[f"{stage}_cpu_wall_ms"] for stage in
                                        ("event_npz_decode", "voxelization", "eraft_forward",
                                         "support_mask_grid_conversion", "visual_branch_propagation"))
                                    for item in update_rows)
                    for stage in ("event_npz_decode", "voxelization", "eraft_forward",
                                  "support_mask_grid_conversion", "visual_branch_propagation"):
                        row[f"on_demand_{stage}_cpu_wall_ms"] = sum(
                            item[f"{stage}_cpu_wall_ms"] for item in update_rows)
                        gpu_values = [item[f"{stage}_gpu_ms"] for item in update_rows
                                      if item[f"{stage}_gpu_ms"] is not None]
                        row[f"on_demand_{stage}_gpu_ms"] = sum(gpu_values) if gpu_values else None
                    row.update(mode=mode, offset=offset, target_frame_id=target, prompt_index=prompt_index,
                               repeat=repeat, warmup=warmup, scenario="maintained_state_query",
                               steady_state_repeat=steady_repeat,
                               execution_order=sequence_index,
                               state_age_frames=(offset if mode == "stale_cached" else 0),
                               background_maintenance_cost_ms=update_ms,
                               on_demand_complete_path_ms=row["complete_query_cpu_wall_ms"] + update_ms,
                               on_demand_ttft_ms=row["ttft_excluding_visual_update_ms"] + update_ms,
                               ttft_including_available_update_ms=row["ttft_excluding_visual_update_ms"],
                               final_visual_token_count=(state.token_count if state is not None
                                                         else anchor_state.token_count),
                               deepstack_layer_count=(len(state.deepstack_paths) if state is not None
                                                      else len(anchor_state.deepstack_paths)),
                               visual_state_source=("target_rgb" if mode.startswith("dense") else
                                                    "anchor_rgb" if mode == "stale_cached" else "causal_events"),
                               target_rgb_used_in_online_path=mode.startswith("dense"))
                    rows.append(row)
                    generation.update(offset=offset, target_frame_id=target, prompt_index=prompt_index,
                                      repeat=repeat, warmup=warmup)
                    generations.append(generation)
        efficiency = summarize_efficiency(rows, maintenance, dense_video_rows, anchor_timings, torch, device)
        dense_query = efficiency["query"]["dense_cached"]["complete_query_cpu_wall_ms"]["mean"]
        event_query = efficiency["query"]["event_masked_cached"]["complete_query_cpu_wall_ms"]["mean"]
        video_cost = efficiency["one_second_video_cost"]
        dense_cost = (video_cost["dense_10_rgb_actual_accumulated_ms"] + dense_query
                      if dense_query is not None else None)
        event_cost = (video_cost["event_1_anchor_plus_9_transitions_actual_accumulated_ms"] + event_query
                      if event_query is not None else None)
        decision = r4_decision(True, dense_cost, event_cost)
        efficiency["decision"] = decision
        efficiency["cost_boundary"] = {
            "query_latency_and_maintenance_separated": True,
            "one_second_dense_reference": "10 dense RGB vision encodes; measured query table retains each full run",
            "one_second_event_maintenance": "1 anchor vision encode + 9 observed Event transitions",
            "totals_use_observed_samples_not_sum_of_stage_p50": True,
            "visual_token_count_reduced": False, "llm_prefill_or_kv_reduced": False}
        write_csv(output / "per_run_metrics.csv", rows)
        atomic_write_jsonl(output / "generations.jsonl", generations)
        atomic_write_json(output / "efficiency.json", efficiency)
        atomic_write_json(output / "visual_state_diagnostics.json", {
            "interface_audit": audit, "anchor": anchor_state.diagnostics(),
            "event_states": {str(offset): state.diagnostics() for offset, state in event_states.items()},
            "dense_states": {str(offset): state.diagnostics() for offset, state in dense_states.items()},
            "vision_counters": adapter.counters})
        atomic_write_json(output / "injection_equivalence.json", equivalence)
        environment = {"python": platform.python_version(), "numpy": np.__version__,
                       "torch": torch.__version__, "transformers": extractor.transformers.__version__,
                       "device": str(device), "gpu": torch.cuda.get_device_name(device),
                       "checkpoint": checkpoint_meta, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        atomic_write_json(output / "environment.json", environment)
        config.update(status=decision["status"], decision=decision)
        atomic_write_json(output / "config.json", config)
        (output / "summary.md").write_text(summary_markdown(equivalence, anchor_state, efficiency),
                                            encoding="utf-8")
    except (OSError, ValueError, RuntimeError, ProbeError, InjectionBlocked, KeyError) as exc:
        config.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(output / "config.json", config)
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {output}; decision={config['status']}")
    return 0


def summary_markdown(equivalence: dict[str, Any], anchor_state: Any,
                     efficiency: dict[str, Any]) -> str:
    decision = efficiency["decision"]
    return f"""# R4a Qwen visual-state injection and efficiency probe

- Complete visual injection equivalent: **{equivalence.get('status') == 'passed'}**.
- Event-updated state entered the language model through full native vision-output replacement: **yes**.
- DeepStack present: **{bool(anchor_state.deepstack_paths)}**; every verified spatial branch was recurrently propagated. Final-only injection was not used.
- Decision: **{decision['status']}**. Dense cost={decision.get('dense_cost_ms')} ms; Event query plus observed transition cost={decision.get('event_cost_ms')} ms.
- Dense versus Event: Event avoids current-RGB vision encoding but pays Event decode, voxelization, E-RAFT, support/grid conversion, and all-branch propagation.
- Stale is cheaper but older; Event is not unconditionally better than stale.
- Maintained-state query latency and background maintenance cost are reported separately. No asynchronous scheduler was implemented.
- Keeps up with 10 Hz in observed transitions: **{efficiency['keeps_up_observed_transitions']}**. This is not a claim of a concurrent real-time system.
- Visual token count changes: **no**. This probe targets vision-encoding cost, not LLM prefill or KV-cache reduction.
- No formal downstream VLM accuracy evaluation was performed; the three prompts are functional/efficiency checks only.
"""


if __name__ == "__main__":
    raise SystemExit(main())
