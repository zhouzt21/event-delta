#!/usr/bin/env python3
"""Probe exact full-Qwen visual-state capture and injection on one RGB/prompt."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from event_delta.json_io import atomic_write_json, atomic_write_jsonl  # noqa: E402
from event_delta.qwen_features import QwenVisionExtractor  # noqa: E402
from event_delta.qwen_injection import (  # noqa: E402
    InjectionBlocked,
    QwenVisualInjectionAdapter,
    audit_qwen_interface,
    compare_logits,
    contract_failure_message,
    fresh_query_inputs,
    injection_passes,
    tensor_tree_summary,
)


DEFAULT_PROMPT = "Describe the current traffic scene in one sentence."


def clone_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return fresh_query_inputs(inputs)


def prepare_inputs(extractor: QwenVisionExtractor, image: Path, prompt: str) -> dict[str, Any]:
    with Image.open(image) as opened:
        rgb = opened.convert("RGB").copy()
    messages = [{"role": "user", "content": [{"type": "image", "image": rgb},
                                                 {"type": "text", "text": prompt}]}]
    processor = extractor.processor
    if not hasattr(processor, "apply_chat_template"):
        raise InjectionBlocked("processor lacks apply_chat_template required for a full VLM probe")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[rgb], padding=True, return_tensors="pt")
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs or "input_ids" not in inputs:
        raise InjectionBlocked(f"processor omitted required multimodal tensors: {list(inputs.keys())}")
    return extractor.move_inputs(dict(inputs))


def forward_once(extractor: QwenVisionExtractor, inputs: dict[str, Any]) -> Any:
    with extractor.torch.inference_mode():
        return extractor.model(**clone_inputs(inputs), use_cache=True, return_dict=True)


def generate_once(extractor: QwenVisionExtractor, inputs: dict[str, Any], max_new_tokens: int) -> Any:
    with extractor.torch.inference_mode():
        return extractor.model.generate(**clone_inputs(inputs), do_sample=False,
                                        max_new_tokens=max_new_tokens, use_cache=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--expected-transformers-version", default="4.57.6")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-absolute-tolerance", type=float, default=1e-5)
    parser.add_argument("--mean-absolute-tolerance", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
                              text=True, capture_output=True, check=False).stdout.strip() or None
    config = {"scope": "R4a full-Qwen visual-state injection equivalence",
              "git_commit": revision, "model": args.model, "image": str(args.image.resolve()),
              "prompt": args.prompt, "device": args.device, "local_files_only": args.local_files_only,
              "expected_transformers_version": args.expected_transformers_version,
              "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                             "fresh_past_key_values_per_query": True},
              "tolerance": {"max_absolute_difference": args.max_absolute_tolerance,
                            "mean_absolute_difference": args.mean_absolute_tolerance,
                            "automatic_relaxation": False},
              "serial_only": True, "status": "starting"}
    atomic_write_json(output_dir / "config.json", config)
    audit_path = output_dir / "source_contract_audit.json"
    audit: dict[str, Any] = {
        "status": "blocked", "phase": "startup", "transformers_version": None,
        "actual_classes": {}, "inspected_methods": {}, "checks": [],
        "failed_required_checks": ["runtime_initialization"],
        "failure_summary": ["runtime_initialization: audit has not started"],
        "evidence": {},
        "vision_return_structure": {"runtime_observed": False, "reason": "runtime has not started"},
        "runtime_vision_return": {"observed": False, "reason": "runtime has not started"},
        "transformers_site_packages_modified": False, "global_monkey_patch": False,
    }
    atomic_write_json(audit_path, audit)
    failure_stage = "argument_validation"
    if not args.image.is_file() or args.max_new_tokens <= 0:
        audit.update(phase="argument_validation",
                     failure_summary=["argument_validation: image is missing or max-new-tokens is invalid"])
        atomic_write_json(audit_path, audit)
        print("error: image is missing or max-new-tokens is invalid", file=sys.stderr)
        return 2
    try:
        failure_stage = "model_initialization"
        extractor = QwenVisionExtractor(args.model, args.local_files_only, args.device)
        torch = extractor.torch
        audit.update(transformers_version=extractor.transformers.__version__, phase="version_validation",
                     actual_classes={"model": extractor.model.__class__.__name__,
                                     "vision": extractor.structure["vision"].__class__.__name__,
                                     "processor": extractor.processor.__class__.__name__})
        atomic_write_json(audit_path, audit)
        failure_stage = "transformers_version_validation"
        if extractor.transformers.__version__ != args.expected_transformers_version:
            raise InjectionBlocked(f"Transformers version mismatch: expected "
                                   f"{args.expected_transformers_version}, got "
                                   f"{extractor.transformers.__version__}")
        failure_stage = "source_contract_audit"
        audit = audit_qwen_interface(extractor.model, extractor.structure["vision"],
                                     extractor.processor, extractor.order,
                                     extractor.transformers.__version__)
        audit["phase"] = "source_contract"
        atomic_write_json(audit_path, audit)
        if audit["status"] != "ready":
            raise InjectionBlocked(contract_failure_message(audit))
        failure_stage = "input_preparation"
        inputs = prepare_inputs(extractor, args.image.resolve(), args.prompt)
        adapter = QwenVisualInjectionAdapter(extractor.model, extractor.structure["vision"],
                                             extractor.merge_size, extractor.order)
        failure_stage = "native_vision_capture"
        with adapter.capture() as first_capture:
            native_a = forward_once(extractor, inputs)
        failure_stage = "visual_state_construction"
        state = adapter.make_state(first_capture["output"], inputs["image_grid_thw"],
                                   state_timestamp="same_image", anchor_timestamp="same_image",
                                   source_audit=audit, require_spatial_layout=False)
        failure_stage = "visual_state_diagnostics"
        state_diagnostics = state.diagnostics()
        audit["runtime_vision_return"] = {
            "observed": True,
            "tensor_tree": tensor_tree_summary(state.vision_output),
            **state.path_diagnostics(),
        }
        audit["vision_return_structure"].update(
            runtime_observed=True,
            final_embedding_path=state.final_path,
            deepstack_paths=list(state.deepstack_paths),
            tensor_tree=tensor_tree_summary(state.vision_output),
        )
        audit["phase"] = "runtime_vision_return_observed"
        atomic_write_json(audit_path, audit)
        failure_stage = "placeholder_contract_validation"
        placeholder = adapter.validate_placeholder_contract(inputs, state)
        failure_stage = "native_repeat_forward"
        with adapter.capture():
            native_b = forward_once(extractor, inputs)
        before_actual = adapter.counters["actual_vision_computation_count"]
        failure_stage = "cached_injected_forward"
        with adapter.replace(state):
            injected = forward_once(extractor, inputs)
        failure_stage = "native_generation"
        bypass_forward = adapter.counters["actual_vision_computation_count"] == before_actual
        with adapter.capture():
            native_ids = generate_once(extractor, inputs, args.max_new_tokens)
        before_actual = adapter.counters["actual_vision_computation_count"]
        failure_stage = "cached_injected_generation"
        with adapter.replace(state):
            injected_ids = generate_once(extractor, inputs, args.max_new_tokens)
        bypass_generate = adapter.counters["actual_vision_computation_count"] == before_actual
        metrics = compare_logits(native_a.logits, native_b.logits, injected.logits)
        rope_contract = {}
        for name in ("rope_deltas", "position_ids"):
            native_value, injected_value = getattr(native_a, name, None), getattr(injected, name, None)
            if native_value is None and injected_value is None:
                rope_contract[name] = {"exposed_by_output": False, "matches": None}
            elif native_value is None or injected_value is None:
                rope_contract[name] = {"exposed_by_output": True, "matches": False}
            else:
                rope_contract[name] = {"exposed_by_output": True,
                                       "shape": list(native_value.shape),
                                       "matches": bool(torch.equal(native_value, injected_value))}
        passed = injection_passes(metrics, native_ids, injected_ids,
                                  args.max_absolute_tolerance, args.mean_absolute_tolerance,
                                  bypass_forward and bypass_generate)
        if any(item["matches"] is False for item in rope_contract.values()):
            passed = False
        output_rope = {name: str(getattr(native_a, name, None)) for name in ("rope_deltas", "position_ids")}
        equivalence = {
            "status": "passed" if passed else "blocked", "metrics": metrics,
            "model": args.model, "source_image": str(args.image.resolve()),
            "native_repeat_measured_before_tolerance_decision": True,
            "tolerance": config["tolerance"], "next_token_argmax_matches":
                metrics["native_next_token_argmax"] == metrics["injected_next_token_argmax"],
            "generated_token_ids_match": bool((native_ids == injected_ids).all().item()),
            "native_generated_token_ids": native_ids[0].tolist(),
            "injected_generated_token_ids": injected_ids[0].tolist(),
            "cache_decode_completed": True, "vision_bypassed_forward": bypass_forward,
            "vision_bypassed_generate": bypass_generate, "counters": adapter.counters,
            "placeholder_contract": placeholder, "processor_input_keys": sorted(inputs),
            "visual_mask_input_keys": sorted(key for key in inputs
                                              if "mask" in key and key != "attention_mask"),
            "input_ids_shape": list(inputs["input_ids"].shape),
            "attention_mask_shape": list(inputs["attention_mask"].shape),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "runtime_position_outputs": output_rope,
            "runtime_position_contract": rope_contract,
            "same_input_ids_attention_mask_and_grid": True,
            "fresh_query_cache": True,
        }
        diagnostics = {"interface_audit": audit, "visual_state": state_diagnostics,
                       "vision_return_tree": tensor_tree_summary(state.vision_output),
                       "deepstack_complete": (not audit["evidence"]["deepstack_in_source"]
                                              or bool(state.deepstack_paths)),
                       "final_only_ablation_used": False}
        generations = [{"mode": "native", "token_ids": native_ids[0].tolist(),
                        "text": extractor.processor.decode(native_ids[0], skip_special_tokens=True)},
                       {"mode": "injected", "token_ids": injected_ids[0].tolist(),
                        "text": extractor.processor.decode(injected_ids[0], skip_special_tokens=True)}]
        environment = {"python": platform.python_version(), "torch": torch.__version__,
                       "transformers": extractor.transformers.__version__, "device": str(extractor.device),
                       "cuda_available": bool(torch.cuda.is_available()),
                       "gpu": torch.cuda.get_device_name(extractor.device) if extractor.device.type == "cuda" else None,
                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                       "attention_implementation": getattr(extractor.model.config, "_attn_implementation", None)}
        atomic_write_json(output_dir / "environment.json", environment)
        atomic_write_json(output_dir / "injection_equivalence.json", equivalence)
        atomic_write_json(output_dir / "visual_state_diagnostics.json", diagnostics)
        atomic_write_jsonl(output_dir / "generations.jsonl", generations)
        config["status"] = equivalence["status"]
        atomic_write_json(output_dir / "config.json", config)
        if not passed:
            print("blocked: same-image cached injection is not numerically equivalent", file=sys.stderr)
            return 2
    except Exception as exc:
        formatted_traceback = traceback.format_exc()
        traceback.print_exc(file=sys.stderr)
        audit["status"] = "blocked"
        audit["terminal_error"] = f"{type(exc).__name__}: {exc}"
        audit["failure_stage"] = failure_stage
        if not audit.get("failure_summary") or audit.get("phase") != "source_contract":
            audit["failure_summary"] = audit.get("failure_summary", []) + [audit["terminal_error"]]
        atomic_write_json(audit_path, audit)
        config.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
        config["failure_stage"] = failure_stage
        atomic_write_json(output_dir / "config.json", config)
        atomic_write_json(output_dir / "injection_equivalence.json",
                          {"status": "blocked", "error": config["error"],
                           "failure_stage": failure_stage, "traceback": formatted_traceback})
        print(f"blocked at {failure_stage}: {exc}", file=sys.stderr)
        return 2
    print(f"injection equivalence passed; wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
