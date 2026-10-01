"""Local, reversible Qwen visual-state capture, replacement and propagation.

The adapter deliberately learns the installed model contract at runtime.  It does
not patch Transformers globally and never claims equivalence before the probe.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from .flow_warp import forward_splat_torch
from .qwen_features import ProbeError, tensor_summary
from .r3_propagation import predicted_flow_to_feature_displacement


class InjectionBlocked(ProbeError):
    """The installed model contract is not safe for visual-state injection."""


def causal_hole_fill(warped: Any, previous: Any, valid: Any) -> Any:
    """Fill holes only from the prior state's same spatial position."""
    if tuple(warped.shape) != tuple(previous.shape) or tuple(valid.shape) != tuple(warped.shape[:2]):
        raise ValueError("causal hole-fill shape contract failed")
    if warped.__class__.__module__.startswith("torch"):
        import torch
        return torch.where(valid[..., None], warped, previous)
    return np.where(np.asarray(valid, dtype=bool)[..., None], warped, previous)


def fresh_query_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Clone a query while explicitly removing cross-query cache/rope state."""
    output = {key: (value.clone() if hasattr(value, "clone") else copy.deepcopy(value))
              for key, value in inputs.items()
              if key not in {"past_key_values", "cache_position", "rope_deltas"}}
    output["past_key_values"] = None
    return output


def _source(value: Any) -> str:
    return _source_record(value).pop("source_text")


def _safe_signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _source_record(value: Any) -> dict[str, Any]:
    source, source_error = "", None
    if value is None:
        source_error = "callable missing"
    try:
        source = inspect.getsource(value) if value is not None else ""
    except (OSError, TypeError) as exc:
        source_error = f"{type(exc).__name__}: {exc}"
    try:
        path = inspect.getsourcefile(value) if value is not None else None
    except TypeError as exc:
        path = None
        source_error = source_error or f"{type(exc).__name__}: {exc}"
    owner = getattr(value, "__self__", None)
    owner_class = owner if inspect.isclass(owner) else getattr(owner, "__class__", None)
    return {"owner_class": (owner_class.__name__ if owner_class is not None else None),
            "method": getattr(value, "__name__", None),
            "qualified_name": getattr(value, "__qualname__", None),
            "file": path, "sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
            "line_count": len(source.splitlines()), "signature": _safe_signature(value),
            "inspect_getsource_succeeded": bool(source), "source_available": bool(source),
            "source_error": source_error, "source_text": source}


def tree_tensors(value: Any, path: str = "output") -> list[tuple[str, Any]]:
    output: list[tuple[str, Any]] = []
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return [(path, value)]
    if isinstance(value, dict):
        for key, item in value.items():
            output.extend(tree_tensors(item, f"{path}.{key}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            output.extend(tree_tensors(item, f"{path}[{index}]"))
    return output


def _replace_tree(value: Any, replacements: dict[str, Any], path: str = "output") -> Any:
    if path in replacements:
        return replacements[path]
    if isinstance(value, dict):
        result = copy.copy(value)
        for key, item in value.items():
            result[key] = _replace_tree(item, replacements, f"{path}.{key}")
        return result
    if isinstance(value, list):
        return [_replace_tree(item, replacements, f"{path}[{index}]")
                for index, item in enumerate(value)]
    if isinstance(value, tuple):
        items = [_replace_tree(item, replacements, f"{path}[{index}]")
                 for index, item in enumerate(value)]
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return tuple(items)
    return value


def _token_axis(tensor: Any, token_count: int) -> int | None:
    shape = tuple(int(x) for x in tensor.shape)
    matches = [axis for axis, size in enumerate(shape[:-1]) if size == token_count]
    if len(matches) != 1:
        return None
    axis = matches[0]
    if len(shape) == 2 and axis == 0:
        return axis
    if len(shape) == 3 and shape[0] == 1 and axis == 1:
        return axis
    return None


@dataclass
class VisualState:
    """Complete native vision output plus verified spatial/cache metadata."""

    vision_output: Any
    image_grid_thw: tuple[int, int, int]
    spatial_grid_hw: tuple[int, int]
    merge_size: int
    final_path: str
    spatial_tensor_paths: tuple[str, ...]
    deepstack_paths: tuple[str, ...]
    branch_grid_hw: dict[str, tuple[int, int]]
    dtype: str
    device: str
    state_timestamp: int | float | str
    anchor_timestamp: int | float | str
    layout_verified: bool
    validity: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return self.spatial_grid_hw[0] * self.spatial_grid_hw[1]

    def diagnostics(self) -> dict[str, Any]:
        tensors = dict(tree_tensors(self.vision_output))
        return {
            "image_grid_thw": list(self.image_grid_thw), "spatial_grid_hw": list(self.spatial_grid_hw),
            "merge_size": self.merge_size, "token_count": self.token_count,
            "final_path": self.final_path, "spatial_tensor_paths": list(self.spatial_tensor_paths),
            "deepstack_present": bool(self.deepstack_paths),
            "deepstack_paths": list(self.deepstack_paths),
            "branch_grid_hw": {path: list(hw) for path, hw in self.branch_grid_hw.items()},
            "visual_tensor_shapes": {path: list(tensor.shape) for path, tensor in tensors.items()},
            "dtype": self.dtype, "device": self.device,
            "state_timestamp": self.state_timestamp, "anchor_timestamp": self.anchor_timestamp,
            "layout_verified": self.layout_verified,
            "validity": {path: {"shape": list(mask.shape),
                                 "valid_ratio": float(mask.float().mean().item())}
                         for path, mask in self.validity.items()},
            "metadata": self.metadata,
        }


def _check(name: str, required: bool, passed: bool, evidence: list[str],
           failure_kind: str | None, failure_reason: str | None) -> dict[str, Any]:
    return {"name": name, "required": required, "passed": bool(passed), "evidence": evidence,
            "failure_kind": None if passed else failure_kind,
            "failure_reason": None if passed else failure_reason}


def audit_qwen_interface(model: Any, vision: Any, processor: Any, order: dict[str, Any],
                         transformers_version: str | None = None) -> dict[str, Any]:
    """Audit installed source using callable structure, signatures and semantic alternatives."""
    base_model = getattr(model, "model", None)
    methods: dict[str, Any] = {
        "conditional_generation.forward": getattr(model, "forward", None),
        "conditional_generation.get_image_features": getattr(model, "get_image_features", None),
        "conditional_generation.get_placeholder_mask": getattr(model, "get_placeholder_mask", None),
        "conditional_generation.prepare_inputs_for_generation": getattr(
            model, "prepare_inputs_for_generation", None),
        "base_model.forward": getattr(base_model, "forward", None),
        "base_model.get_rope_index": next((getattr(base_model, name) for name in
                                            ("get_rope_index", "_get_rope_index")
                                            if base_model is not None and hasattr(base_model, name)), None),
        "conditional_generation.get_rope_index": next((getattr(model, name) for name in
                                                        ("get_rope_index", "_get_rope_index")
                                                        if hasattr(model, name)), None),
        "vision.forward": getattr(vision, "forward", None),
        "processor.apply_chat_template": getattr(processor, "apply_chat_template", None),
    }
    records = {name: _source_record(value) for name, value in methods.items()}
    sources = {name: record.pop("source_text") for name, record in records.items()}
    lower = {name: source.lower() for name, source in sources.items()}

    model_forward = lower["conditional_generation.forward"]
    image_features = lower["conditional_generation.get_image_features"]
    placeholder = lower["conditional_generation.get_placeholder_mask"]
    prepare = lower["conditional_generation.prepare_inputs_for_generation"]
    base_forward = lower["base_model.forward"]
    rope_sources = "\n".join((lower["base_model.get_rope_index"],
                               lower["conditional_generation.get_rope_index"], base_forward, model_forward))
    vision_source = lower["vision.forward"]
    combined = "\n".join(lower.values())

    model_signature = records["conditional_generation.forward"]["signature"] or ""
    vision_signature = records["vision.forward"]["signature"] or ""
    prepare_signature = records["conditional_generation.prepare_inputs_for_generation"]["signature"] or ""
    placeholder_signature = records["conditional_generation.get_placeholder_mask"]["signature"] or ""
    image_signature = records["conditional_generation.get_image_features"]["signature"] or ""

    forward_readable = records["conditional_generation.forward"]["source_available"]
    vision_readable = records["vision.forward"]["source_available"]
    prepare_readable = records["conditional_generation.prepare_inputs_for_generation"]["source_available"]
    vision_pixels = any(name in vision_signature for name in ("pixel_values", "hidden_states"))
    vision_grid = any(name in vision_signature for name in ("grid_thw", "image_grid_thw"))
    image_helper = methods["conditional_generation.get_image_features"] is not None
    routes_vision = (image_helper and ("pixel_values" in image_signature or "image_grid_thw" in image_signature)
                     or "get_image_features" in model_forward
                     or any(token in model_forward for token in ("self.visual(", "self.vision(")))
    placeholder_helper = methods["conditional_generation.get_placeholder_mask"] is not None
    placeholder_contract = (placeholder_helper
                            and "input_ids" in placeholder_signature
                            and any(term in placeholder_signature for term in ("image_features", "inputs_embeds")))
    placeholder_ops = (any(term in placeholder for term in ("eq(", "==", "image_token_id"))
                       and any(term in placeholder + model_forward for term in
                               ("masked_scatter", "scatter", "where")))
    placeholder_inline = ("image_token_id" in model_forward
                          and any(term in model_forward for term in ("masked_scatter", "scatter", "where")))
    placeholder_proven = (placeholder_contract and placeholder_ops) or placeholder_inline
    grid_proven = (vision_grid and ("image_grid_thw" in model_signature
                                    or "image_grid_thw" in image_signature
                                    or "image_grid_thw" in model_forward))
    prepare_cache_signature = "past_key_values" in prepare_signature
    prepare_position_signature = "cache_position" in prepare_signature
    prepare_cache_logic = ("past_key_values" in prepare
                           and any(term in prepare for term in ("cache_position", "input_ids", "pixel_values")))
    generation_proven = prepare_cache_signature and (prepare_position_signature or prepare_cache_logic)
    rope_callable_present = (methods["base_model.get_rope_index"] is not None
                             or methods["conditional_generation.get_rope_index"] is not None)
    rope_proven = (rope_callable_present
                   and any(term in rope_sources for term in ("position_ids", "rope_delta", "mrope")))
    position_ids_proven = "position_ids" in model_signature or "position_ids" in base_forward
    visual_mask_proven = any(term in placeholder + model_forward + base_forward for term in
                             ("visual_pos_masks", "visual_position", "image_mask"))
    deepstack_present = any(term in combined for term in
                            ("deepstack", "deep_stack", "deepstack_visual_embeds"))
    deepstack_routed = (not deepstack_present or
                        (any(term in image_features + vision_source for term in ("deepstack", "deep_stack"))
                         and any(term in model_forward + base_forward for term in
                                 ("deepstack", "deep_stack"))))

    checks = [
        _check("conditional_generation_forward_source", True, forward_readable,
               ["inspect.getsource(Qwen conditional-generation forward)"], "source_unreadable",
               records["conditional_generation.forward"]["source_error"] or "source unavailable"),
        _check("vision_forward_source", True, vision_readable,
               ["inspect.getsource(actual vision forward)"], "source_unreadable",
               records["vision.forward"]["source_error"] or "source unavailable"),
        _check("generation_prepare_source", True, prepare_readable,
               ["inspect.getsource(prepare_inputs_for_generation)"], "source_unreadable",
               records["conditional_generation.prepare_inputs_for_generation"]["source_error"]
               or "source unavailable"),
        _check("vision_pixel_and_grid_signature", True, vision_pixels and vision_grid,
               [f"vision signature={vision_signature}", f"pixel argument recognized={vision_pixels}",
                f"grid argument recognized={vision_grid}"], "naming_variant_unresolved",
               "vision parameters did not match supported semantic alternatives"),
        _check("model_routes_pixels_to_vision", True, routes_vision,
               [f"get_image_features callable={image_helper}",
                "checked get_image_features call and self.visual/self.vision alternatives"],
               "interface_missing" if forward_readable else "source_unreadable",
               "could not prove that the multimodal forward invokes the vision path"),
        _check("image_placeholder_replacement", True, placeholder_proven,
               [f"get_placeholder_mask callable={placeholder_helper}",
                f"helper signature={placeholder_signature}",
                f"contract signature recognized={placeholder_contract}",
                f"mask/scatter operations recognized={placeholder_ops or placeholder_inline}"],
               "naming_variant_unresolved" if placeholder_helper else "interface_missing",
               "placeholder helper/inline path exists under an unrecognized contract"
               if placeholder_helper else "no placeholder replacement helper or inline path found"),
        _check("image_grid_thw_route", True, grid_proven,
               [f"model signature={model_signature}", f"image helper signature={image_signature}",
                f"vision signature={vision_signature}"], "naming_variant_unresolved",
               "could not trace image_grid_thw (or grid_thw at vision boundary) through signatures/source"),
        _check("generation_prefill_decode_contract", True, generation_proven,
               [f"prepare signature={prepare_signature}",
                f"past_key_values signature={prepare_cache_signature}",
                f"cache_position signature={prepare_position_signature}",
                f"cache branching logic recognized={prepare_cache_logic}"],
               "naming_variant_unresolved" if methods[
                   "conditional_generation.prepare_inputs_for_generation"] is not None else "interface_missing",
               "generation prepare method exists but prefill/decode cache contract was not recognized"),
        _check("position_ids_and_mrope_contract", True, position_ids_proven and rope_proven,
               [f"position_ids recognized={position_ids_proven}",
                f"rope callable present={rope_callable_present}", f"MRoPE/rope evidence={rope_proven}"],
               "naming_variant_unresolved" if rope_callable_present else "interface_missing",
               "position/MRoPE implementation exists under an unrecognized path"
               if rope_callable_present else "no get_rope_index variant found"),
        _check("visual_position_mask_contract", False, visual_mask_proven,
               ["checked get_placeholder_mask, conditional forward, and base-model forward"],
               "naming_variant_unresolved", "visual position mask naming/route was not recognized"),
        _check("deepstack_route_if_present", True, deepstack_routed,
               [f"DeepStack symbols present={deepstack_present}",
                f"DeepStack routed through vision and language model={deepstack_routed}"],
               "interface_missing", "DeepStack is declared but its complete route was not proven"),
        _check("visual_token_order", True,
               order.get("status") == "confirmed_from_installed_source",
               [f"token order status={order.get('status')}",
                f"warning={order.get('warning')}"], "naming_variant_unresolved",
               order.get("warning") or "visual token order is unresolved"),
    ]
    failed_required = [item for item in checks if item["required"] and not item["passed"]]
    evidence = {
        "image_placeholder_replacement": placeholder_proven,
        "image_grid_thw": grid_proven, "position_ids": position_ids_proven,
        "mrope_or_rope_deltas": rope_proven,
        "generation_prefill_decode_branch": generation_proven,
        "deepstack_in_source": deepstack_present,
        "deepstack_route_proven": deepstack_routed,
        "visual_position_mask": visual_mask_proven,
        "token_order_status": order.get("status"),
    }
    return {
        "status": "ready" if not failed_required else "blocked",
        "transformers_version": transformers_version,
        "actual_classes": {"model": model.__class__.__name__, "vision": vision.__class__.__name__,
                           "processor": processor.__class__.__name__,
                           "base_model": base_model.__class__.__name__ if base_model is not None else None},
        "inspected_methods": records, "checks": checks,
        "failed_required_checks": [item["name"] for item in failed_required],
        "failure_summary": [f"{item['name']}: {item['failure_kind']}: {item['failure_reason']}"
                            for item in failed_required],
        "evidence": evidence,
        "vision_return_structure": {
            "runtime_observed": False,
            "forward_signature": vision_signature,
            "return_statements_in_source": vision_source.count("return "),
            "deepstack_symbols_in_source": deepstack_present,
            "note": "Source evidence only; the full probe replaces this with the observed tensor tree.",
        },
        "runtime_vision_return": {"observed": False,
                                  "reason": "source-only phase; populated after native vision execution"},
        "transformers_site_packages_modified": False, "global_monkey_patch": False,
    }


def contract_failure_message(report: dict[str, Any]) -> str:
    failures = report.get("failure_summary", [])
    return "source contract failed: " + ("; ".join(failures) if failures else "unknown check")


class QwenVisualInjectionAdapter:
    """Serial-only adapter for exact native-output capture and local replacement."""

    def __init__(self, model: Any, vision: Any, merge_size: int, order: dict[str, Any]) -> None:
        self.model, self.vision, self.merge_size, self.order = model, vision, int(merge_size), order
        self._lock = threading.Lock()
        self.counters = {"vision_forward_call_count": 0, "actual_vision_computation_count": 0,
                         "cached_replacement_count": 0, "cached_replacement_cpu_wall_ms": 0.0}

    @contextmanager
    def capture(self) -> Iterator[dict[str, Any]]:
        captured: dict[str, Any] = {}

        def hook(_module: Any, _args: Any, output: Any) -> None:
            self.counters["vision_forward_call_count"] += 1
            self.counters["actual_vision_computation_count"] += 1
            captured["output"] = output

        handle = self.vision.register_forward_hook(hook)
        try:
            yield captured
        finally:
            handle.remove()

    @contextmanager
    def replace(self, state: VisualState) -> Iterator[None]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("visual injection probe is serial-only; concurrent replacement is forbidden")
        original = self.vision.forward

        def replacement(*_args: Any, **_kwargs: Any) -> Any:
            started = time.perf_counter()
            self.counters["vision_forward_call_count"] += 1
            self.counters["cached_replacement_count"] += 1
            try:
                return state.vision_output
            finally:
                self.counters["cached_replacement_cpu_wall_ms"] += (time.perf_counter() - started) * 1000.0

        self.vision.forward = replacement
        try:
            yield
        finally:
            self.vision.forward = original
            self._lock.release()

    def make_state(self, output: Any, image_grid_thw: Any, state_timestamp: Any,
                   anchor_timestamp: Any, source_audit: dict[str, Any]) -> VisualState:
        if self.order.get("status") != "confirmed_from_installed_source":
            raise InjectionBlocked("visual token ordering is not proven by installed source")
        grid = tuple(int(x) for x in image_grid_thw.reshape(-1).tolist())
        if len(grid) != 3 or grid[0] != 1 or grid[1] % self.merge_size or grid[2] % self.merge_size:
            raise InjectionBlocked(f"unsupported image_grid_thw for one-image state: {grid}")
        spatial_hw = (grid[1] // self.merge_size, grid[2] // self.merge_size)
        token_count = spatial_hw[0] * spatial_hw[1]
        pre_hw = (grid[1], grid[2])
        pre_token_count = pre_hw[0] * pre_hw[1]
        tensors = tree_tensors(output)
        final_candidates = [(path, tensor) for path, tensor in tensors
                            if _token_axis(tensor, token_count) is not None]
        if not final_candidates:
            raise InjectionBlocked("vision output has no tensor matching the verified post-merger grid")
        final_path = final_candidates[0][0]
        branch_grid: dict[str, tuple[int, int]] = {}
        spatial = []
        for path, tensor in tensors:
            if _token_axis(tensor, token_count) is not None:
                branch_grid[path] = spatial_hw
                spatial.append((path, tensor))
            elif _token_axis(tensor, pre_token_count) is not None:
                branch_grid[path] = pre_hw
                spatial.append((path, tensor))
        unresolved = [path for path, tensor in tensors
                      if getattr(tensor, "ndim", len(tensor.shape)) >= 2 and path not in branch_grid]
        if unresolved:
            raise InjectionBlocked("vision output contains tensor branches without a proven spatial layout: "
                                   + ", ".join(unresolved))
        deepstack = tuple(path for path, _tensor in spatial if path != final_path)
        source_deepstack = bool(source_audit.get("evidence", {}).get("deepstack_in_source"))
        if source_deepstack and not deepstack:
            raise InjectionBlocked("installed source contains DeepStack but full vision output did not expose it")
        first = spatial[0][1]
        torch = __import__("torch")
        validity = {path: torch.ones(branch_grid[path], dtype=torch.float32, device=tensor.device)
                    for path, tensor in spatial}
        return VisualState(output, grid, spatial_hw, self.merge_size, final_path,
                           tuple(path for path, _tensor in spatial), deepstack, branch_grid,
                           str(first.dtype), str(first.device), state_timestamp, anchor_timestamp, True,
                           validity, {"source_deepstack_detected": source_deepstack,
                                      "all_vision_return_values_preserved": True})

    def validate_placeholder_contract(self, inputs: dict[str, Any], state: VisualState) -> dict[str, Any]:
        input_ids = inputs.get("input_ids")
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if input_ids is None or image_token_id is None:
            raise InjectionBlocked("input_ids or model.config.image_token_id is unavailable")
        count = int((input_ids == image_token_id).sum().item())
        if count != state.token_count:
            raise InjectionBlocked(f"image placeholder count {count} != visual token count {state.token_count}")
        grid = tuple(int(x) for x in inputs["image_grid_thw"].reshape(-1).tolist())
        if grid != state.image_grid_thw:
            raise InjectionBlocked(f"processor grid {grid} != cached state grid {state.image_grid_thw}")
        return {"image_placeholder_count": count, "visual_token_count": state.token_count,
                "image_grid_thw": list(grid), "matches": True}


def propagate_visual_state(state: VisualState, flow_pixels: Any, state_timestamp: Any,
                           support_mask: Any | None = None) -> VisualState:
    """Propagate every verified spatial branch and causally fill splat holes from prior state."""
    if not state.layout_verified:
        raise InjectionBlocked("cannot propagate an unverified visual layout")
    torch = __import__("torch")
    if support_mask is not None:
        support = torch.as_tensor(support_mask, dtype=flow_pixels.dtype, device=flow_pixels.device)
        flow_pixels = flow_pixels * support[..., None]
    tensors = dict(tree_tensors(state.vision_output))
    replacements, validity, branch_diagnostics = {}, {}, {}
    for path in state.spatial_tensor_paths:
        tensor = tensors[path]
        branch_hw = state.branch_grid_hw[path]
        branch_tokens = branch_hw[0] * branch_hw[1]
        axis = _token_axis(tensor, branch_tokens)
        if axis is None:
            raise InjectionBlocked(f"cached visual branch lost its spatial contract: {path}")
        squeezed = tensor.squeeze(0) if tensor.ndim == 3 else tensor
        grid = squeezed.reshape(*branch_hw, squeezed.shape[-1])
        branch_displacement = predicted_flow_to_feature_displacement(
            flow_pixels, branch_hw, tuple(int(x) for x in flow_pixels.shape[:2]))
        confidence = state.validity[path]
        warped, valid, propagated_confidence = forward_splat_torch(
            grid, branch_displacement, confidence)
        # Fixed causal fallback: retain the previous state's same-position token at holes.
        filled = causal_hole_fill(warped, grid, valid)
        replacements[path] = filled.reshape_as(squeezed).unsqueeze(0) if tensor.ndim == 3 else filled.reshape_as(tensor)
        validity[path] = torch.where(valid, propagated_confidence, confidence)
        branch_diagnostics[path] = {"invalid_ratio_before_fallback": float((~valid).float().mean().item()),
                                    "hole_filling_policy": "previous_state_same_position",
                                    "target_rgb_or_gt_fill": False,
                                    "finite": bool(torch.isfinite(filled).all().item())}
    return VisualState(_replace_tree(state.vision_output, replacements), state.image_grid_thw,
                       state.spatial_grid_hw, state.merge_size, state.final_path,
                       state.spatial_tensor_paths, state.deepstack_paths, state.branch_grid_hw,
                       state.dtype, state.device,
                       state_timestamp, state.anchor_timestamp, True, validity,
                       {**state.metadata, "propagation": branch_diagnostics,
                        "all_spatial_visual_branches_propagated": True})


def compare_logits(native_a: Any, native_b: Any, injected: Any) -> dict[str, Any]:
    """Numerical equivalence metrics for full prefill logits."""
    def difference(left: Any, right: Any) -> dict[str, float]:
        delta = (left.detach().float() - right.detach().float()).abs()
        return {"max_absolute_difference": float(delta.max().item()),
                "mean_absolute_difference": float(delta.mean().item())}
    return {"native_repeat": difference(native_a, native_b),
            "injected_vs_native": difference(native_a, injected),
            "last_prompt_token": difference(native_a[:, -1], injected[:, -1]),
            "native_next_token_argmax": int(native_a[:, -1].argmax(dim=-1).item()),
            "injected_next_token_argmax": int(injected[:, -1].argmax(dim=-1).item())}


def injection_passes(metrics: dict[str, Any], native_ids: Any, injected_ids: Any,
                     max_atol: float, mean_atol: float, bypassed: bool) -> bool:
    comparison = metrics["injected_vs_native"]
    return bool(comparison["max_absolute_difference"] <= max_atol
                and comparison["mean_absolute_difference"] <= mean_atol
                and metrics["native_next_token_argmax"] == metrics["injected_next_token_argmax"]
                and bool((native_ids == injected_ids).all().item()) and bypassed)


def tensor_tree_summary(output: Any) -> dict[str, Any]:
    return {path: tensor_summary(tensor) for path, tensor in tree_tensors(output)}
