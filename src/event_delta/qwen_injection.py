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
    try:
        return inspect.getsource(value)
    except (OSError, TypeError):
        return ""


def _source_record(value: Any) -> dict[str, Any]:
    source = _source(value)
    try:
        path = inspect.getsourcefile(value)
    except TypeError:
        path = None
    return {"file": path, "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "line_count": len(source.splitlines()), "source_available": bool(source)}


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


def audit_qwen_interface(model: Any, vision: Any, processor: Any, order: dict[str, Any]) -> dict[str, Any]:
    """Audit the installed sources that govern multimodal forward and generation."""
    forward = _source(model.forward)
    vision_forward = _source(vision.forward)
    prepare = _source(getattr(model, "prepare_inputs_for_generation", None))
    rope_callable = next((getattr(model, name) for name in ("get_rope_index", "_get_rope_index")
                          if hasattr(model, name)), None)
    rope = _source(rope_callable)
    combined = "\n".join((forward, vision_forward, prepare, rope)).lower()
    report = {
        "model_forward": _source_record(model.forward),
        "vision_forward": _source_record(vision.forward),
        "prepare_inputs_for_generation": _source_record(
            getattr(model, "prepare_inputs_for_generation", None)),
        "rope_function": _source_record(rope_callable),
        "model_forward_signature": str(inspect.signature(model.forward)),
        "vision_forward_signature": str(inspect.signature(vision.forward)),
        "generation_prepare_signature": str(inspect.signature(model.prepare_inputs_for_generation)),
        "evidence": {
            "image_placeholder_replacement": any(word in forward.lower() for word in
                                                   ("masked_scatter", "image_mask", "image_token")),
            "image_grid_thw": "image_grid_thw" in combined,
            "position_ids": "position_ids" in combined,
            "mrope_or_rope_deltas": "rope_delta" in combined or "mrope" in combined,
            "generation_prefill_decode_branch": ("past_key_values" in prepare.lower()
                                                   and "cache_position" in prepare.lower()),
            "deepstack_in_source": "deepstack" in combined or "deep_stack" in combined,
            "visual_position_mask": "visual_pos" in combined or "image_mask" in combined,
            "token_order_status": order.get("status"),
        },
        "processor_class": processor.__class__.__name__,
        "transformers_site_packages_modified": False,
        "global_monkey_patch": False,
    }
    required = (report["model_forward"]["source_available"], report["vision_forward"]["source_available"],
                report["prepare_inputs_for_generation"]["source_available"],
                report["evidence"]["image_placeholder_replacement"], report["evidence"]["image_grid_thw"],
                report["evidence"]["generation_prefill_decode_branch"],
                order.get("status") == "confirmed_from_installed_source")
    report["status"] = "ready" if all(required) else "blocked"
    if report["status"] == "blocked":
        report["warning"] = "installed source does not prove the full visual injection contract"
    return report


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
