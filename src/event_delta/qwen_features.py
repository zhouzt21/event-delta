"""Qwen vision-only structure probing and spatial feature extraction.

All model-specific inspection lives here. No language-model forward or generation is used.
"""

from __future__ import annotations

import hashlib
import inspect
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class ProbeError(RuntimeError):
    """Base error for a probe that cannot safely continue."""


class TokenOrderError(ProbeError):
    """The installed source does not prove a reversible 2D token order."""


def require_runtime() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ProbeError("R2 requires installed torch and transformers; no dependency is downloaded by this tool") from exc
    return torch, transformers


def first_tensor(value: Any) -> Any | None:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return value
    if isinstance(value, (tuple, list)):
        return next((found for item in value if (found := first_tensor(item)) is not None), None)
    if isinstance(value, dict):
        return next((found for item in value.values() if (found := first_tensor(item)) is not None), None)
    return None


def tensor_summary(tensor: Any | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    work = tensor.detach().float()
    finite = work[work.isfinite()]
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "finite_count": int(finite.numel()), "non_finite_count": int(work.numel() - finite.numel()),
            "min": float(finite.min().item()) if finite.numel() else None,
            "max": float(finite.max().item()) if finite.numel() else None,
            "mean": float(finite.mean().item()) if finite.numel() else None,
            "std": float(finite.std().item()) if finite.numel() > 1 else 0.0}


def module_path(model: Any, target: Any) -> str:
    return next((name or "<root>" for name, module in model.named_modules() if module is target), "<unknown>")


def discover_vision_structure(model: Any) -> dict[str, Any]:
    """Find the visual tower, merger and last block by behavior/name, not a fixed attribute path."""
    modules = list(model.named_modules())
    candidates = []
    for name, module in modules:
        class_name = module.__class__.__name__.lower()
        try:
            parameters = set(inspect.signature(module.forward).parameters)
        except (TypeError, ValueError):
            parameters = set()
        descendants = [(child_name, child) for child_name, child in modules
                       if name and child_name.startswith(name + ".")]
        has_merger = any("merger" in child_name.lower() or "patchmerger" in child.__class__.__name__.lower()
                         for child_name, child in descendants)
        block_count = sum("block" in child_name.lower() for child_name, _child in descendants)
        score = (4 * ("vision" in class_name or "visual" in class_name)
                 + 4 * bool({"pixel_values", "grid_thw"} & parameters)
                 + 5 * has_merger + min(block_count, 3))
        if score:
            candidates.append((score, -name.count("."), name, module, descendants))
    if not candidates:
        raise ProbeError("could not discover a vision module from installed model structure")
    _score, _depth, vision_name, vision, descendants = max(candidates, key=lambda item: (item[0], item[1]))
    mergers = [(name, module) for name, module in descendants
               if "merger" in name.lower() or "patchmerger" in module.__class__.__name__.lower()]
    if not mergers:
        raise ProbeError(f"vision candidate {vision_name!r} has no discoverable merger")
    merger_name, merger = min(mergers, key=lambda item: item[0].count("."))
    blocks = [(name, module) for name, module in descendants
              if "block" in module.__class__.__name__.lower()
              and "list" not in module.__class__.__name__.lower()]
    if not blocks:
        raise ProbeError(f"vision candidate {vision_name!r} has no discoverable vision blocks")
    block_name, block = blocks[-1]
    return {"vision": vision, "vision_path": vision_name, "merger": merger, "merger_path": merger_name,
            "last_block": block, "last_block_path": block_name}


def _source(value: Any) -> str:
    try:
        return inspect.getsource(value)
    except (OSError, TypeError):
        return ""


def _class_sources(instance: Any) -> str:
    return "\n".join(_source(base) for base in instance.__class__.__mro__
                      if base is not object and ("qwen" in base.__module__.lower()
                                                 or "transformers" in base.__module__.lower()))


def _source_file(value: Any) -> str | None:
    try:
        return inspect.getsourcefile(value)
    except TypeError:
        return None


def analyze_token_order(vision: Any, merger: Any, processor: Any) -> dict[str, Any]:
    """Conservatively recognize Qwen's merge-blocked patch order from installed source."""
    vision_source = _class_sources(vision)
    merger_source = _class_sources(merger)
    image_processor = getattr(processor, "image_processor", processor)
    processor_source = _class_sources(image_processor)
    combined = "\n".join((vision_source, merger_source, processor_source)).lower()
    compact_processor = re.sub(r"\s+", "", processor_source.lower())
    compact_merger = re.sub(r"\s+", "", merger_source.lower())
    known_block_permutation = any(pattern in compact_processor for pattern in (
        "permute(0,2,5,3,6,1,4,7)",
        "permute(0,1,4,7,5,8,3,2,6,9)",
        "transpose(patches,(1,4,2,5,0,3,6))",
    ))
    qwen_family = "qwen" in vision.__class__.__module__.lower() and "qwen" in vision.__class__.__name__.lower()
    evidence = {
        "qwen_family": qwen_family,
        "spatial_merge_size_in_source": "spatial_merge_size" in combined,
        "reshape_or_view_in_source": "reshape" in combined or ".view(" in combined,
        "known_merge_block_permutation": known_block_permutation,
        "merger_groups_consecutive_tokens": (".view(-1,self.hidden_size)" in compact_merger
                                              or ".reshape(-1,self.hidden_size)" in compact_merger),
        "processor_mentions_merge_size": ("merge_size" in processor_source.lower()
                                           or "spatial_merge_size" in processor_source.lower()),
    }
    reliable = all(evidence.values())
    source_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return {"status": "confirmed_from_installed_source" if reliable else "unresolved",
            "layout": "merge_blocked_raster" if reliable else None,
            "post_merger_layout": "row_major_merged_grid" if reliable else None,
            "pre_merger_inverse": "reshape(t,h/m,w/m,m,m,c)->permute(0,1,3,2,4,5)->reshape(t,h,w,c)"
                                  if reliable else None,
            "source_sha256": source_hash,
            "source_files": {"vision": _source_file(vision.__class__),
                             "merger": _source_file(merger.__class__),
                             "image_processor": _source_file(image_processor.__class__)},
            "evidence": evidence,
            "warning": None if reliable else
            "installed model/processor source did not prove Qwen merge-blocked ordering; do not reshape tokens"}


def config_value(objects: list[Any], names: tuple[str, ...]) -> Any | None:
    queue = list(objects)
    visited = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        for name in names:
            value = getattr(current, name, None)
            if value is not None and isinstance(value, (int, float, str, tuple, list)):
                return value
        for child_name in ("vision_config", "visual_config", "config"):
            child = getattr(current, child_name, None)
            if child is not None and child is not current:
                queue.append(child)
    return None


def normalize_tokens(tensor: Any) -> Any:
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ProbeError(f"expected a single-image NxC visual tensor, got {tuple(tensor.shape)}")
    return tensor


def tokens_to_grid(tokens: Any, grid_thw: Any, merge_size: int, stage: str, order: dict[str, Any]) -> Any:
    if order.get("status") != "confirmed_from_installed_source":
        raise TokenOrderError(order.get("warning") or "visual token ordering is unresolved")
    values = [int(value) for value in grid_thw.reshape(-1).tolist()]
    if len(values) != 3:
        raise ProbeError(f"expected one image_grid_thw triplet, got {values}")
    time, height, width = values
    if time != 1:
        raise ProbeError(f"R2 smoke supports single images only; image_grid_thw has temporal size {time}")
    tokens = normalize_tokens(tokens)
    channels = tokens.shape[-1]
    if stage == "pre_merger":
        expected = time * height * width
        if tokens.shape[0] != expected or height % merge_size or width % merge_size:
            raise ProbeError(f"pre-merger token/grid mismatch: tokens={tokens.shape[0]}, grid={values}, merge={merge_size}")
        blocked = tokens.reshape(time, height // merge_size, width // merge_size,
                                 merge_size, merge_size, channels)
        if hasattr(blocked, "permute"):
            blocked = blocked.permute(0, 1, 3, 2, 4, 5)
        else:
            blocked = np.transpose(blocked, (0, 1, 3, 2, 4, 5))
        return blocked.reshape(time, height, width, channels)[0]
    if stage == "post_merger":
        expected = time * height * width // (merge_size * merge_size)
        if tokens.shape[0] != expected or height % merge_size or width % merge_size:
            raise ProbeError(f"post-merger token/grid mismatch: tokens={tokens.shape[0]}, grid={values}, merge={merge_size}")
        return tokens.reshape(time, height // merge_size, width // merge_size, channels)[0]
    raise ValueError(f"unsupported feature stage: {stage}")


class QwenVisionExtractor:
    """Load a Qwen model and expose vision features without invoking its language model."""

    def __init__(self, model_name: str, local_files_only: bool = False, device: str = "cuda") -> None:
        torch, transformers = require_runtime()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ProbeError("CUDA is required for the BF16 Qwen R2 probe, but torch.cuda.is_available() is false")
        auto_model = getattr(transformers, "AutoModelForImageTextToText", None)
        if auto_model is None:
            auto_model = getattr(transformers, "AutoModelForVision2Seq", None)
        if auto_model is None:
            raise ProbeError("installed transformers has no image-text auto model class")
        self.torch, self.transformers = torch, transformers
        self.processor = transformers.AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = auto_model.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa", local_files_only=local_files_only)
        self.model.to(device).eval()
        self.device = torch.device(device)
        self.structure = discover_vision_structure(self.model)
        self.order = analyze_token_order(self.structure["vision"], self.structure["merger"], self.processor)
        self.patch_size = config_value([self.structure["vision"], self.model.config], ("patch_size",))
        self.merge_size = int(config_value([self.structure["vision"], self.model.config],
                                           ("spatial_merge_size",)) or 0)
        self.hidden_size = int(config_value([self.structure["vision"], self.model.config],
                                            ("hidden_size", "embed_dim")) or 0)
        patch_values = ([self.patch_size] if isinstance(self.patch_size, (int, float)) else self.patch_size or [])
        if not patch_values or any(float(value) <= 0 for value in patch_values) or self.merge_size <= 0:
            raise ProbeError("could not determine patch_size/spatial_merge_size from actual model configuration")

    def preprocess(self, image: Image.Image | Path) -> dict[str, Any]:
        if isinstance(image, Path):
            with Image.open(image) as opened:
                image = opened.convert("RGB").copy()
        try:
            inputs = self.processor(images=[image], return_tensors="pt")
            self.preprocess_mode = "official processor image-only call"
        except (TypeError, ValueError):
            messages = [{"role": "user", "content": [{"type": "image"}]}]
            if not hasattr(self.processor, "apply_chat_template"):
                raise ProbeError("processor supports neither image-only calls nor apply_chat_template")
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            self.preprocess_mode = "official processor with image-only chat placeholder"
        if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            raise ProbeError(f"processor output lacks pixel_values/image_grid_thw; keys={list(inputs.keys())}")
        return inputs

    def _vision_forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def block_hook(_module: Any, _args: Any, output: Any) -> None:
            captured["last_block"] = first_tensor(output)

        def merger_pre_hook(_module: Any, args: Any) -> None:
            captured["pre_merger"] = first_tensor(args)

        def merger_hook(_module: Any, _args: Any, output: Any) -> None:
            captured["post_merger"] = first_tensor(output)

        handles = [self.structure["last_block"].register_forward_hook(block_hook),
                   self.structure["merger"].register_forward_pre_hook(merger_pre_hook),
                   self.structure["merger"].register_forward_hook(merger_hook)]
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.torch.bfloat16)
        grid_thw = inputs["image_grid_thw"].to(self.device)
        try:
            with self.torch.inference_mode():
                parameters = inspect.signature(self.structure["vision"].forward).parameters
                kwargs = {}
                if "pixel_values" in parameters:
                    kwargs["pixel_values"] = pixel_values
                elif "hidden_states" in parameters:
                    kwargs["hidden_states"] = pixel_values
                else:
                    raise ProbeError(f"unsupported vision forward parameters: {list(parameters)}")
                if "grid_thw" in parameters:
                    kwargs["grid_thw"] = grid_thw
                elif "image_grid_thw" in parameters:
                    kwargs["image_grid_thw"] = grid_thw
                else:
                    raise ProbeError(f"vision forward has no grid argument: {list(parameters)}")
                output = self.structure["vision"](**kwargs)
                pooler_output = getattr(output, "pooler_output", None)
                captured["vision_output"] = pooler_output if pooler_output is not None else first_tensor(output)
        finally:
            for handle in handles:
                handle.remove()
        if captured.get("pre_merger") is None or captured.get("post_merger") is None:
            raise ProbeError("vision hooks did not capture both pre- and post-merger tensors")
        captured["pixel_values"] = pixel_values
        captured["image_grid_thw"] = grid_thw
        return captured

    def encode_inputs(self, inputs: dict[str, Any], stage: str = "post_merger") -> tuple[Any, dict[str, Any]]:
        captured = self._vision_forward(inputs)
        grid = tokens_to_grid(captured[stage], captured["image_grid_thw"], self.merge_size, stage, self.order)
        return grid, self.probe_report(captured)

    def feature_grid_from_inputs(self, inputs: dict[str, Any], stage: str = "post_merger") -> Any:
        """Vision-only hot path used for CUDA-event latency profiling."""
        captured = self._vision_forward(inputs)
        return tokens_to_grid(captured[stage], captured["image_grid_thw"], self.merge_size, stage, self.order)

    def encode_image(self, image: Image.Image | Path, stage: str = "post_merger") -> tuple[Any, dict[str, Any]]:
        return self.encode_inputs(self.preprocess(image), stage)

    def feature_grid_from_image(self, image: Image.Image | Path, stage: str = "post_merger") -> Any:
        return self.feature_grid_from_inputs(self.preprocess(image), stage)

    def probe_image(self, image: Image.Image | Path) -> dict[str, Any]:
        return self.probe_report(self._vision_forward(self.preprocess(image)))

    def probe_report(self, captured: dict[str, Any]) -> dict[str, Any]:
        grid_values = [int(value) for value in captured["image_grid_thw"].reshape(-1).tolist()]
        pre = normalize_tokens(captured["pre_merger"])
        post = normalize_tokens(captured["post_merger"])
        expected_pre = int(np.prod(grid_values))
        expected_post = expected_pre // (self.merge_size * self.merge_size)
        return {
            "status": "ready" if self.order["status"] == "confirmed_from_installed_source" else "blocked",
            "versions": {"python": platform.python_version(), "torch": self.torch.__version__,
                         "transformers": self.transformers.__version__, "numpy": np.__version__},
            "model_class": self.model.__class__.__name__, "processor_class": self.processor.__class__.__name__,
            "vision_module": {"path": self.structure["vision_path"],
                              "class": self.structure["vision"].__class__.__name__},
            "last_vision_block": {"path": self.structure["last_block_path"],
                                  "class": self.structure["last_block"].__class__.__name__,
                                  "feature": tensor_summary(captured.get("last_block"))},
            "merger": {"path": self.structure["merger_path"],
                       "class": self.structure["merger"].__class__.__name__},
            "pixel_values": tensor_summary(captured["pixel_values"]),
            "image_grid_thw": grid_values, "patch_size": self.patch_size,
            "spatial_merge_size": self.merge_size, "vision_hidden_size": self.hidden_size,
            "pre_merger_feature": tensor_summary(pre), "post_merger_feature": tensor_summary(post),
            "final_visual_tokens": tensor_summary(captured.get("vision_output")),
            "grid_token_consistency": {"expected_pre_merger_tokens": expected_pre,
                                       "actual_pre_merger_tokens": int(pre.shape[0]),
                                       "pre_merger_matches": int(pre.shape[0]) == expected_pre,
                                       "expected_post_merger_tokens": expected_post,
                                       "actual_post_merger_tokens": int(post.shape[0]),
                                       "post_merger_matches": int(post.shape[0]) == expected_post},
            "spatial_recoverability": {
                "pre_merger": self.order["status"] == "confirmed_from_installed_source",
                "post_merger": self.order["status"] == "confirmed_from_installed_source",
                "pre_merger_grid_hw": grid_values[1:],
                "post_merger_grid_hw": [grid_values[1] // self.merge_size, grid_values[2] // self.merge_size]},
            "token_order": self.order,
            "execution": {"dtype": "torch.bfloat16", "attention": "sdpa", "eval": True,
                          "inference_mode": True, "language_model_forward_used": False,
                          "generation_used": False, "preprocess_mode": getattr(self, "preprocess_mode", None)},
        }
