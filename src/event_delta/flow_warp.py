"""SEVD GT-flow conversion and forward feature splatting.

The confirmed R1.2 convention is source-indexed flow_i for RGB_i -> RGB_{i+1}:
dx_pixel = (u + 1/1024) * original_width / 2, and likewise for y.
"""

from __future__ import annotations

from typing import Any

import numpy as np


SEVD_BACKGROUND = (-1.0 / 1024.0, -1.0 / 1024.0)


def _is_torch(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch")


def sevd_flow_to_pixel_displacement(flow: Any, original_hw: tuple[int, int],
                                    background: tuple[float, float] = SEVD_BACKGROUND) -> Any:
    """Apply confirmed background subtraction and half-image-size scaling."""
    height, width = original_hw
    if tuple(flow.shape[-1:]) != (2,):
        raise ValueError(f"expected flow [..., 2], got {tuple(flow.shape)}")
    if _is_torch(flow):
        import torch
        bg = torch.as_tensor(background, dtype=flow.dtype, device=flow.device)
        scale = torch.as_tensor((width / 2.0, height / 2.0), dtype=flow.dtype, device=flow.device)
    else:
        bg = np.asarray(background, dtype=np.asarray(flow).dtype)
        scale = np.asarray((width / 2.0, height / 2.0), dtype=np.asarray(flow).dtype)
    return (flow - bg) * scale


def pixel_to_feature_displacement(pixel_displacement: Any, original_hw: tuple[int, int],
                                  feature_hw: tuple[int, int]) -> Any:
    """Convert original-image pixel displacement to feature-grid cell displacement."""
    original_height, original_width = original_hw
    feature_height, feature_width = feature_hw
    if _is_torch(pixel_displacement):
        import torch
        scale = torch.as_tensor((feature_width / original_width, feature_height / original_height),
                                dtype=pixel_displacement.dtype, device=pixel_displacement.device)
    else:
        scale = np.asarray((feature_width / original_width, feature_height / original_height),
                           dtype=np.asarray(pixel_displacement).dtype)
    return pixel_displacement * scale


def resize_flow_to_grid(flow: Any, feature_hw: tuple[int, int]) -> Any:
    """Bilinearly sample raw flow components at feature-grid resolution."""
    feature_height, feature_width = feature_hw
    if _is_torch(flow):
        import torch.nn.functional as functional
        values = flow.permute(2, 0, 1).unsqueeze(0)
        resized = functional.interpolate(values, size=feature_hw, mode="bilinear", align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0)
    array = np.asarray(flow)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"expected HxWx2 flow, got {array.shape}")
    from PIL import Image
    channels = []
    for channel in range(2):
        image = Image.fromarray(array[..., channel].astype(np.float32), mode="F")
        channels.append(np.asarray(image.resize((feature_width, feature_height), Image.Resampling.BILINEAR),
                                   dtype=np.float32))
    return np.stack(channels, axis=-1)


def sevd_flow_to_feature_displacement(flow: Any, feature_hw: tuple[int, int],
                                      original_hw: tuple[int, int] | None = None,
                                      background: tuple[float, float] = SEVD_BACKGROUND) -> Any:
    """Resize flow values, then explicitly convert image pixels to feature-grid cells."""
    if original_hw is None:
        original_hw = tuple(int(value) for value in flow.shape[:2])
    resized = resize_flow_to_grid(flow, feature_hw)
    pixel = sevd_flow_to_pixel_displacement(resized, original_hw, background)
    return pixel_to_feature_displacement(pixel, original_hw, feature_hw)


def _validate_feature_inputs(features: Any, displacement: Any, confidence: Any | None) -> None:
    if len(features.shape) != 3:
        raise ValueError(f"expected HxWxC features, got {tuple(features.shape)}")
    if tuple(displacement.shape) != (*tuple(features.shape[:2]), 2):
        raise ValueError(f"displacement shape {tuple(displacement.shape)} does not match feature grid")
    if confidence is not None and tuple(confidence.shape) != tuple(features.shape[:2]):
        raise ValueError("confidence shape does not match feature grid")


def forward_splat_numpy(features: np.ndarray, displacement: np.ndarray,
                        confidence: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward bilinear splat HxWxC features and propagate confidence."""
    features = np.asarray(features)
    displacement = np.asarray(displacement)
    _validate_feature_inputs(features, displacement, confidence)
    height, width, channels = features.shape
    confidence = (np.ones((height, width), dtype=np.float32) if confidence is None
                  else np.asarray(confidence, dtype=np.float32))
    yy, xx = np.indices((height, width), dtype=np.float32)
    tx, ty = xx + displacement[..., 0], yy + displacement[..., 1]
    finite = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(confidence) & (confidence > 0)
    numerator = np.zeros((height * width, channels), dtype=np.float64)
    weights = np.zeros(height * width, dtype=np.float64)
    fractions_x, fractions_y = tx - np.floor(tx), ty - np.floor(ty)
    neighbors = (
        (np.floor(tx), np.floor(ty), (1 - fractions_x) * (1 - fractions_y)),
        (np.floor(tx) + 1, np.floor(ty), fractions_x * (1 - fractions_y)),
        (np.floor(tx), np.floor(ty) + 1, (1 - fractions_x) * fractions_y),
        (np.floor(tx) + 1, np.floor(ty) + 1, fractions_x * fractions_y),
    )
    for x_target, y_target, bilinear in neighbors:
        safe_x = np.where(finite, x_target, 0).astype(np.int64)
        safe_y = np.where(finite, y_target, 0).astype(np.int64)
        valid = finite & (safe_x >= 0) & (safe_x < width) & (safe_y >= 0) & (safe_y < height)
        contribution = bilinear * confidence
        valid &= contribution > 0
        indices = (safe_y[valid] * width + safe_x[valid]).reshape(-1)
        np.add.at(weights, indices, contribution[valid])
        np.add.at(numerator, indices, features[valid] * contribution[valid, None])
    valid_target = weights > 1e-8
    output = np.zeros_like(numerator, dtype=np.float64)
    output[valid_target] = numerator[valid_target] / weights[valid_target, None]
    propagated_confidence = np.clip(weights, 0.0, 1.0).reshape(height, width).astype(np.float32)
    return output.reshape(height, width, channels).astype(features.dtype), valid_target.reshape(height, width), propagated_confidence


def forward_splat_torch(features: Any, displacement: Any, confidence: Any | None = None) -> tuple[Any, Any, Any]:
    """CUDA-compatible forward bilinear splat using scatter_add_."""
    import torch
    _validate_feature_inputs(features, displacement, confidence)
    height, width, channels = features.shape
    work_dtype = torch.float32
    values = features.to(work_dtype)
    motion = displacement.to(work_dtype)
    confidence = (torch.ones((height, width), dtype=work_dtype, device=features.device)
                  if confidence is None else confidence.to(work_dtype))
    yy, xx = torch.meshgrid(torch.arange(height, device=features.device, dtype=work_dtype),
                            torch.arange(width, device=features.device, dtype=work_dtype), indexing="ij")
    tx, ty = xx + motion[..., 0], yy + motion[..., 1]
    finite = torch.isfinite(tx) & torch.isfinite(ty) & torch.isfinite(confidence) & (confidence > 0)
    floor_x, floor_y = torch.floor(torch.where(finite, tx, torch.zeros_like(tx))), torch.floor(
        torch.where(finite, ty, torch.zeros_like(ty)))
    fraction_x, fraction_y = tx - floor_x, ty - floor_y
    numerator = torch.zeros((height * width, channels), dtype=work_dtype, device=features.device)
    weights = torch.zeros(height * width, dtype=work_dtype, device=features.device)
    neighbors = (
        (floor_x, floor_y, (1 - fraction_x) * (1 - fraction_y)),
        (floor_x + 1, floor_y, fraction_x * (1 - fraction_y)),
        (floor_x, floor_y + 1, (1 - fraction_x) * fraction_y),
        (floor_x + 1, floor_y + 1, fraction_x * fraction_y),
    )
    flat_values = values.reshape(-1, channels)
    for x_target, y_target, bilinear in neighbors:
        x_long, y_long = x_target.long(), y_target.long()
        valid = finite & (x_long >= 0) & (x_long < width) & (y_long >= 0) & (y_long < height)
        contribution = bilinear * confidence
        valid &= contribution > 0
        indices = (y_long[valid] * width + x_long[valid]).reshape(-1)
        weights.scatter_add_(0, indices, contribution[valid])
        numerator.scatter_add_(0, indices[:, None].expand(-1, channels),
                               flat_values[valid.reshape(-1)] * contribution[valid, None])
    valid_target = weights > 1e-8
    output = torch.zeros_like(numerator)
    output[valid_target] = numerator[valid_target] / weights[valid_target, None]
    propagated_confidence = weights.clamp(0, 1).reshape(height, width)
    return output.reshape(height, width, channels).to(features.dtype), valid_target.reshape(height, width), propagated_confidence


def forward_splat(features: Any, displacement: Any, confidence: Any | None = None) -> tuple[Any, Any, Any]:
    if _is_torch(features):
        return forward_splat_torch(features, displacement, confidence)
    return forward_splat_numpy(features, displacement, confidence)


def recurrent_forward_splat(features: Any, displacements: list[Any], confidence: Any | None = None) -> tuple[Any, Any, list[float], Any]:
    """Recurrently splat features while carrying validity/confidence forward."""
    current = features
    if confidence is None:
        if _is_torch(features):
            import torch
            confidence = torch.ones(features.shape[:2], dtype=torch.float32, device=features.device)
        else:
            confidence = np.ones(features.shape[:2], dtype=np.float32)
    coverage = []
    valid = confidence > 0
    for displacement in displacements:
        current, valid, confidence = forward_splat(current, displacement, confidence)
        if _is_torch(valid):
            coverage.append(float(valid.float().mean().item()))
        else:
            coverage.append(float(np.mean(valid)))
    return current, valid, coverage, confidence
