from __future__ import annotations

import json
import time
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from src.event_delta.qwen_injection import (
    QwenVisualInjectionAdapter,
    VisualState,
    audit_qwen_interface,
    contract_failure_message,
    causal_hole_fill,
    fresh_query_inputs,
    injection_passes,
    tree_tensors,
)
from src.event_delta.r4_efficiency import (
    interleaved_schedule,
    measured_stage,
    r4_decision,
    validate_complete_timing,
    REQUIRED_STAGES,
)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                    reason="real Torch/Qwen injection equivalence requires the server GPU environment")
def test_real_torch_runtime_is_deferred_to_server_probe() -> None:
    import torch
    assert torch is not None


class FakeVision:
    def __init__(self) -> None:
        self.forward = self.actual_forward
        self.hooks = []

    def actual_forward(self, *_args, **_kwargs):
        output = (np.ones((4, 3)), [np.full((4, 3), 2), np.full((4, 3), 3)])
        for hook in list(self.hooks):
            hook(self, (), output)
        return output

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        parent = self

        class Handle:
            def remove(self):
                parent.hooks.remove(hook)
        return Handle()


def fake_state(output=None) -> VisualState:
    output = output if output is not None else (
        np.ones((4, 3)), [np.full((4, 3), 2), np.full((4, 3), 3)])
    return VisualState(output, (1, 4, 4), (2, 2), 2, "output[0]",
                       ("output[0]", "output[1][0]", "output[1][1]"),
                       ("output[1][0]", "output[1][1]"),
                       {"output[0]": (2, 2), "output[1][0]": (2, 2), "output[1][1]": (2, 2)},
                       "float32", "cpu", 1, 0, True)


def test_complete_vision_output_replacement_keeps_deepstack_and_bypasses_compute() -> None:
    vision = FakeVision()
    model = SimpleNamespace(config=SimpleNamespace(image_token_id=99))
    adapter = QwenVisualInjectionAdapter(model, vision, 2, {"status": "confirmed_from_installed_source"})
    state = fake_state()
    with adapter.replace(state):
        returned = vision.forward("pixels")
    assert returned is state.vision_output
    assert len(tree_tensors(returned)) == 3
    assert len(state.deepstack_paths) == 2
    assert adapter.counters["cached_replacement_count"] == 1
    assert adapter.counters["actual_vision_computation_count"] == 0
    assert vision.forward == vision.actual_forward


def test_replacement_restores_after_exception() -> None:
    vision = FakeVision()
    adapter = QwenVisualInjectionAdapter(SimpleNamespace(), vision, 2,
                                         {"status": "confirmed_from_installed_source"})
    original = vision.forward
    with pytest.raises(RuntimeError):
        with adapter.replace(fake_state()):
            raise RuntimeError("boom")
    assert vision.forward == original


def test_capture_counts_actual_computation_and_hook_is_removed() -> None:
    vision = FakeVision()
    adapter = QwenVisualInjectionAdapter(SimpleNamespace(), vision, 2,
                                         {"status": "confirmed_from_installed_source"})
    with adapter.capture() as captured:
        vision.forward()
    assert len(tree_tensors(captured["output"])) == 3
    assert adapter.counters["actual_vision_computation_count"] == 1
    assert vision.hooks == []


def test_token_grid_placeholder_contract() -> None:
    vision = FakeVision()
    model = SimpleNamespace(config=SimpleNamespace(image_token_id=99))
    adapter = QwenVisualInjectionAdapter(model, vision, 2, {"status": "confirmed_from_installed_source"})
    inputs = {"input_ids": np.asarray([[1, 99, 99, 99, 99, 2]]),
              "image_grid_thw": np.asarray([[1, 4, 4]])}
    report = adapter.validate_placeholder_contract(inputs, fake_state())
    assert report["matches"] and report["image_placeholder_count"] == 4


def test_holes_use_only_previous_state_and_no_target_rgb() -> None:
    previous = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    warped = np.full_like(previous, 100)
    valid = np.asarray([[True, False], [False, True]])
    filled = causal_hole_fill(warped, previous, valid)
    np.testing.assert_array_equal(filled[~valid], previous[~valid])
    np.testing.assert_array_equal(filled[valid], warped[valid])


def test_fresh_query_removes_cross_query_cache_and_rope_state() -> None:
    inputs = {"input_ids": np.asarray([[1, 2]]), "past_key_values": "old",
              "cache_position": "old-position", "rope_deltas": "old-rope"}
    fresh = fresh_query_inputs(inputs)
    assert fresh["past_key_values"] is None
    assert "cache_position" not in fresh and "rope_deltas" not in fresh
    assert fresh["input_ids"] is not inputs["input_ids"]


class SourceAuditedModel:
    config = SimpleNamespace()

    def get_image_features(self, pixel_values=None, image_grid_thw=None):
        deepstack_visual_embeds = image_grid_thw
        return pixel_values, deepstack_visual_embeds

    def get_placeholder_mask(self, input_ids=None, inputs_embeds=None, image_features=None):
        image_token_id = 1
        image_mask = input_ids == image_token_id
        masked_scatter = inputs_embeds, image_features
        return image_mask, masked_scatter

    def forward(self, pixel_values=None, image_grid_thw=None, position_ids=None,
                deepstack_visual_embeds=None):
        image_features = self.get_image_features(pixel_values, image_grid_thw)
        image_mask = self.get_placeholder_mask(None, image_features=image_features)
        masked_scatter = image_mask
        rope_deltas = position_ids
        return masked_scatter, rope_deltas, deepstack_visual_embeds

    def prepare_inputs_for_generation(self, past_key_values=None, cache_position=None,
                                      image_grid_thw=None):
        return past_key_values, cache_position, image_grid_thw

    def get_rope_index(self, image_grid_thw=None):
        mrope = image_grid_thw
        return mrope


class SourceAuditedVision:
    def forward(self, pixel_values=None, grid_thw=None):
        deepstack = grid_thw
        return pixel_values, deepstack


def test_installed_source_contract_audit_records_deepstack_and_generation_logic() -> None:
    report = audit_qwen_interface(SourceAuditedModel(), SourceAuditedVision(),
                                  SimpleNamespace(), {"status": "confirmed_from_installed_source"},
                                  "4.57.6")
    assert report["status"] == "ready"
    assert report["transformers_version"] == "4.57.6"
    assert report["evidence"]["deepstack_in_source"] is True
    assert report["evidence"]["generation_prefill_decode_branch"] is True
    assert all("failure_reason" in check for check in report["checks"])
    assert report["inspected_methods"]["vision.forward"]["inspect_getsource_succeeded"]
    assert report["inspected_methods"]["vision.forward"]["sha256"]


class UnreadableModel(SourceAuditedModel):
    forward = len


class MissingInterfaceModel:
    def forward(self, image_grid_thw=None, position_ids=None):
        return image_grid_thw, position_ids

    def prepare_inputs_for_generation(self, past_key_values=None, cache_position=None):
        return past_key_values, cache_position


class RenamedVision:
    def forward(self, frames=None, layout=None):
        return frames, layout


def test_source_contract_distinguishes_unreadable_and_missing_interfaces() -> None:
    unreadable = audit_qwen_interface(UnreadableModel(), SourceAuditedVision(),
                                      SimpleNamespace(), {"status": "confirmed_from_installed_source"})
    forward_check = next(check for check in unreadable["checks"]
                         if check["name"] == "conditional_generation_forward_source")
    assert not forward_check["passed"] and forward_check["failure_kind"] == "source_unreadable"

    missing = audit_qwen_interface(MissingInterfaceModel(), SourceAuditedVision(),
                                   SimpleNamespace(), {"status": "confirmed_from_installed_source"})
    route = next(check for check in missing["checks"] if check["name"] == "model_routes_pixels_to_vision")
    placeholder = next(check for check in missing["checks"]
                       if check["name"] == "image_placeholder_replacement")
    assert route["failure_kind"] == "interface_missing"
    assert placeholder["failure_kind"] == "interface_missing"
    message = contract_failure_message(missing)
    assert "model_routes_pixels_to_vision" in message
    assert "image_placeholder_replacement" in message

    renamed = audit_qwen_interface(SourceAuditedModel(), RenamedVision(),
                                   SimpleNamespace(), {"status": "confirmed_from_installed_source"})
    signature = next(check for check in renamed["checks"]
                     if check["name"] == "vision_pixel_and_grid_signature")
    assert signature["failure_kind"] == "naming_variant_unresolved"


def test_correctness_probe_writes_source_audit_even_before_runtime(tmp_path) -> None:
    from tools.probe_qwen_injection import main

    output = tmp_path / "probe"
    code = main(["--image", str(tmp_path / "missing.png"), "--output-dir", str(output)])
    assert code == 2
    report = json.loads((output / "source_contract_audit.json").read_text())
    assert report["status"] == "blocked"
    assert report["phase"] == "argument_validation"
    assert "image is missing" in report["failure_summary"][0]


def test_read_only_audit_cli_writes_failure_artifact_without_transformers(tmp_path, monkeypatch) -> None:
    import tools.audit_qwen_source_contract as audit_cli

    def fail_extractor(*_args, **_kwargs):
        raise RuntimeError("runtime deliberately unavailable")

    monkeypatch.setattr(audit_cli, "QwenVisionExtractor", fail_extractor)
    output = tmp_path / "source_contract_audit.json"
    code = audit_cli.main(["--output", str(output), "--local-files-only"])
    assert code == 2
    report = json.loads(output.read_text())
    assert report["read_only_source_audit"] is True
    assert report["vision_or_language_forward_executed"] is False
    assert "runtime deliberately unavailable" in report["terminal_error"]


def test_equivalence_failure_blocks() -> None:
    metrics = {"injected_vs_native": {"max_absolute_difference": 0.1,
                                       "mean_absolute_difference": 0.01},
               "native_next_token_argmax": 1, "injected_next_token_argmax": 2}
    ids = np.asarray([[1, 2]])
    assert not injection_passes(metrics, ids, ids, 1e-5, 1e-6, True)
    assert r4_decision(False, 10.0, 5.0)["status"] == "blocked"


class CpuDevice:
    type = "cpu"


def test_stage_wall_time_is_independent_and_schedule_rotates() -> None:
    record = {}
    with measured_stage(SimpleNamespace(), CpuDevice(), record, "complete_query"):
        time.sleep(0.001)
    for stage in REQUIRED_STAGES:
        record.setdefault(f"{stage}_cpu_wall_ms", 0.0)
        record.setdefault(f"{stage}_gpu_ms", None)
    record["complete_query_derived_from_stage_percentiles"] = False
    validate_complete_timing(record)
    assert record["complete_query_cpu_wall_ms"] > 0
    assert interleaved_schedule(["a", "b", "c"], 2) == ["a", "b", "c", "b", "c", "a"]
