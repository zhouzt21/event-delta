#!/usr/bin/env python3
"""Probe Qwen vision feature stages without language-model generation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from event_delta.json_io import atomic_write_json, json_dumps  # noqa: E402
from event_delta.qwen_features import (  # noqa: E402
    ProbeError,
    QwenVisionExtractor,
    TokenOrderError,
    require_runtime,
    select_device_name,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
                              text=True, capture_output=True, check=False).stdout.strip() or None
    report = {"scope": "Qwen vision-only R2a probe", "model": args.model,
              "image": str(args.image.expanduser().resolve()),
              "local_files_only": args.local_files_only, "git_commit": revision}
    try:
        if not args.image.is_file():
            raise ProbeError(f"image does not exist: {args.image}")
        torch, _transformers = require_runtime()
        startup_runtime = {"requested_device": args.device,
                           "selected_device": select_device_name(args.device, bool(torch.cuda.is_available())),
                           "torch_cuda_is_available": bool(torch.cuda.is_available()),
                           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        report["runtime"] = startup_runtime
        atomic_write_json(output, report)
        print(json_dumps(startup_runtime), flush=True)
        print("loading processor/model", flush=True)
        extractor = QwenVisionExtractor(args.model, args.local_files_only, args.device)
        inputs = extractor.preprocess(args.image)
        feature, details = extractor.encode_inputs(inputs, "post_merger")
        details["runtime"] = extractor.runtime_diagnostics(inputs, feature, "post_merger")
        report.update(details)
        print(json_dumps(details["runtime"]), flush=True)
        if report["token_order"]["status"] != "confirmed_from_installed_source":
            raise TokenOrderError(report["token_order"]["warning"])
    except (ProbeError, OSError, ValueError) as exc:
        report.update(status="blocked", error=f"{type(exc).__name__}: {exc}",
                      stop_reason="feature propagation was not attempted")
        atomic_write_json(output, report)
        print(f"blocked: {exc}; wrote {output}", file=sys.stderr)
        return 2
    atomic_write_json(output, report)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
