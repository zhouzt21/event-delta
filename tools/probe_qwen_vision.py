#!/usr/bin/env python3
"""Probe Qwen vision feature stages without language-model generation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from event_delta.qwen_features import ProbeError, QwenVisionExtractor, TokenOrderError  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda")
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
        extractor = QwenVisionExtractor(args.model, args.local_files_only, args.device)
        details = extractor.probe_image(args.image)
        report.update(details)
        if report["token_order"]["status"] != "confirmed_from_installed_source":
            raise TokenOrderError(report["token_order"]["warning"])
    except (ProbeError, OSError, ValueError) as exc:
        report.update(status="blocked", error=f"{type(exc).__name__}: {exc}",
                      stop_reason="feature propagation was not attempted")
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"blocked: {exc}; wrote {output}", file=sys.stderr)
        return 2
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
