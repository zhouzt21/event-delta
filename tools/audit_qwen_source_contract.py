#!/usr/bin/env python3
"""Read-only audit of the installed Qwen visual-injection source contract.

This loads the locally available model/processor to discover their actual classes, but it
does not run the vision tower, language model, generation, or write to site-packages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from event_delta.json_io import atomic_write_json  # noqa: E402
from event_delta.qwen_features import QwenVisionExtractor  # noqa: E402
from event_delta.qwen_injection import (  # noqa: E402
    audit_qwen_interface,
    contract_failure_message,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--expected-transformers-version", default="4.57.6")
    return parser.parse_args(argv)


def initial_report() -> dict[str, Any]:
    return {
        "status": "blocked",
        "phase": "runtime_initialization",
        "transformers_version": None,
        "actual_classes": {},
        "inspected_methods": {},
        "checks": [],
        "failed_required_checks": ["runtime_initialization"],
        "failure_summary": ["runtime_initialization: model has not been loaded"],
        "evidence": {},
        "vision_return_structure": {
            "runtime_observed": False,
            "reason": "source-only audit intentionally does not execute the vision tower",
        },
        "runtime_vision_return": {
            "observed": False,
            "reason": "source-only audit intentionally does not execute the vision tower",
        },
        "read_only_source_audit": True,
        "vision_or_language_forward_executed": False,
        "transformers_site_packages_modified": False,
        "global_monkey_patch": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = initial_report()
    atomic_write_json(output, report)
    try:
        extractor = QwenVisionExtractor(args.model, args.local_files_only, args.device)
        version = extractor.transformers.__version__
        report = audit_qwen_interface(
            extractor.model,
            extractor.structure["vision"],
            extractor.processor,
            extractor.order,
            version,
        )
        report.update(
            phase="source_contract",
            expected_transformers_version=args.expected_transformers_version,
            version_matches_expected=version == args.expected_transformers_version,
            read_only_source_audit=True,
            vision_or_language_forward_executed=False,
        )
        report["runtime_vision_return"] = {
            "observed": False,
            "reason": "source-only audit intentionally does not execute the vision tower",
        }
        if not report["version_matches_expected"]:
            report["status"] = "blocked"
            report["failed_required_checks"].insert(0, "transformers_version")
            report["failure_summary"].insert(
                0,
                f"transformers_version: expected {args.expected_transformers_version}, got {version}",
            )
        atomic_write_json(output, report)
    except Exception as exc:
        report["status"] = "blocked"
        report["terminal_error"] = f"{type(exc).__name__}: {exc}"
        report["failure_summary"] = report.get("failure_summary", []) + [report["terminal_error"]]
        atomic_write_json(output, report)
        print(f"blocked: {report['terminal_error']}; audit={output}", file=sys.stderr)
        return 2
    if report["status"] != "ready":
        print(f"blocked: {contract_failure_message(report)}; audit={output}", file=sys.stderr)
        return 2
    print(f"source contract ready; audit={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
