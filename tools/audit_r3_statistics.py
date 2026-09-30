#!/usr/bin/env python3
"""Reaggregate an existing R3b per-offset CSV without rerunning GPU inference."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from event_delta.json_io import atomic_write_json, json_dumps  # noqa: E402
from event_delta.r3_propagation import (  # noqa: E402
    expanded_decision,
    observation_ratio_diagnostics,
    paired_window_bootstrap,
    stable_aggregate_report,
)
from tools.run_r2_oracle import write_csv  # noqa: E402


IDENTIFIER_FIELDS = {"anchor_frame_id", "source_frame_id", "target_frame_id", "flow_frame_id", "offset"}
STRING_FIELDS = {
    "recording", "activity_stratum", "activity_group", "evaluation_mask_definition",
    "old_event_path", "new_event_path", "gt_flow_path",
}
REQUIRED_FIELDS = {
    "recording", "anchor_frame_id", "offset", "activity_group",
    "all_valid_token_count", "motion_token_count",
    *{f"{scope}_{method}_{distance}" for scope in ("all_valid", "motion")
      for method in ("stale", "raw_eraft", "masked_eraft", "gt_oracle")
      for distance in ("cosine_distance", "l1_distance", "l2_distance")},
    "raw_eraft_coverage_ratio", "gt_oracle_coverage_ratio", "complete_online_delta_path_ms",
}


def read_metrics_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError("existing CSV lacks paired aggregation fields: " + ", ".join(missing))
        rows = []
        for raw in reader:
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in STRING_FIELDS:
                    row[key] = value
                elif value == "" or value is None:
                    row[key] = None
                elif key in IDENTIFIER_FIELDS:
                    row[key] = int(float(value))
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            rows.append(row)
    if not rows:
        raise ValueError("per-offset CSV is empty")
    return rows


def _matplotlib() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "eventdelta-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_stable_reference(stats: list[dict[str, Any]], path: Path) -> None:
    """Plot offset ratio-of-aggregates; N/A gaps remain blank instead of spiking."""
    plt = _matplotlib()
    fig, axis = plt.subplots(figsize=(8, 4.5))
    offsets = sorted({int(row["group"]) for row in stats if row["group_type"] == "offset"})
    for scope, linestyle in (("all_valid", "-"), ("motion", "--")):
        for method, marker in (("raw_eraft", "o"), ("masked_eraft", "s")):
            values = []
            for offset in offsets:
                match = [row for row in stats if row["group_type"] == "offset"
                         and row["group"] == str(offset) and row["scope"] == scope
                         and row["aggregation"] == "window_macro" and row["method"] == method
                         and row["distance_metric"] == "cosine_distance"]
                value = match[0]["stable_reference_ratio"] if len(match) == 1 else None
                values.append(np.nan if value is None else value)
            axis.plot(offsets, values, linestyle=linestyle, marker=marker,
                      label=f"{method}/{scope}")
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="offset from anchor", ylabel="stable Event / GT-flow improvement ratio",
             title="Ratio of aggregate improvements (small GT denominator = N/A)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _stat(stats: list[dict[str, Any]], scope: str, method: str) -> dict[str, Any]:
    matches = [row for row in stats if row["group_type"] == "overall" and row["group"] == "overall"
               and row["scope"] == scope and row["aggregation"] == "window_macro"
               and row["method"] == method and row["distance_metric"] == "cosine_distance"]
    if len(matches) != 1:
        raise ValueError(f"missing overall statistic for {scope}/{method}")
    return matches[0]


def summary_markdown(stats: list[dict[str, Any]], diagnostics: list[dict[str, Any]],
                     bootstrap: dict[str, Any], decision: dict[str, Any]) -> str:
    raw_all = _stat(stats, "all_valid", "raw_eraft")
    masked_all = _stat(stats, "all_valid", "masked_eraft")
    raw_motion = _stat(stats, "motion", "raw_eraft")
    masked_motion = _stat(stats, "motion", "masked_eraft")
    gt_motion = _stat(stats, "motion", "gt_oracle")
    result = ("promising exploratory smoke" if decision["status"] == "expanded"
              else "exploratory stop/diagnostic smoke")
    return f"""# R3b statistical and decision audit

Decision: **{decision['status']}**. Current result: **{result}**.

## Stable aggregate findings

- Raw Event improves motion tokens but hurts all-valid tokens: motion recovery={raw_motion['aggregate_recovery']}; all-valid recovery={raw_all['aggregate_recovery']}.
- The fixed Event-support mask restores positive all-valid recovery: masked all-valid recovery={masked_all['aggregate_recovery']}; masked motion recovery={masked_motion['aggregate_recovery']}.
- GT-flow geometric reference motion recovery={gt_motion['aggregate_recovery']}. It is a geometric reference, not a strict upper bound.
- Raw stable reference ratio={raw_motion['stable_reference_ratio']}; masked={masked_motion['stable_reference_ratio']}. These are ratios of aggregate absolute improvements, may exceed 1, and are informational only.
- Gate details: `{json_dumps(decision['criteria'])}`.

## Statistical boundary

- Stable recovery first aggregates stale and method distances, then computes their ratio. The legacy mean of observation-level ratios is retained only in `observation_ratio_diagnostics.csv` and is never used by the gate or main plot.
- Bootstrap sampling unit is the complete window with all nine offsets. Window count={bootstrap['window_count']}; exploratory={str(bootstrap['exploratory']).lower()}. The 95% intervals are reported but are not a hard smoke gate.
- Motion windows without motion tokens are N/A rather than failures and are excluded from eligible motion-window aggregates.
- This is not yet a VLM downstream accuracy result.
- This is not yet a real-time system result; measured timing is a smoke profile, not an end-to-end deployment claim.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-offset-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260929)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.per_offset_metrics.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source.is_file() or args.bootstrap_replicates <= 0:
        print("error: metrics CSV must exist and bootstrap replicates must be positive", file=sys.stderr)
        return 2
    try:
        rows = read_metrics_csv(source)
        stats = stable_aggregate_report(rows)
        diagnostics = observation_ratio_diagnostics(rows)
        bootstrap = paired_window_bootstrap(rows, args.bootstrap_replicates, args.seed)
        recordings = sorted({str(row["recording"]) for row in rows})
        decision = expanded_decision(rows, recordings, stable_stats=stats)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "aggregate_metrics.csv", stats)
    write_csv(output / "observation_ratio_diagnostics.csv", diagnostics)
    atomic_write_json(output / "bootstrap.json", bootstrap)
    atomic_write_json(output / "decision.json", {
        "source_per_offset_metrics": str(source), "source_rows_modified": False,
        "gpu_rerun_required": False, "decision": decision,
        "aggregation_contract": "aggregate paired distances first; calculate ratios second",
    })
    plot_stable_reference(stats, output / "oracle_gap_closed.png")
    (output / "summary.md").write_text(
        summary_markdown(stats, diagnostics, bootstrap, decision), encoding="utf-8")
    print(f"wrote statistical audit to {output}; decision={decision['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
