"""Pure aggregation helpers for the one-window R4a efficiency diagnostic."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def event_read_reuse_evidence(anchor: int, max_offset: int) -> dict[str, Any]:
    if max_offset <= 0:
        raise ValueError("max_offset must be positive")
    reads = [frame for offset in range(1, max_offset + 1)
             for frame in (anchor + offset - 1, anchor + offset)]
    counts = Counter(reads)
    return {
        "logical_read_sequence": reads,
        "total_packet_reads_without_cache": len(reads),
        "unique_event_packets": len(counts),
        "repeated_packet_reads": len(reads) - len(counts),
        "per_frame_read_count": {str(frame): count for frame, count in sorted(counts.items())},
        "adjacent_packet_reuse_implemented": False,
    }


def stage_shares(row: dict[str, Any], stage_keys: Iterable[str],
                 total_key: str) -> list[dict[str, Any]]:
    total = float(row[total_key])
    if total <= 0:
        raise ValueError("total timing must be positive")
    values = [(key, float(row.get(key) or 0.0)) for key in stage_keys]
    diagnostic_sum = sum(value for _key, value in values)
    result = []
    for key, value in values:
        result.append({"stage": key, "cpu_wall_ms": value,
                       "ratio_to_separately_measured_normal_path": value / total,
                       "share_within_synchronized_diagnostic_stages":
                           value / diagnostic_sum if diagnostic_sum > 0 else None})
    return sorted(result, key=lambda item: item["cpu_wall_ms"], reverse=True)


def query_latency_record(prefill_ms: float, decode_ms: float, decode_steps: int,
                         complete_ms: float, output_tokens: int) -> dict[str, Any]:
    if min(prefill_ms, decode_ms, complete_ms) < 0 or decode_steps < 0 or output_tokens < 0:
        raise ValueError("query timings and counts must be non-negative")
    return {
        "ttft_ms": float(prefill_ms),
        "decode_ms": float(decode_ms),
        "decode_steps": int(decode_steps),
        "decode_ms_per_token": (float(decode_ms) / decode_steps if decode_steps else None),
        "complete_answer_latency_ms": float(complete_ms),
        "output_token_count": int(output_tokens),
    }


def repeated_work_evidence(event_reads_during_queries: int,
                           state_rebuilds_during_queries: int,
                           fresh_input_clone_per_query: bool) -> dict[str, Any]:
    return {
        "event_reads_during_queries": int(event_reads_during_queries),
        "state_rebuilds_during_queries": int(state_rebuilds_during_queries),
        "queries_reuse_completed_visual_states": state_rebuilds_during_queries == 0,
        "fresh_query_inputs_cloned_per_query": bool(fresh_input_clone_per_query),
        "claim_of_eliminable_overhead": False,
        "interpretation": (
            "Repeated queries do not rebuild Event state; input tensors are cloned to prevent KV/rope "
            "cross-query contamination. Adjacent transition packet reuse is evaluated separately."
        ),
    }
