from __future__ import annotations

import pytest

from src.event_delta.r4_diagnostics import (
    event_read_reuse_evidence,
    query_latency_record,
    repeated_work_evidence,
    stage_shares,
)


def test_adjacent_transition_packet_reuse_is_reported_without_optimizing() -> None:
    report = event_read_reuse_evidence(100, 9)
    assert report["total_packet_reads_without_cache"] == 18
    assert report["unique_event_packets"] == 10
    assert report["repeated_packet_reads"] == 8
    assert report["per_frame_read_count"]["101"] == 2
    assert report["adjacent_packet_reuse_implemented"] is False


def test_stage_shares_keep_normal_total_separate_from_synchronized_diagnostic() -> None:
    rows = stage_shares({"complete": 100.0, "a": 80.0, "b": 40.0}, ("a", "b"), "complete")
    assert rows[0]["ratio_to_separately_measured_normal_path"] == pytest.approx(0.8)
    assert sum(row["share_within_synchronized_diagnostic_stages"] for row in rows) == pytest.approx(1.0)


def test_query_latency_separates_ttft_decode_rate_and_complete_answer() -> None:
    record = query_latency_record(12.0, 40.0, 20, 55.0, 21)
    assert record["ttft_ms"] == 12.0
    assert record["decode_ms_per_token"] == 2.0
    assert record["complete_answer_latency_ms"] == 55.0
    assert query_latency_record(1.0, 0.0, 0, 1.0, 1)["decode_ms_per_token"] is None


def test_repeated_queries_report_reuse_and_fresh_input_copy_boundary() -> None:
    evidence = repeated_work_evidence(0, 0, True)
    assert evidence["queries_reuse_completed_visual_states"] is True
    assert evidence["fresh_query_inputs_cloned_per_query"] is True
    assert evidence["claim_of_eliminable_overhead"] is False


def test_diagnostic_refuses_to_overwrite_existing_artifacts(tmp_path) -> None:
    from tools.diagnose_r4_efficiency import ensure_new_output_directory

    output = tmp_path / "diagnostic"
    output.mkdir()
    (output / "existing.json").write_text("{}")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ensure_new_output_directory(output)
