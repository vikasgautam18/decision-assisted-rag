import asyncio
import json
import os
import stat

import pytest

from clients import telemetry
from clients.telemetry import normalized_usage, observation, span


def events(recorder):
    return [json.loads(line) for line in recorder.path.read_text().splitlines()]


def test_private_append_only_hierarchy_and_context_propagation(tmp_path):
    async def scenario():
        with observation(tmp_path / "private" / "events", {"arm": "G", "secret": "hidden"}) as rec:
            with span("parent") as attributes:
                attributes["result_count"] = 2
                attributes["authorization"] = "hidden-token"

                async def child():
                    with span("async_child"):
                        pass

                def worker():
                    with span("thread_child"):
                        pass

                await asyncio.gather(child(), asyncio.to_thread(worker))
            return rec

    recorder = asyncio.run(scenario())
    rows = events(recorder)
    starts = {row["name"]: row for row in rows if row["event"] == "span_start"}
    assert starts["parent"]["parent_span_id"] == starts["observation"]["span_id"]
    for name in ("async_child", "thread_child"):
        assert starts[name]["parent_span_id"] == starts["parent"]["span_id"]
    assert {row["trace_id"] for row in rows} == {recorder.trace_id}
    for row in rows:
        assert row["utc"].endswith("+00:00") and row["arm"] == "G"
        if row["event"] == "span_end":
            assert row["duration_seconds"] == row["end_monotonic"] - row["start_monotonic"]
            assert row["status"] == "ok"
    assert "hidden" not in recorder.path.read_text()
    assert stat.S_IMODE(recorder.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(recorder.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(recorder.path.parent.parent.stat().st_mode) == 0o700


def test_inactive_span_has_no_io_and_mutable_attributes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected file access")

    monkeypatch.setattr(os, "open", forbidden)
    with span("legacy", role="writer") as attributes:
        attributes["state"] = "completed"
    assert attributes["state"] == "completed"


def test_failure_cancellation_and_context_reset_do_not_export_error_text(tmp_path):
    with observation(tmp_path, {}) as recorder:
        with pytest.raises(ValueError):
            with span("failure"):
                raise ValueError("secret signed-url token")
        with pytest.raises(asyncio.CancelledError):
            with span("cancelled"):
                raise asyncio.CancelledError("private reason")
        with span("sibling"):
            pass
    rows = events(recorder)
    ends = {row["name"]: row for row in rows if row["event"] == "span_end"}
    assert ends["failure"]["status"] == "error"
    assert ends["cancelled"]["status"] == "cancelled"
    assert ends["sibling"]["parent_span_id"] == ends["observation"]["span_id"]
    assert "secret" not in recorder.path.read_text()
    assert "private reason" not in recorder.path.read_text()


@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
def test_each_boundary_is_fsynced_and_export_failures_propagate(tmp_path, monkeypatch, fail_at):
    sync = os.fsync
    calls = 0

    def fsync(fd):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("disk unavailable")
        sync(fd)

    monkeypatch.setattr(telemetry.os, "fsync", fsync)
    with pytest.raises(OSError, match="disk unavailable"):
        with observation(tmp_path, {}):
            with span("operation"):
                pass
    with span("context_was_reset"):
        pass


def test_nested_observation_restores_parent_and_rejects_symlink(tmp_path):
    with observation(tmp_path / "outer", {}) as outer:
        with span("parent"):
            with observation(tmp_path / "inner", {}):
                pass
            with span("restored"):
                pass
    starts = {row["name"]: row for row in events(outer) if row["event"] == "span_start"}
    assert starts["restored"]["parent_span_id"] == starts["parent"]["span_id"]
    (tmp_path / "link").symlink_to(tmp_path / "outer", target_is_directory=True)
    with pytest.raises(ValueError):
        with observation(tmp_path / "link", {}):
            pass


def test_export_failure_latches_and_blocks_subsequent_work(tmp_path, monkeypatch):
    original = os.fsync
    dispatched = []

    def fail(fd):
        raise OSError("export interrupted")

    with pytest.raises(OSError, match="export interrupted"):
        with observation(tmp_path, {}):
            monkeypatch.setattr(telemetry.os, "fsync", fail)
            with pytest.raises(OSError):
                with span("failed"):
                    pass
            monkeypatch.setattr(telemetry.os, "fsync", original)
            with span("must_not_dispatch"):
                dispatched.append(True)
    assert dispatched == []


def test_usage_unknown_zero_and_reasoning_subset():
    assert all(value is None for value in normalized_usage(None).values())
    usage = {
        "prompt_tokens": 0, "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 12},
    }
    counters = normalized_usage(usage)
    assert counters["input_tokens"] == 0
    assert counters["cached_input_tokens"] == 0
    assert counters["output_tokens"] == 20
    assert counters["reasoning_tokens"] == 12
    assert counters["image_tokens"] is None
