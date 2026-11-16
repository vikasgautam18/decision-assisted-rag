"""Private, synchronous, fail-closed local observation export."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
_INSTANCE_ID = uuid.uuid4().hex
_RECORDER: ContextVar[Recorder | None] = ContextVar("observation", default=None)
_PARENT: ContextVar[str | None] = ContextVar("observation_parent", default=None)
_KEYS = frozenset(
    """
experiment_id phase arm pass pass_id case_id logical_request_id observation_id attempt_id
pass_number attempt ordinal concurrency target_concurrency response_headers_seconds
response_body_sent_seconds grading_attempt grading_call_id evaluator_deployment sku
process_cpu_interval_seconds process_peak_rss_kib process_metrics_status
configuration_hash config_hash instance_id service provider role category deployment model
model_version
api_version resource_id region operation method path_hash payload_hash prompt_hash schema_hash
request_bytes response_bytes requested_output_limit generation_settings usage usage_status
input_tokens output_tokens cached_input_tokens cache_write_tokens reasoning_tokens image_tokens
audio_tokens request_id client_request_id response_id request_ids finish_reason ledger_id
call_id state http_status
retry_after retry_origin retry_index sdk_retries error_type error_code cancellation_reason
terminal_state
deadline_reason estimated_tokens token_capacity request_capacity wait_seconds wait_reason
result_count historical figures platform top question_hash vector_hash index_hash
transport_timing_status admitted transport_status parse_status endpoint_hash elapsed_seconds
backend native_attempt native_attempts sdk_attempt sdk_http_attempts transport_call_id
request_sha256 endpoint_host provider_ids normalized_usage id status service_tier _request_id
output baseline_error_response
""".split()
)
_HEADER_KEYS = (
    "x-ms-request-id",
    "apim-request-id",
    "x-request-id",
    "request-id",
    "x-ms-client-request-id",
    "traceparent",
)


def payload_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def response_ids(headers) -> dict:
    return {key: headers[key] for key in _HEADER_KEYS if key in headers}


def _safe(attributes: dict) -> dict:
    return {key: value for key, value in attributes.items() if key in _KEYS}


class Recorder:
    def __init__(self, directory: Path, metadata: dict):
        self.trace_id = uuid.uuid4().hex
        self.metadata = {
            **dict.fromkeys(
                (
                    "experiment_id",
                    "phase",
                    "arm",
                    "pass",
                    "case_id",
                    "logical_request_id",
                    "attempt_id",
                    "configuration_hash",
                )
            ),
            "instance_id": _INSTANCE_ID,
            **_safe(metadata),
        }
        self.lock = threading.Lock()
        self.failure: BaseException | None = None
        directory = Path(directory)
        if any(parent.is_symlink() for parent in (directory, *directory.parents)):
            raise ValueError("observation directory must not traverse symlinks")
        # Do not silently follow a user-controlled symlink to a different export destination.
        missing = []
        current = directory
        while not current.exists():
            missing.append(current)
            current = current.parent
        for parent in reversed(missing):
            parent.mkdir(mode=0o700, exist_ok=True)
            parent.chmod(0o700)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("observation directory must be a real directory")
        directory.chmod(0o700)
        self.path = directory / f"{self.trace_id}.jsonl"
        self.fd = os.open(
            self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )

    def emit(self, event: dict) -> None:
        with self.lock:
            if self.failure is not None:
                raise self.failure
            try:
                payload = (
                    json.dumps(
                        {
                            **self.metadata,
                            **event,
                            "schema_version": SCHEMA_VERSION,
                            "trace_id": self.trace_id,
                            "process_id": os.getpid(),
                        },
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode()
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(self.fd, remaining)
                    if written <= 0:
                        raise OSError("observation export made no progress")
                    remaining = remaining[written:]
                os.fsync(self.fd)
            except BaseException as exc:
                self.failure = exc
                raise


@contextmanager
def observation(directory: Path, metadata: dict):
    recorder = Recorder(directory, metadata)
    token = _RECORDER.set(recorder)
    parent = _PARENT.set(None)
    try:
        with span("observation") as metrics:
            cpu_started = time.process_time()
            try:
                yield recorder
            finally:
                metrics["process_cpu_interval_seconds"] = time.process_time() - cpu_started
                if sys.platform == "linux":
                    import resource

                    metrics["process_peak_rss_kib"] = resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    metrics["process_metrics_status"] = (
                        "process_interval_cpu_and_lifetime_peak_rss_not_per_request_allocation"
                    )
                else:
                    metrics["process_metrics_status"] = "cpu_only_rss_units_not_normalized"
    finally:
        _PARENT.reset(parent)
        _RECORDER.reset(token)
        os.close(recorder.fd)


@contextmanager
def span(name: str, **attributes):
    recorder = _RECORDER.get()
    if recorder is None:
        yield attributes
        return
    span_id = uuid.uuid4().hex
    started = time.monotonic()
    utc = datetime.now(UTC).isoformat()
    identity = {"name": name, "span_id": span_id, "parent_span_id": _PARENT.get()}
    attributes.update(
        trace_id=recorder.trace_id, span_id=span_id, parent_span_id=identity["parent_span_id"]
    )
    recorder.emit(
        {
            **identity,
            "event": "span_start",
            "status": "started",
            "utc": utc,
            "monotonic": started,
            "start_utc": utc,
            "start_monotonic": started,
            "attributes": _safe(attributes),
        }
    )
    token = _PARENT.set(span_id)
    status = "ok"
    try:
        yield attributes
    except BaseException as exc:
        status = (
            "cancelled"
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, GeneratorExit))
            else "error"
        )
        attributes["error_type"] = type(exc).__name__
        if status == "cancelled":
            attributes["cancellation_reason"] = "task_or_context_cancelled"
        raise
    finally:
        _PARENT.reset(token)
        ended = time.monotonic()
        end_utc = datetime.now(UTC).isoformat()
        recorder.emit(
            {
                **identity,
                "event": "span_end",
                "status": status,
                "utc": end_utc,
                "monotonic": ended,
                "start_utc": utc,
                "end_utc": end_utc,
                "start_monotonic": started,
                "end_monotonic": ended,
                "duration_seconds": ended - started,
                "attributes": _safe(attributes),
            }
        )


def normalized_usage(usage: dict | None) -> dict:
    """Keep unknown counters null; reasoning is a subset of output, not extra output."""
    usage = usage if isinstance(usage, dict) else {}
    inputs = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    outputs = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    return {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "cached_input_tokens": inputs.get("cached_tokens"),
        "cache_write_tokens": inputs.get(
            "cache_write_tokens", usage.get("cache_creation_input_tokens")
        ),
        "reasoning_tokens": outputs.get("reasoning_tokens"),
        "image_tokens": inputs.get("image_tokens"),
        "audio_tokens": inputs.get("audio_tokens"),
    }
