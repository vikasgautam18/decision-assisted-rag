import asyncio
import contextvars
import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from evaluation.baseline_copy import REFERENCE_ENV, SOURCE_SHA, prepare_copy, reference_checkout
from evaluation.baseline_observability import (
    CORRELATION_HEADERS,
    BaselineObservationMiddleware,
    create_baseline_launcher,
    instrument_foundry,
    safe_result,
)

REFERENCE = reference_checkout()
REFERENCE_REQUIRED = f"set {REFERENCE_ENV} to a checkout of the pinned baseline reference"
_active = contextvars.ContextVar("test_baseline_observation", default=None)


class Telemetry:
    def __init__(self, fail=None):
        self.events = []
        self.observations = []
        self.fail = fail

    @contextmanager
    def observation(self, directory, metadata):
        recorder = SimpleNamespace(trace_id="fixture-trace", path=directory / "fixture.jsonl")
        self.observations.append((directory, metadata, recorder))
        token = _active.set(metadata)
        try:
            yield recorder
        finally:
            _active.reset(token)

    @contextmanager
    def span(self, name, **attrs):
        if self.fail == (name, "enter"):
            raise OSError("export unavailable")
        state = dict(attrs)
        self.events.append((name, state, _active.get()))
        try:
            yield state
        except BaseException as exc:
            state["error_type"] = type(exc).__name__
            raise
        finally:
            if self.fail == (name, "exit"):
                raise OSError("export unavailable")


class Dump:
    def __init__(self, value):
        self.value = value

    def model_dump(self, **kwargs):
        return self.value

    def __getattr__(self, name):
        try:
            return self.value[name]
        except KeyError:
            raise AttributeError(name) from None


def response_fixture():
    return SimpleNamespace(
        id="resp-fixture",
        _request_id="request-fixture",
        model="model-fixture",
        status="completed",
        service_tier="default",
        output_text="The answer.",
        usage=Dump(
            {
                "input_tokens": 40,
                "output_tokens": 20,
                "total_tokens": 60,
                "input_tokens_details": {
                    "cached_tokens": 10,
                    "audio_tokens": 2,
                    "image_tokens": 3,
                    "future_category": {"tokens": 7},
                },
                "output_tokens_details": {"reasoning_tokens": 5, "audio_tokens": 1},
                "provider_new_meter": 6,
            }
        ),
        output=[
            Dump(
                {
                    "type": "message",
                    "id": "message-1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "The answer."}],
                }
            ),
            Dump(
                {
                    "type": "mcp_call",
                    "id": "tool-1",
                    "name": "knowledge_base_retrieve",
                    "server_label": "knowledge_base",
                    "arguments": "SECRET",
                    "output": "SECRET",
                    "authorization": "SECRET",
                    "run_id": "run-1",
                    "activity_id": "activity-1",
                }
            ),
        ],
    )


@pytest.fixture
def baseline(monkeypatch):
    """Execute the actual pinned runtime with deterministic offline SDK substitutes."""
    monkeypatch.setenv("AGENT_BACKEND", "foundry")
    if REFERENCE is None or not REFERENCE.is_dir():
        pytest.skip(REFERENCE_REQUIRED)
    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(REFERENCE),
            "show",
            f"{SOURCE_SHA}:backend/agent/ask.py",
        ]
    )

    class ConnectionError_(Exception):
        pass

    class TimeoutError_(Exception):
        pass

    class BadRequestError_(Exception):
        pass

    class Project:
        client = None
        invocations = []

        def __init__(self, **kwargs):
            self.invocations.append(kwargs)

        def get_openai_client(self):
            return self.client

    modules = {
        "openai": SimpleNamespace(
            APIConnectionError=ConnectionError_,
            APITimeoutError=TimeoutError_,
            BadRequestError=BadRequestError_,
        ),
        "azure.ai.projects": SimpleNamespace(AIProjectClient=Project),
        "azure.identity": SimpleNamespace(AzureCliCredential=lambda: "fixture-credential"),
        "backend.agent.setup_agent": SimpleNamespace(
            AGENTS={"fixture-kb": "existing-agent"},
            INSTRUCTIONS="unchanged",
            mcp_url=lambda kb: "https://fixture.invalid/mcp",
        ),
        "common": SimpleNamespace(
            config=SimpleNamespace(
                KB_NAME="fixture-kb",
                AI_ENDPOINT="https://fixture.invalid",
                project_endpoint=lambda: "https://fixture.invalid/project",
            )
        ),
        "common.answer": SimpleNamespace(
            cited_spans=lambda text: [],
            figures_from_text=lambda text: [],
        ),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    ask = ModuleType("baseline_fixture")
    exec(compile(source, "pinned/backend/agent/ask.py", "exec"), ask.__dict__)
    sleeps = []
    ask.time = SimpleNamespace(time=lambda: 100.0, sleep=sleeps.append)
    return SimpleNamespace(
        ask=ask,
        project=Project,
        sleeps=sleeps,
        connection=ConnectionError_,
        timeout=TimeoutError_,
        bad_request=BadRequestError_,
    )


class MockSDK:
    """SDK-shaped fixture with HTTPX transport and the SDK's own retry boundary."""

    def __init__(self, baseline, *, native_failures=(), http_statuses=(200,), terminal=None):
        self.baseline = baseline
        self.native_failures = list(native_failures)
        self.http_statuses = iter(http_statuses)
        self.terminal = terminal
        self.requests = []
        self.creates = []
        self.sdk_retries = []
        self._client = httpx.Client(transport=httpx.MockTransport(self.handle))
        self.responses = SimpleNamespace(create=self.create)

    def handle(self, request):
        self.requests.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": request.content,
            }
        )
        if self.terminal is not None:
            raise self.terminal
        return httpx.Response(next(self.http_statuses), headers={"x-request-id": "transport-id"})

    def create(self, **kwargs):
        self.creates.append(kwargs)
        # Native failures happen before the fixture SDK transport, as can local
        # SDK validation; the original six/two-attempt loops still own policy.
        if self.native_failures:
            error = self.native_failures.pop(0)
            if error is not None:
                raise error
        body = {**kwargs, "tools": [{"authorization": "SECRET"}]}
        for retry in range(3):
            try:
                request = self._client.build_request(
                    "POST",
                    "https://fixture.invalid/responses",
                    headers={"Authorization": "Bearer SECRET", "x-sdk-retry": str(retry)},
                    json=body,
                )
                response = self._client.send(request)
                if response.status_code == 429 and retry < 2:
                    self.sdk_retries.append(retry)
                    continue
                return response_fixture()
            except Exception:
                if retry == 2:
                    raise
                self.sdk_retries.append(retry)


def execute(baseline, *, observed, native_failures=(), http_statuses=(200,), terminal=None):
    sdk = MockSDK(
        baseline,
        native_failures=native_failures,
        http_statuses=http_statuses,
        terminal=terminal,
    )
    baseline.project.client = sdk
    telemetry = Telemetry()
    context = (
        instrument_foundry(baseline.ask, baseline.project, telemetry=telemetry)
        if observed
        else nullcontext()
    )
    try:
        with context:
            result = baseline.ask.run(
                "same question",
                history=[{"role": "assistant", "content": "same history"}],
            )
        return result, sdk, telemetry
    finally:
        sdk._client.close()


@pytest.mark.parametrize(
    "scenario",
    ["success", "sdk_retry", "native_timeout", "native_429", "combined_retries"],
)
def test_pinned_outgoing_and_public_response_parity(baseline, scenario):
    def options():
        if scenario == "sdk_retry":
            return {"http_statuses": (429, 429, 200)}
        if scenario == "native_timeout":
            return {"native_failures": [baseline.timeout("SECRET"), None]}
        if scenario == "native_429":
            return {"native_failures": [baseline.bad_request("TooManyRequests SECRET"), None]}
        if scenario == "combined_retries":
            return {
                "native_failures": [baseline.timeout("SECRET"), None],
                "http_statuses": (429, 200),
            }
        return {}

    plain, plain_sdk, _ = execute(baseline, observed=False, **options())
    sleeps_plain = list(baseline.sleeps)
    baseline.sleeps.clear()
    observed, observed_sdk, telemetry = execute(baseline, observed=True, **options())
    assert observed == plain
    assert observed["tokens"] == {"in": 40, "out": 20}
    assert observed_sdk.creates == plain_sdk.creates
    assert observed_sdk.responses.create == observed_sdk.create
    assert observed_sdk.requests == plain_sdk.requests
    assert observed_sdk.sdk_retries == plain_sdk.sdk_retries
    assert baseline.sleeps == sleeps_plain
    backoffs = [attrs for name, attrs, _ in telemetry.events if name == "baseline.native_backoff"]
    assert [attrs["wait_seconds"] for attrs in backoffs] == sleeps_plain
    assert observed_sdk.creates[-1] == {
        "input": [
            {"role": "assistant", "content": "same history"},
            {"role": "user", "content": "same question"},
        ],
        "extra_body": {"agent_reference": {"name": "existing-agent", "type": "agent_reference"}},
        "timeout": 170,
    }
    attempts = [attrs for name, attrs, _ in telemetry.events if name == "baseline.sdk_http_attempt"]
    assert len(attempts) == len(plain_sdk.requests)
    assert len({attrs["transport_call_id"] for attrs in attempts}) == len(attempts)
    assert (
        attempts[0]["request_sha256"]
        == hashlib.sha256(
            observed_sdk.requests[0]["body"],
        ).hexdigest()
    )
    assert attempts[0]["provider_ids"] == {"x-request-id": "transport-id"}
    successful = [
        attrs
        for name, attrs, _ in telemetry.events
        if name == "baseline.sdk_create" and "usage" in attrs
    ][0]
    assert successful["usage"] == response_fixture().usage.value
    assert successful["normalized_usage"]["reasoning_tokens"] == 5
    assert successful["normalized_usage"]["cache_write_tokens"] is None
    assert "SECRET" not in json.dumps(telemetry.events)
    assert baseline.ask.run_foundry.__name__ == "run_foundry"


@pytest.mark.parametrize(
    "error_kind,count",
    [("timeout", 2), ("connection", 2), ("bad_request", 1)],
)
def test_native_failure_response_and_attempt_parity(baseline, error_kind, count):
    def options():
        return {"native_failures": [getattr(baseline, error_kind)("private failure")] * count}

    plain, plain_sdk, _ = execute(baseline, observed=False, **options())
    observed, observed_sdk, telemetry = execute(baseline, observed=True, **options())
    assert plain == observed
    assert plain_sdk.creates == observed_sdk.creates
    assert observed["answer"].startswith("ERROR")
    assert len([1 for n, _, _ in telemetry.events if n == "baseline.sdk_create"]) == count
    assert telemetry.events[0][1]["baseline_error_response"] is True
    assert "private failure" not in json.dumps(telemetry.events)


def test_native_429_exhaustion_preserves_all_original_sleeps(baseline):
    def options():
        return {"native_failures": [baseline.bad_request("TooManyRequests")] * 6}

    plain, _, _ = execute(baseline, observed=False, **options())
    original_sleeps = list(baseline.sleeps)
    baseline.sleeps.clear()
    observed, _, telemetry = execute(baseline, observed=True, **options())
    assert plain == observed
    assert baseline.sleeps == original_sleeps == [15, 30, 45, 60, 75]
    assert len([1 for n, _, _ in telemetry.events if n == "baseline.sdk_create"]) == 6
    assert [
        attrs["wait_seconds"]
        for name, attrs, _ in telemetry.events
        if name == "baseline.native_backoff"
    ] == original_sleeps


@pytest.mark.parametrize("boundary", ["enter", "exit"])
def test_native_backoff_export_failure_stops_retries(baseline, boundary):
    sdk = MockSDK(
        baseline,
        native_failures=[baseline.bad_request("TooManyRequests"), None],
    )
    baseline.project.client = sdk
    original_time = baseline.ask.time
    telemetry = Telemetry(fail=("baseline.native_backoff", boundary))
    with instrument_foundry(baseline.ask, baseline.project, telemetry=telemetry):
        with pytest.raises(OSError, match="export unavailable"):
            baseline.ask.run("same question")
    assert len(sdk.creates) == 1
    assert not sdk.requests
    assert baseline.sleeps == ([] if boundary == "enter" else [15])
    assert baseline.ask.time is original_time
    sdk._client.close()


def test_time_proxy_forwards_original_arguments_results_and_errors(baseline):
    calls = []
    original_time = baseline.ask.time
    sentinel = object()

    def sleep(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    original_time.sleep = sleep
    with instrument_foundry(baseline.ask, baseline.project, telemetry=Telemetry()):
        assert baseline.ask.time.time is original_time.time
        assert baseline.ask.time.sleep(1, marker="untouched") is sentinel
    assert calls == [((1,), {"marker": "untouched"})]
    assert baseline.ask.time is original_time


def test_native_sleep_error_propagates_without_new_attempt(baseline):
    error = RuntimeError("private sleeper failure")
    sdk = MockSDK(
        baseline,
        native_failures=[baseline.bad_request("TooManyRequests"), None],
    )
    baseline.project.client = sdk

    def sleep(seconds):
        baseline.sleeps.append(seconds)
        raise error

    baseline.ask.time.sleep = sleep
    telemetry = Telemetry()
    with instrument_foundry(baseline.ask, baseline.project, telemetry=telemetry):
        with pytest.raises(RuntimeError) as caught:
            baseline.ask.run("same question")
    assert caught.value is error
    assert len(sdk.creates) == 1
    assert baseline.sleeps == [15]
    assert "private sleeper failure" not in json.dumps(telemetry.events)
    sdk._client.close()


@pytest.mark.parametrize("error", [RuntimeError("SECRET"), asyncio.CancelledError()])
def test_transport_errors_and_cancellation_propagate_unchanged(baseline, error):
    for observed in (False, True):
        with pytest.raises(type(error)) as caught:
            execute(baseline, observed=observed, terminal=error)
        assert caught.value is error


@pytest.mark.parametrize("boundary", ["enter", "exit"])
def test_export_failure_never_causes_sdk_or_native_retry(baseline, boundary):
    sdk = MockSDK(baseline, http_statuses=(200,) * 20)
    baseline.project.client = sdk
    telemetry = Telemetry(fail=("baseline.sdk_http_attempt", boundary))
    with instrument_foundry(baseline.ask, baseline.project, telemetry=telemetry):
        with pytest.raises(OSError, match="export unavailable"):
            baseline.ask.run("same question")
    assert len(sdk.requests) == (0 if boundary == "enter" else 1)
    assert len(sdk.creates) == 1
    assert not sdk.sdk_retries
    sdk._client.close()


def test_backend_is_rejected_not_switched(baseline, monkeypatch):
    monkeypatch.setenv("AGENT_BACKEND", "responses")
    with pytest.raises(ValueError, match="foundry"):
        with instrument_foundry(baseline.ask, baseline.project, telemetry=Telemetry()):
            pytest.fail("must reject before installing")
    assert baseline.ask.os.getenv("AGENT_BACKEND") == "responses"


def test_missing_usage_is_unknown_and_tool_payloads_are_excluded():
    response = response_fixture()
    response.usage = None
    data = safe_result(response)
    assert data["usage"] is None
    assert all(value is None for value in data["normalized_usage"].values())
    assert data["output"][1]["activity_id"] == "activity-1"
    assert "SECRET" not in json.dumps(data)


def test_baseline_normalizes_reported_cache_writes_and_modality_counters():
    response = response_fixture()
    response.usage.value["input_tokens_details"]["cache_write_tokens"] = 4
    data = safe_result(response)
    assert data["usage"]["input_tokens_details"]["cache_write_tokens"] == 4
    assert data["normalized_usage"]["cache_write_tokens"] == 4
    assert data["normalized_usage"]["cached_input_tokens"] == 10
    assert data["normalized_usage"]["image_tokens"] == 3
    assert data["normalized_usage"]["audio_tokens"] == 2
    assert data["normalized_usage"]["total_tokens"] == 60


def identity_headers(**changes):
    values = {
        "experiment_id": "experiment-1",
        "phase": "pilot",
        "arm": "A",
        "pass": "1",
        "case_id": "q1",
        "logical_request_id": "b" * 32,
        "attempt_id": "1",
        "observation_id": "a" * 32,
    }
    values.update(changes)
    return [(name.encode(), values[field].encode()) for name, field in CORRELATION_HEADERS.items()]


def server_metadata():
    return {"arm": "A", "experiment_id": "experiment-1", "implementation_sha256": "c" * 64}


def test_asgi_exact_join_and_thread_context_without_response_or_body_changes(baseline, tmp_path):
    telemetry = Telemetry()
    sdk = MockSDK(baseline)
    baseline.project.client = sdk
    received = []
    messages = []

    async def app(scope, receive, send):
        received.append(await receive())
        result = await asyncio.to_thread(baseline.ask.run, "same question")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": json.dumps(result).encode()})

    middleware = BaselineObservationMiddleware(
        app,
        tmp_path / "private",
        metadata=server_metadata(),
        telemetry=telemetry,
    )

    async def receive():
        return {"type": "http.request", "body": b"SECRET request"}

    async def send(message):
        messages.append(message)

    with instrument_foundry(baseline.ask, baseline.project, telemetry=telemetry):
        asyncio.run(
            middleware(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/api/chat",
                    "headers": identity_headers(),
                },
                receive,
                send,
            )
        )
    assert received == [{"type": "http.request", "body": b"SECRET request"}]
    assert len(messages) == 2
    assert messages[0] == {
        "type": "http.response.start",
        "status": 200,
        "headers": [
            (b"x-observation-id", b"a" * 32),
            (b"x-experiment-trace-id", b"fixture-trace"),
        ],
    }
    assert telemetry.observations[0][1]["observation_id"] == "a" * 32
    assert telemetry.observations[0][1]["attempt_id"] == "1"
    assert all(meta["logical_request_id"] == "b" * 32 for _, _, meta in telemetry.events)
    assert "SECRET" not in json.dumps(telemetry.events)
    sdk._client.close()


@pytest.mark.parametrize("kind", ["duplicate", "missing", "arm", "path", "experiment"])
def test_invalid_runner_identity_cannot_dispatch_or_choose_destination(tmp_path, kind):
    telemetry = Telemetry()
    dispatched = []

    async def app(*args):
        dispatched.append(True)

    headers = identity_headers()
    if kind == "duplicate":
        headers.append(headers[0])
    elif kind == "missing":
        headers.pop()
    elif kind == "arm":
        headers = identity_headers(arm="F")
    elif kind == "path":
        headers = identity_headers(observation_id="../secret")
    else:
        headers = identity_headers(experiment_id="wrong-experiment")
    middleware = BaselineObservationMiddleware(
        app,
        tmp_path,
        metadata=server_metadata(),
        telemetry=telemetry,
    )
    with pytest.raises(ValueError):
        asyncio.run(
            middleware(
                {"type": "http", "path": "/api/chat", "headers": headers},
                None,
                None,
            )
        )
    assert not dispatched
    assert not telemetry.observations


def test_uninstrumented_browser_requests_pass_through(tmp_path):
    dispatched = []

    async def app(scope, receive, send):
        dispatched.append(scope)

    telemetry = Telemetry()
    middleware = BaselineObservationMiddleware(
        app,
        tmp_path,
        metadata=server_metadata(),
        telemetry=telemetry,
    )
    scope = {"type": "http", "path": "/api/chat", "headers": [(b"cookie", b"SECRET")]}
    asyncio.run(middleware(scope, None, None))
    assert dispatched == [scope]
    assert not telemetry.events


@pytest.mark.skipif(REFERENCE is None or not REFERENCE.is_dir(), reason=REFERENCE_REQUIRED)
def test_copy_reads_only_pinned_blobs_and_records_source_inventory(tmp_path):
    destination = tmp_path / "explicit-new-copy"
    manifest = prepare_copy(REFERENCE, destination)
    assert manifest["source_sha"] == SOURCE_SHA
    assert manifest["source_patches"] == []
    snapshot = dict(manifest)
    snapshot_hash = snapshot.pop("implementation_sha256")
    assert (
        snapshot_hash
        == hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest()
    )
    assert (destination / "LICENSE").is_file()
    assert not (destination / ".git").exists()
    assert not (destination / ".env").exists()
    assert not (destination / ".venv").exists()
    assert not (destination / "infra").exists()
    assert not (destination / "enrich").exists()
    for item in manifest["files"]:
        data = (destination / item["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        expected = subprocess.check_output(
            [
                "git",
                "-C",
                str(REFERENCE),
                "show",
                f"{SOURCE_SHA}:{item['path']}",
            ]
        )
        assert data == expected
        assert (destination / item["path"]).stat().st_mode & 0o077 == 0
    assert destination.stat().st_mode & 0o777 == 0o700
    assert (destination / "BASELINE_COPY_MANIFEST.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        prepare_copy(REFERENCE, destination)
    with pytest.raises(ValueError, match="separate"):
        prepare_copy(REFERENCE, REFERENCE / "not-created-fixture")
    with pytest.raises(ValueError, match="separate"):
        prepare_copy(REFERENCE / "backend", REFERENCE / "not-created-fixture")
    assert not (REFERENCE / "not-created-fixture").exists()


def test_copy_excludes_symlinks_secret_paths_and_nonruntime_blobs(tmp_path, monkeypatch):
    from evaluation import baseline_copy

    tree = (
        b"100644 blob a\tLICENSE\0"
        b"100644 blob b\tbackend/app.py\0"
        b"120000 blob c\tbackend/linked.py\0"
        b"100644 blob d\tbackend/secrets.py\0"
        b"100644 blob e\t.env\0"
        b"100644 blob f\t.venv/config.py\0"
        b"100644 blob g\tinfra/deploy.py\0"
        b"100644 blob h\tcommon/key.pem\0"
    )
    commands = []

    def git(source, *args):
        commands.append(args)
        if args == ("rev-parse", "--show-toplevel"):
            return str(source).encode()
        if args[0] == "rev-parse":
            return SOURCE_SHA.encode()
        if args[0] == "ls-tree":
            return tree
        assert args == ("cat-file", "blob", "a") or args == ("cat-file", "blob", "b")
        return b"fixture source bytes"

    monkeypatch.setattr(baseline_copy, "_git", git)
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "new-copy"
    manifest = prepare_copy(source, destination)
    assert {item["path"] for item in manifest["files"]} == {"LICENSE", "backend/app.py"}
    assert len(manifest["excluded"]) == 6
    assert all(command[0] in {"rev-parse", "ls-tree", "cat-file"} for command in commands)


def test_real_shared_export_preserves_baseline_metadata_and_span_ancestry(baseline, tmp_path):
    from clients import telemetry

    sdk = MockSDK(
        baseline,
        native_failures=[baseline.bad_request("TooManyRequests"), None],
        http_statuses=(429, 200),
    )
    baseline.project.client = sdk
    with telemetry.observation(
        tmp_path / "private", {"arm": "A", "observation_id": "exact-1"}
    ) as rec:
        with instrument_foundry(baseline.ask, baseline.project):
            baseline.ask.run("same question")
    events = [json.loads(line) for line in rec.path.read_text().splitlines()]
    ends = [event for event in events if event["event"] == "span_end"]
    create = next(
        event
        for event in ends
        if event["name"] == "baseline.sdk_create" and "usage" in event["attributes"]
    )
    assert create["attributes"]["usage"] == response_fixture().usage.value
    assert create["attributes"]["output"][1]["activity_id"] == "activity-1"
    assert create["attributes"]["sdk_http_attempts"] == 2
    attempts = [event for event in ends if event["name"] == "baseline.sdk_http_attempt"]
    assert all(event["parent_span_id"] == create["span_id"] for event in attempts)
    assert all(event["observation_id"] == "exact-1" for event in events)
    backoff = next(event for event in ends if event["name"] == "baseline.native_backoff")
    assert backoff["attributes"]["wait_seconds"] == 15
    assert backoff["attributes"]["retry_origin"] == "native"
    assert backoff["attributes"]["retry_index"] == 1
    assert backoff["status"] == "ok"
    assert "SECRET" not in rec.path.read_text()
    assert rec.path.stat().st_mode & 0o777 == 0o600
    sdk._client.close()


def test_health_returns_frozen_copy_identity_without_dispatch(tmp_path):
    telemetry = Telemetry()
    messages = []

    async def app(*args):
        pytest.fail("health must not invoke baseline application or SDK")

    async def send(message):
        messages.append(message)

    middleware = BaselineObservationMiddleware(
        app,
        tmp_path,
        metadata=server_metadata(),
        telemetry=telemetry,
    )
    asyncio.run(middleware({"type": "http", "method": "GET", "path": "/health"}, None, send))
    assert messages[0]["status"] == 200
    assert json.loads(messages[1]["body"]) == server_metadata()
    assert not telemetry.events


def test_launcher_factory_keeps_hooks_for_context_without_azure_probe(baseline, tmp_path):
    original_time = baseline.ask.time
    original_run = baseline.ask.run_foundry
    original_get = baseline.project.get_openai_client
    telemetry = Telemetry()
    messages = []
    initial_invocations = list(baseline.project.invocations)

    async def app(*args):
        pytest.fail("identity health must not claim or probe Azure readiness")

    async def send(message):
        messages.append(message)

    with create_baseline_launcher(
        app,
        baseline.ask,
        baseline.project,
        tmp_path,
        metadata=server_metadata(),
        telemetry=telemetry,
    ) as wrapped:
        assert baseline.ask.run_foundry is not original_run
        assert baseline.ask.time is not original_time
        asyncio.run(wrapped({"type": "http", "method": "GET", "path": "/health"}, None, send))
        assert json.loads(messages[1]["body"]) == server_metadata()
        assert baseline.project.invocations == initial_invocations
    assert baseline.ask.time is original_time
    assert baseline.ask.run_foundry is original_run
    assert baseline.project.get_openai_client is original_get
    assert not telemetry.events


@pytest.mark.parametrize(
    "changes",
    [
        {"observation_id": "not-hex"},
        {"logical_request_id": "not-hex"},
        {"pass": "6"},
        {"case_id": "Q1"},
        {"attempt_id": "3"},
        {"phase": "unapproved"},
    ],
)
def test_runner_field_validation_before_dispatch(tmp_path, changes):
    async def app(*args):
        pytest.fail("invalid runner metadata dispatched")

    middleware = BaselineObservationMiddleware(
        app,
        tmp_path,
        metadata=server_metadata(),
        telemetry=Telemetry(),
    )
    with pytest.raises(ValueError):
        asyncio.run(
            middleware(
                {"type": "http", "path": "/api/chat", "headers": identity_headers(**changes)},
                None,
                None,
            )
        )
