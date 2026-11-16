"""Offline-only import/factory checks; never starts a listener or calls an agent."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import site
import socket
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def _blocked(*args, **kwargs):
    raise AssertionError("offline preflight forbids network, subprocesses and SDK dispatch")


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


async def _exercise(application, snapshot: str, experiment_id: str) -> list[str]:
    import backend.agent.ask as ask
    from azure.ai.projects import AIProjectClient

    incoming = asyncio.Queue()
    outgoing = asyncio.Queue()
    original = AIProjectClient.get_openai_client
    with patch.object(AIProjectClient, "__init__", _blocked), patch.object(ask, "run", _blocked):
        lifespan = asyncio.create_task(
            application(
                {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}},
                incoming.get,
                outgoing.put,
            )
        )
        try:
            await incoming.put({"type": "lifespan.startup"})
            started = await asyncio.wait_for(outgoing.get(), timeout=10)
            assert started["type"] == "lifespan.startup.complete", started["type"]
            assert AIProjectClient.get_openai_client is not original
            messages = []

            async def send(message):
                messages.append(message)

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.0"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/health",
                "raw_path": b"/health",
                "query_string": b"",
                "root_path": "",
                "headers": [],
                "server": ("offline.invalid", 80),
                "client": ("127.0.0.1", 1),
            }
            await application(scope, receive, send)
            assert messages[0]["status"] == 200
            assert json.loads(messages[1]["body"]) == {
                "arm": "A",
                "experiment_id": experiment_id,
                "implementation_sha256": snapshot,
            }
            scope.update(
                method="POST",
                path="/api/chat",
                raw_path=b"/api/chat",
                headers=[(b"x-experiment-id", experiment_id.encode())],
            )
            try:
                await application(scope, receive, send)
            except ValueError as exc:
                assert "incomplete" in str(exc)
            else:
                raise AssertionError("partial correlation identity was accepted")
            await incoming.put({"type": "lifespan.shutdown"})
            stopped = await asyncio.wait_for(outgoing.get(), timeout=10)
            assert stopped["type"] == "lifespan.shutdown.complete", stopped["type"]
            await asyncio.wait_for(lifespan, timeout=10)
        finally:
            if not lifespan.done():
                lifespan.cancel()
                try:
                    await lifespan
                except asyncio.CancelledError:
                    pass
        assert AIProjectClient.get_openai_client is original
    return [
        "actual_asgi_lifespan_installs_hooks",
        "local_health_exact_snapshot",
        "partial_chat_identity_rejected_before_agent",
        "actual_asgi_shutdown_restores_hooks",
    ]


def run(copy: Path, manifest_path: Path, lock: Path, settings_path: Path | None = None) -> dict:
    """Run in a fresh dedicated process; environment changes are process-local."""
    sys.dont_write_bytecode = True
    platform.platform()  # Cache local uname metadata before blocking all subprocesses.
    manifest = json.loads(manifest_path.read_bytes())
    expected = {
        _canonical(item["name"]): item["version"]
        for item in manifest["original_public_distributions"] + manifest["observer_additions"]
    }
    installed = {}
    for distribution in importlib.metadata.distributions(path=site.getsitepackages()):
        name = _canonical(distribution.metadata["Name"])
        metadata = distribution.read_text("METADATA") or ""
        item = {
            "name": name,
            "version": distribution.version,
            "metadata_sha256": hashlib.sha256(metadata.encode()).hexdigest(),
            "record_sha256": hashlib.sha256(
                (distribution.read_text("RECORD") or "").encode()
            ).hexdigest(),
        }
        assert name not in installed or installed[name] == item, "ambiguous installed package"
        assert (
            Path(distribution.locate_file("")).resolve().is_relative_to(Path(sys.prefix).resolve())
        ), "distribution escaped isolated runtime"
        installed[name] = item
    assert {name: item["version"] for name, item in installed.items()} == expected
    for item in manifest["original_public_distributions"]:
        assert installed[_canonical(item["name"])]["metadata_sha256"] == item["metadata_sha256"]

    copy = copy.resolve(strict=True)
    approved_settings = json.loads(settings_path.read_bytes()) if settings_path else {}
    allowed_settings = {
        "SUBSCRIPTION_ID",
        "RG",
        "SUFFIX",
        "SEARCH_ENDPOINT",
        "AI_ENDPOINT",
        "STORAGE_ACCOUNT",
        "STORAGE_NAME",
        "PROJECT_ENDPOINT",
        "PROJECT_ACCOUNT",
        "PROJECT_NAME",
        "CHAT_DEPLOYMENT",
        "SOURCE_MODE",
        "DOCS_CONTAINER",
        "FIGURES_CONTAINER",
        "EMBED_DEPLOYMENT",
        "EMBED_DIMENSIONS",
        "KB_NAME",
        "AGENT_NAME",
        "SEARCH_API_VERSION",
        "AGENT_BACKEND",
    }
    assert set(approved_settings) <= allowed_settings, "unexpected explicit setting"
    assert all(isinstance(value, str) and value for value in approved_settings.values())
    os.environ.clear()
    os.environ.update(
        {
            "SUBSCRIPTION_ID": "00000000-0000-0000-0000-000000000000",
            "SUFFIX": "offlineonly",
            "SEARCH_ENDPOINT": "https://offline.invalid",
            "AI_ENDPOINT": "https://offline.invalid",
            "STORAGE_ACCOUNT": "offlineonly",
            "PROJECT_ENDPOINT": "https://offline.invalid/api/projects/offline",
            "KB_NAME": "offline-only-kb",
            "CHAT_DEPLOYMENT": "offline-only-deployment",
            "AGENT_BACKEND": "foundry",
            "DECISION_RAG_BASELINE_COPY": str(copy),
            "DECISION_RAG_PROTOCOL": "/offline-only-in-memory-protocol.json",
            "DECISION_RAG_APPROVAL": "/offline-only-in-memory-approval.json",
        }
    )
    os.environ.update(approved_settings)
    checks = ["exact_isolated_distribution_pins_and_metadata"]
    with ExitStack() as stack:
        for target in (
            "socket.socket.connect",
            "socket.socket.connect_ex",
            "socket.socket.bind",
            "socket.create_connection",
            "socket.getaddrinfo",
            "subprocess.Popen",
            "os.system",
        ):
            stack.enter_context(patch(target, _blocked))
        for operation, args in (
            (socket.create_connection, (("offline.invalid", 443),)),
            (subprocess.Popen, (["az", "account", "show"],)),
        ):
            try:
                operation(*args)
            except AssertionError:
                pass
            else:
                raise AssertionError("offline guard failed")
        checks.append("network_listener_and_subprocess_guards_active")

        from evaluation.baseline_serve import create_baseline_app, validate_copy
        from evaluation.experiment import Approval, Protocol

        snapshot = validate_copy(copy)["implementation_sha256"]
        checks.append("copy_and_frozen_hook_manifest_valid")
        protocol = Protocol.model_validate_json(
            json.dumps(
                {
                    "experiment_id": "offline-only-baseline-preflight",
                    "phase": "pilot",
                    "seed": 0,
                    "passes": 1,
                    "case_ids": ["q1"],
                    "targets": {
                        arm: {
                            "origin": f"http://127.0.0.1:{19000 + index}",
                            "application_sha256": snapshot if arm == "A" else "0" * 64,
                            "timeout_seconds": 1.0,
                            "placement": "offline-fixture-not-a-serving-target",
                        }
                        for index, arm in enumerate("ABCDEFG")
                    },
                    **{
                        name: "0" * 64
                        for name in (
                            "code_sha256",
                            "dataset_sha256",
                            "rate_card_sha256",
                            "review_overlay_sha256",
                        )
                    },
                }
            )
        )
        # Schema-shaped fixture exists only in memory and cannot authorize a runner.
        approval = Approval.model_validate_json(
            json.dumps(
                {
                    "protocol_sha256": protocol.sha256,
                    "approval_reference": "OFFLINE FIXTURE ONLY - NOT USER AUTHORIZATION",
                    "user_approved_paid_execution": True,
                    "max_application_attempts": 1,
                    "max_grading_attempts": 0,
                    "max_reserved_usd": "0.01",
                    "reservation_usd_per_application": "0.01",
                    "reservation_usd_per_grade": "0.01",
                    "cost_ceiling_kind": "conservative_reservation_not_invoice_guarantee",
                }
            )
        )
        fixtures = {
            Path(os.environ["DECISION_RAG_PROTOCOL"]): protocol.model_dump_json().encode(),
            Path(os.environ["DECISION_RAG_APPROVAL"]): approval.model_dump_json().encode(),
        }
        read_bytes = Path.read_bytes

        def fixture_bytes(path):
            return fixtures[path] if path in fixtures else read_bytes(path)

        stack.enter_context(patch.object(Path, "read_bytes", fixture_bytes))
        fixtures[Path(os.environ["DECISION_RAG_APPROVAL"])] = (
            approval.model_copy(update={"protocol_sha256": "f" * 64}).model_dump_json().encode()
        )
        try:
            create_baseline_app()
        except ValueError as exc:
            assert "approval differs" in str(exc)
        else:
            raise AssertionError("mismatched fixture approval accepted")
        checks.append("mismatched_protocol_approval_rejected")
        fixtures[Path(os.environ["DECISION_RAG_APPROVAL"])] = approval.model_dump_json().encode()
        with patch.dict(os.environ, {"PROJECT_ENDPOINT": ""}):
            try:
                create_baseline_app()
            except ValueError as exc:
                assert "incomplete" in str(exc)
            else:
                raise AssertionError("incomplete explicit settings accepted")
        checks.append("missing_explicit_resource_setting_rejected")
        application = create_baseline_app()
        import backend
        import backend.agent.ask as ask
        import common.config as config
        import httpx
        import httpx2
        import openai
        import uvicorn

        assert Path(backend.__path__[0]).resolve() == copy / "backend"
        assert Path(ask.__file__).resolve().is_relative_to(copy)
        assert Path(config.__file__).resolve().is_relative_to(copy)
        assert not config.ENV_PATH.exists()
        assert openai.DefaultHttpxClient.__mro__[1].__module__.startswith("httpx2")
        assert httpx.Client is not httpx2.Client
        assert callable(uvicorn.run)
        checks.append("factory_and_uvicorn_import_native_sdk_uses_httpx2")
        checks.extend(asyncio.run(_exercise(application, snapshot, protocol.experiment_id)))
        assert validate_copy(copy)["implementation_sha256"] == snapshot
        checks.append("runtime_copy_unchanged_after_preflight")
    return {
        "schema_version": 1,
        "status": "offline_ready",
        "paid_execution_authorized_by_this_evidence": False,
        "network_or_agent_calls": 0,
        "service_started": False,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "runtime_prefix": sys.prefix,
        "copy_directory": str(copy),
        "implementation_sha256": snapshot,
        "dependency_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "requirements_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "preflight_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings_sha256": (
            hashlib.sha256(settings_path.read_bytes()).hexdigest() if settings_path else None
        ),
        "checks": checks,
        "installed_distributions": sorted(installed.values(), key=lambda item: item["name"]),
        "remaining_gates": [
            "Parent supplies authentic protocol/approval and confirms explicit resource settings",
            "Parent alone owns Azure provisioning, paid preflight and service launch",
            "Local health proves copy identity only, not hosted deployment readiness",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy", type=Path, required=True)
    parser.add_argument("--dependency-manifest", type=Path, required=True)
    parser.add_argument("--requirements-lock", type=Path, required=True)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite preflight evidence")
    result = run(args.copy, args.dependency_manifest, args.requirements_lock, args.settings)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"PASS: {len(result['checks'])} offline checks; evidence: {args.output}")


if __name__ == "__main__":
    main()
