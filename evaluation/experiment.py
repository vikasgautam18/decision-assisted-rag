from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import uuid
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluation.comparison import validate_origin
from pipelines.arms import ARMS, Arm


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Target(FrozenModel):
    origin: str
    application_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    timeout_seconds: float = Field(gt=0, le=1800)
    placement: str = Field(min_length=1, max_length=100)

    @field_validator("origin")
    @classmethod
    def origin_only(cls, value: str) -> str:
        return validate_origin(value)


class Evaluator(FrozenModel):
    deployment: str = Field(pattern=r"^[A-Za-z0-9._-]{1,100}$")
    tokens_per_minute: int = Field(ge=1)
    requests_per_minute: int = Field(ge=1)


class CostScope(FrozenModel):
    excluded_categories: tuple[Literal["search_retrieval", "defender_ai"], ...] = ()
    exclusion_reference: str | None = None

    @model_validator(mode="after")
    def explicit_exclusions(self):
        if len(set(self.excluded_categories)) != len(self.excluded_categories):
            raise ValueError("duplicate cost exclusion")
        if self.excluded_categories and not (
            self.exclusion_reference and self.exclusion_reference.strip()
        ):
            raise ValueError("cost exclusions require an explicit user decision reference")
        return self

    def report(self) -> dict:
        return {
            "label": "Model/API usage cost, not total serving cost",
            "excluded_categories": list(self.excluded_categories),
            "exclusion_reference": self.exclusion_reference,
            "applies_to_arms": list(ARMS),
            "excluded_charges_are_zero": False,
            "total_serving_cost_claim": False,
            "limitations": "Other missing included-model usage remains unpriced; "
            "exclusions do not waive telemetry, pricing or paid-approval gates.",
        }


class Protocol(FrozenModel):
    schema_version: Literal["1"] = "1"
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    phase: Literal["pilot", "main", "load"]
    seed: int
    passes: int = Field(ge=1, le=5)
    case_ids: tuple[str, ...]
    targets: dict[Arm, Target]
    code_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rate_card_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    review_overlay_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extra_cases_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    baseline_visibility_accepted: bool = False
    cost_scope: CostScope = CostScope()
    concurrency_levels: tuple[int, ...] = (1, 2, 4)
    evaluator_pool: tuple[Evaluator, ...] = (
        Evaluator(
            deployment="decision-rag-evaluator",
            tokens_per_minute=30_000,
            requests_per_minute=100,
        ),
    )

    @model_validator(mode="after")
    def valid_protocol(self):
        if set(self.targets) != set(ARMS):
            raise ValueError("experiment requires exactly arms A-G")
        if len({target.origin for target in self.targets.values()}) != len(ARMS):
            raise ValueError("each arm requires a distinct immutable application origin")
        if not self.evaluator_pool or len({item.deployment for item in self.evaluator_pool}) != len(
            self.evaluator_pool
        ):
            raise ValueError("evaluator replicas must be nonempty and unique")
        if not self.case_ids or len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("case IDs must be nonempty and unique")
        if any(not key.startswith("q") or not key[1:].isdigit() for key in self.case_ids):
            raise ValueError("invalid experiment case ID")
        if self.phase == "main" and (len(self.case_ids) != 72 or self.passes != 5):
            raise ValueError("main protocol requires 72 cases and five passes")
        if self.phase == "load" and (
            len(self.case_ids) != 12 or self.passes != 1 or self.concurrency_levels != (1, 2, 4)
        ):
            raise ValueError("load protocol requires 12 cases at concurrency 1, 2 and 4")
        return self

    @property
    def sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


class Observation(FrozenModel):
    experiment_id: str
    phase: Literal["pilot", "main", "load"]
    observation_id: str
    logical_request_id: str
    arm: Arm
    pass_number: int
    case_id: str
    attempt: int = Field(ge=1, le=2)
    ordinal: int
    concurrency: int = 1

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-Experiment-ID": self.experiment_id,
            "X-Observation-ID": self.observation_id,
            "X-Logical-Request-ID": self.logical_request_id,
            "X-Experiment-Arm": self.arm,
            "X-Experiment-Pass": str(self.pass_number),
            "X-Experiment-Case": self.case_id,
            "X-Experiment-Attempt": str(self.attempt),
            "X-Experiment-Phase": self.phase,
        }


def digest(value: dict | list) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def identity(*parts: object) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(parts, separators=(",", ":"))).hex


def schedule(protocol: Protocol) -> list[Observation]:
    rng = random.Random(protocol.seed)
    if protocol.phase == "load":
        rows = []
        for level in protocol.concurrency_levels:
            order = rng.sample(list(ARMS), len(ARMS))
            cases = rng.sample(list(protocol.case_ids), len(protocol.case_ids))
            for arm in order:
                for case_id in cases:
                    logical = identity(
                        protocol.experiment_id, protocol.phase, level, 1, case_id, arm
                    )
                    rows.append(
                        Observation(
                            experiment_id=protocol.experiment_id,
                            phase="load",
                            observation_id=identity(logical, 1),
                            logical_request_id=logical,
                            arm=arm,
                            pass_number=1,
                            case_id=case_id,
                            attempt=1,
                            ordinal=len(rows),
                            concurrency=level,
                        )
                    )
        return rows
    base = list(ARMS)
    rng.shuffle(base)
    rows = []
    levels = protocol.concurrency_levels if protocol.phase == "load" else (1,)
    blocks = 0
    for level in levels:
        for pass_number in range(1, protocol.passes + 1):
            cases = rng.sample(list(protocol.case_ids), len(protocol.case_ids))
            for case_id in cases:
                # Rotations and reversals spread positions and immediate predecessors.
                shift = blocks % len(base)
                order = base[shift:] + base[:shift]
                if blocks // len(base) % 2:
                    order.reverse()
                blocks += 1
                for arm in order:
                    logical = identity(
                        protocol.experiment_id, protocol.phase, level, pass_number, case_id, arm
                    )
                    rows.append(
                        Observation(
                            experiment_id=protocol.experiment_id,
                            phase=protocol.phase,
                            observation_id=identity(logical, 1),
                            logical_request_id=logical,
                            arm=arm,
                            pass_number=pass_number,
                            case_id=case_id,
                            attempt=1,
                            ordinal=len(rows),
                            concurrency=level,
                        )
                    )
    return rows


def schedule_audit(rows: list[Observation]) -> dict:
    counts = Counter(row.arm for row in rows)
    positions = {arm: Counter() for arm in ARMS}
    predecessors = {arm: Counter() for arm in ARMS}
    for offset in range(0, len(rows), len(ARMS)):
        block = rows[offset : offset + len(ARMS)]
        if block[0].phase == "load":
            continue
        for position, row in enumerate(block):
            positions[row.arm][position] += 1
            if position:
                predecessors[row.arm][block[position - 1].arm] += 1
    return {
        "observations": len(rows),
        "unique_observation_ids": len({row.observation_id for row in rows}),
        "by_arm": dict(counts),
        "arm_positions": {arm: dict(values) for arm, values in positions.items()},
        "within_block_predecessors": {arm: dict(values) for arm, values in predecessors.items()},
        "limitations": "Finite schedule; position/predecessor imbalance is reported, "
        "not assumed zero.",
    }


class Approval(FrozenModel):
    protocol_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    approval_reference: str = Field(min_length=1)
    user_approved_paid_execution: Literal[True]
    max_application_attempts: int = Field(ge=1)
    max_grading_attempts: int = Field(ge=0)
    max_reserved_usd: Decimal = Field(gt=0, allow_inf_nan=False)
    reservation_usd_per_application: Decimal = Field(gt=0, allow_inf_nan=False)
    reservation_usd_per_grade: Decimal = Field(gt=0, allow_inf_nan=False)
    cost_ceiling_kind: Literal["conservative_reservation_not_invoice_guarantee"]
    runtime_cumulative_limits: dict[str, int] = Field(
        default_factory=lambda: {
            "di_analysis": 6,
            "decision": 4800,
            "model": 1200,
            "grading": 600,
        }
    )

    @model_validator(mode="after")
    def cumulative_limits(self):
        if set(self.runtime_cumulative_limits) != {
            "di_analysis",
            "decision",
            "model",
            "grading",
        } or any(
            type(value) is not int or value < 0 for value in self.runtime_cumulative_limits.values()
        ):
            raise ValueError("explicit cumulative quota categories required")
        if self.runtime_cumulative_limits["di_analysis"] != 6:
            raise ValueError("rerun does not authorize new ingestion")
        return self


class ExperimentLedger:
    """Durable pre-dispatch intent; uncertain attempts are never auto-resubmitted."""

    def __init__(self, path: Path, protocol: Protocol, approval: Approval):
        if approval.protocol_sha256 != protocol.sha256:
            raise ValueError("paid approval does not match the frozen protocol")
        if not protocol.baseline_visibility_accepted:
            raise ValueError("baseline visibility must be explicitly accepted before spending")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not path.exists():
            path.touch(mode=0o600, exist_ok=False)
        path.chmod(0o600)
        self.path, self.protocol, self.approval = path, protocol, approval
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, "
                "kind TEXT NOT NULL, state TEXT NOT NULL, reserved_usd TEXT NOT NULL, "
                "started_utc TEXT NOT NULL, completed_utc TEXT, record TEXT)"
            )
            for key, value in (
                ("protocol_sha256", protocol.sha256),
                ("approval_sha256", digest(approval.model_dump(mode="json"))),
            ):
                existing = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
                if existing and existing[0] != value:
                    raise ValueError("existing ledger has different protocol or approval")
                db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", (key, value))

    def reserve(self, attempt_id: str, kind: Literal["application", "grading"]) -> bool:
        if kind not in ("application", "grading"):
            raise ValueError("unknown spending category")
        amount = (
            self.approval.reservation_usd_per_application
            if kind == "application"
            else self.approval.reservation_usd_per_grade
        )
        limit = (
            self.approval.max_application_attempts
            if kind == "application"
            else self.approval.max_grading_attempts
        )
        with sqlite3.connect(self.path, isolation_level=None) as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT kind FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if prior:
                if prior[0] != kind:
                    raise ValueError("attempt identity reused across spending roles")
                return False
            count = db.execute("SELECT COUNT(*) FROM attempts WHERE kind=?", (kind,)).fetchone()[0]
            reserved = sum(
                (Decimal(row[0]) for row in db.execute("SELECT reserved_usd FROM attempts")),
                Decimal(0),
            )
            if count >= limit or reserved + amount > self.approval.max_reserved_usd:
                raise ValueError("approved attempt or reserved-spend ceiling exhausted")
            db.execute(
                "INSERT INTO attempts VALUES (?,?, 'reserved', ?, ?,NULL,NULL)",
                (attempt_id, kind, str(amount), datetime.now(UTC).isoformat()),
            )
            db.execute("COMMIT")
        return True

    def complete(self, attempt_id: str, record: dict) -> None:
        with sqlite3.connect(self.path) as db:
            result = db.execute(
                "UPDATE attempts SET state='recorded',completed_utc=?,record=? "
                "WHERE id=? AND state='reserved'",
                (datetime.now(UTC).isoformat(), json.dumps(record, allow_nan=False), attempt_id),
            )
            if result.rowcount != 1:
                raise ValueError("attempt missing or already recorded")

    def records(self) -> list[dict]:
        with sqlite3.connect(self.path) as db:
            return [
                {
                    "id": row[0],
                    "kind": row[1],
                    "state": row[2],
                    "reserved_usd": row[3],
                    "started_utc": row[4],
                    "completed_utc": row[5],
                    "record": json.loads(row[6]) if row[6] else None,
                }
                for row in db.execute("SELECT * FROM attempts ORDER BY rowid")
            ]
