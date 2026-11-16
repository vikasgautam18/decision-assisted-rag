"""Offline, hash-frozen observable model/API usage reconciliation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from evaluation.experiment import FrozenModel, Protocol, identity, schedule
from evaluation.observed_costs import Meter, reconcile_calls
from evaluation.observed_spans import collect
from pipelines.arms import Arm

SHA256 = r"^[a-f0-9]{64}$"
CALL_KINDS = Literal[
    "model.attempt", "baseline.sdk_create", "baseline.sdk_http_attempt", "evaluation.provider"
]


class DeploymentBinding(FrozenModel):
    binding_id: str = Field(min_length=1)
    arms: tuple[Arm, ...] = Field(min_length=1)
    call_kinds: tuple[CALL_KINDS, ...] = Field(min_length=1)
    purpose: Literal["serving", "evaluation"]
    application_sha256_by_arm: dict[Arm, str] = Field(default_factory=dict)
    configuration_hash: str | None = Field(default=None, pattern=SHA256)
    resource_id: str = Field(min_length=1)
    endpoint_host: str = Field(min_length=1)
    deployment: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    reported_models: tuple[str, ...] = Field(min_length=1)
    role: str = Field(min_length=1)
    region: str = Field(min_length=1)
    sku: str = Field(min_length=1)
    context_tiering: bool
    allow_unobserved_deployment: bool = False
    configuration_evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def explicit_mapping(self):
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("duplicate inventory arm")
        if self.purpose == "serving":
            if set(self.application_sha256_by_arm) != set(self.arms):
                raise ValueError("serving mapping requires an exact application hash for every arm")
            for value in self.application_sha256_by_arm.values():
                if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                    raise ValueError("invalid application hash")
        elif self.allow_unobserved_deployment or self.role != "grading":
            raise ValueError("evaluation requires observed deployment and grading role")
        if self.allow_unobserved_deployment and not set(self.call_kinds) <= {
            "baseline.sdk_create",
            "baseline.sdk_http_attempt",
        }:
            raise ValueError("only pinned baseline configuration may supply unobserved deployment")
        return self


class Inventory(FrozenModel):
    schema_version: Literal["1"] = "1"
    status: Literal["draft", "frozen"]
    experiment_id: str
    bindings: tuple[DeploymentBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_bindings(self):
        if len({item.binding_id for item in self.bindings}) != len(self.bindings):
            raise ValueError("duplicate inventory binding identity")
        return self


class RateCard(FrozenModel):
    schema_version: Literal["1"] = "1"
    status: Literal["draft", "frozen"]
    inventory_sha256: str = Field(pattern=SHA256)
    evidence_sha256: dict[str, str]
    meters: tuple[Meter, ...] = Field(min_length=1)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen(
    protocol: Protocol, inventory_path: Path, rate_card_path: Path, evidence_root: Path
) -> tuple[Inventory, RateCard]:
    card_bytes = rate_card_path.read_bytes()
    if hashlib.sha256(card_bytes).hexdigest() != protocol.rate_card_sha256:
        raise ValueError("rate card drifted from protocol")
    card = RateCard.model_validate_json(card_bytes)
    inventory_bytes = inventory_path.read_bytes()
    if hashlib.sha256(inventory_bytes).hexdigest() != card.inventory_sha256:
        raise ValueError("inventory drifted from frozen rate card")
    inventory = Inventory.model_validate_json(inventory_bytes)
    if card.status != "frozen" or inventory.status != "frozen":
        raise ValueError("draft inventory/rate card is not pilot acceptance evidence")
    if inventory.experiment_id != protocol.experiment_id:
        raise ValueError("inventory belongs to a different experiment")
    root = evidence_root.resolve()
    for relative, expected in card.evidence_sha256.items():
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root):
            raise ValueError("evidence paths must remain inside evidence root")
        if file_hash(path) != expected:
            raise ValueError(f"frozen evidence drifted: {relative}")
    for binding in inventory.bindings:
        if binding.configuration_evidence not in card.evidence_sha256:
            raise ValueError("deployment configuration evidence must be hash frozen")
        for arm, expected in binding.application_sha256_by_arm.items():
            if protocol.targets[arm].application_sha256 != expected:
                raise ValueError("inventory application differs from frozen protocol")
        if binding.purpose == "evaluation" and binding.deployment not in {
            item.deployment for item in protocol.evaluator_pool
        }:
            raise ValueError("inventory evaluator is not in frozen protocol pool")
    for meter in card.meters:
        if not meter.resource_id or not meter.model_version:
            raise ValueError("frozen meters require exact resource and model version")
        if not meter.evidence_refs or not set(meter.evidence_refs) <= card.evidence_sha256.keys():
            raise ValueError("meter pricing and semantics evidence must be hash frozen")
        if not meter.component_meter_ids:
            raise ValueError("frozen meters require category-to-meter identity")
        required = {"input_tokens"}
        if meter.output_per_million:
            required.add("output_tokens")
        if meter.cached_per_million:
            required.add("cached_tokens")
        if (
            meter.cache_write_per_million
            and "cache_write_tokens" not in meter.documented_absent_zero
        ):
            required.add("cache_write_tokens")
        if not required <= meter.component_meter_ids.keys():
            raise ValueError("priced categories require exact component meter IDs")
        exact = [
            item
            for item in inventory.bindings
            if all(
                getattr(item, key) == getattr(meter, key)
                for key in (
                    "resource_id",
                    "deployment",
                    "model",
                    "model_version",
                    "role",
                    "region",
                    "sku",
                )
            )
        ]
        if not exact:
            raise ValueError("rate metadata has no exact frozen inventory identity")
        if any(item.context_tiering for item in exact) != (meter.context_tier != "not_applicable"):
            raise ValueError("rate context tier contradicts deployment inventory")
        if (
            meter.input_min_tokens is not None or meter.input_max_tokens is not None
        ) and meter.context_evidence not in meter.evidence_refs:
            raise ValueError("context threshold evidence must be explicitly hash frozen")
    return inventory, card


def hydrate_call(call: dict, inventory: Inventory, protocol: Protocol) -> dict:
    """Supply only configuration-bound identities, never a deployment inferred from model."""
    row = dict(call)
    purpose = "evaluation" if row.get("span_name") == "evaluation.provider" else "serving"
    row["purpose"] = purpose
    observations = {}
    for item in schedule(protocol):
        observations[item.observation_id] = item
        observations[identity(item.logical_request_id, 2)] = item.model_copy(
            update={"attempt": 2, "observation_id": identity(item.logical_request_id, 2)}
        )
    observation = observations.get(row.get("observation_id"))
    if observation is None or row.get("experiment_id") != protocol.experiment_id:
        return {**row, "identity_status": "unpriced_unknown_protocol_observation"}
    expected_phase = "grading" if purpose == "evaluation" else protocol.phase
    if row.get("phase") != expected_phase:
        return {**row, "identity_status": "unpriced_protocol_metadata_mismatch"}
    attempt = row.get("attempt")
    invalid_attempt = attempt is not None and (
        type(attempt) not in (int, str) or str(attempt) != str(observation.attempt)
    )
    if row.get("arm") not in (None, observation.arm) or (purpose == "serving" and invalid_attempt):
        return {**row, "identity_status": "unpriced_protocol_metadata_mismatch"}
    row["arm"] = observation.arm
    candidates = [
        item
        for item in inventory.bindings
        if observation.arm in item.arms
        and purpose == item.purpose
        and row.get("span_name") in item.call_kinds
        and (
            row.get("deployment") == item.deployment
            or (row.get("deployment") is None and item.allow_unobserved_deployment)
        )
        and row.get("role") in (None, item.role)
    ]
    if len(candidates) > 1:
        candidates = [
            item
            for item in candidates
            if row.get("resource_id") in (None, item.resource_id)
            and row.get("endpoint_host") in (None, item.endpoint_host)
            and (
                item.configuration_hash is None
                or row.get("configuration_hash") == item.configuration_hash
            )
        ]
    if len(candidates) != 1:
        return {
            **row,
            "identity_status": (
                "unpriced_ambiguous_inventory" if candidates else "unpriced_missing_exact_inventory"
            ),
        }
    binding = candidates[0]
    supplied = {
        key: getattr(binding, key)
        for key in ("resource_id", "deployment", "model", "model_version", "role", "region", "sku")
    }
    conflicts = [
        key
        for key, value in supplied.items()
        if row.get(key) is not None
        and (row[key] not in binding.reported_models if key == "model" else row[key] != value)
    ]
    if row.get("endpoint_host") not in (None, binding.endpoint_host):
        conflicts.append("endpoint_host")
    if binding.configuration_hash and row.get("configuration_hash") != binding.configuration_hash:
        conflicts.append("configuration_hash")
    if row.get("service_tier") not in (None, "default", "standard"):
        conflicts.append("service_tier")
    if conflicts:
        return {
            **row,
            "identity_status": "unpriced_observed_inventory_mismatch",
            "conflicting_fields": sorted(conflicts),
            "binding_id": binding.binding_id,
        }
    return {
        **row,
        **supplied,
        "observed_model": row.get("model"),
        "binding_id": binding.binding_id,
        "identity_status": "matched",
        "identity_provenance": "explicit_hash_frozen_deployment_arm_configuration",
    }


def build_report(
    protocol_path: Path,
    spans: list[Path],
    inventory_path: Path,
    rate_card_path: Path,
    evidence_root: Path,
) -> dict:
    protocol_bytes = protocol_path.read_bytes()
    protocol = Protocol.model_validate_json(protocol_bytes)
    inventory, card = load_frozen(protocol, inventory_path, rate_card_path, evidence_root)
    exported = collect(spans)
    known_observations = {}
    for item in schedule(protocol):
        known_observations[item.observation_id] = item
        known_observations[identity(item.logical_request_id, 2)] = item
    for trace in exported["traces"]:
        observation = known_observations.get(trace["observation_id"])
        if (
            trace["experiment_id"] != protocol.experiment_id
            or observation is None
            or trace["arm"] not in (None, observation.arm)
            or trace["phase"] not in (protocol.phase, "grading")
        ):
            raise ValueError("span export includes a trace outside the frozen protocol")
    calls = [hydrate_call(call, inventory, protocol) for call in exported["calls"]]
    for call in calls:
        if call.get("context_tier") and (call.get("context_evidence") not in card.evidence_sha256):
            call["identity_status"] = "unpriced_unfrozen_call_context_evidence"
    report = reconcile_calls(calls, list(card.meters), cost_scope=protocol.cost_scope)
    report.update(
        protocol_sha256=protocol.sha256,
        protocol_file_sha256=file_hash(protocol_path),
        rate_card_sha256=file_hash(rate_card_path),
        inventory_sha256=file_hash(inventory_path),
        trace_count=exported["trace_count"],
        incomplete_traces=exported["incomplete_traces"],
        source_exports=exported["source_exports"],
        protocol_serving_observations=len(schedule(protocol)),
        exported_serving_observations=len(
            {
                trace["observation_id"]
                for trace in exported["traces"]
                if trace["phase"] == protocol.phase
            }
        ),
        exported_serving_logical_requests=len(
            {
                known_observations[trace["observation_id"]].logical_request_id
                for trace in exported["traces"]
                if trace["phase"] == protocol.phase
            }
        ),
        limitations=exported["limitations"]
        + [
            "Only exported provider attempts are reconciled; missing hosted children are unknown.",
            "All retries are retained; failed/absent usage is never assumed zero.",
            "Grading is separate from serving; priced subtotals are not complete serving costs.",
            "Non-default/non-standard request processing tiers are not priced by this rate card.",
            "No inference, Azure changes, billing queries, or paid approval occurs in this CLI.",
        ],
    )
    report["observable_calls_fully_priced"] = bool(calls) and not (
        report["unpriced_calls"] or exported["incomplete_traces"]
    )
    report["pilot_acceptance"] = (
        "Requires separate parent verification of scheduled-request/export completeness, "
        "hosted included-model visibility, budgets, and delayed billing; "
        "fully priced observable calls alone are not pilot acceptance."
    )
    if protocol_path.read_bytes() != protocol_bytes:
        raise ValueError("protocol changed during offline reconciliation")
    load_frozen(protocol, inventory_path, rate_card_path, evidence_root)
    for source in exported["source_exports"]:
        if file_hash(Path(source["path"])) != source["sha256"]:
            raise ValueError("span export changed during offline reconciliation")
    return report


def write_private_report(output: Path, report: dict) -> None:
    # Exclusive creation preserves earlier reports and inputs; no overwrite flag.
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("report output must not traverse symlinks")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--spans", type=Path, action="append", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--rate-card", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.protocol, args.spans, args.inventory, args.rate_card, args.evidence_root
    )
    write_private_report(args.output, report)


if __name__ == "__main__":
    main()
