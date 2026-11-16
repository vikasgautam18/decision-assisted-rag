from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from app.resources import ExperimentResources
from evaluation.run import ROOT, persist


def azure_json(arguments: list[str]) -> object:
    result = subprocess.run(
        ["az", *arguments, "--output", "json", "--only-show-errors"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def replica_capacity(usages: list[dict], requirements: dict[str, int]) -> list[dict]:
    result = []
    for name, requested in requirements.items():
        matches = [item for item in usages if item["name"] == name]
        if len(matches) != 1:
            result.append({"name": name, "status": "missing_or_ambiguous_quota"})
            continue
        usage = matches[0]
        available = usage["limit"] - usage["used"]
        result.append(
            {
                "name": name,
                "status": "capacity_available" if available >= requested else "insufficient",
                "available_units": available,
                "requested_units": requested,
                "unit": usage["unit"],
                "note": "Allocation units are not a measured throughput guarantee.",
            }
        )
    return result


def candidate_inventory(candidates: list[dict], deployments: list[dict]) -> list[dict]:
    result = []
    for candidate in candidates:
        existing = [item for item in deployments if item["name"] == candidate["name"]]
        if len(existing) > 1:
            raise ValueError("ambiguous existing replica deployment")
        status = "not_created"
        if existing:
            status = (
                "existing_matches_proposal"
                if all(
                    existing[0].get(key) == candidate.get(key)
                    for key in ("model", "version", "sku", "capacity", "upgrade")
                )
                and existing[0].get("state") == "Succeeded"
                else "existing_configuration_mismatch"
            )
        result.append({**candidate, "deployment_status": status})
    return result


def inspect(resources: ExperimentResources) -> dict:
    account = resources.ai_endpoint.split("//", 1)[1].split(".", 1)[0]
    common = [
        "--resource-group",
        resources.resource_group,
        "--name",
        account,
        "--subscription",
        resources.subscription_id,
    ]
    account_data = azure_json(
        [
            "cognitiveservices",
            "account",
            "show",
            *common,
            "--query",
            "{name:name,location:location,kind:kind,sku:sku.name}",
        ]
    )
    deployments = azure_json(
        [
            "cognitiveservices",
            "account",
            "deployment",
            "list",
            *common,
            "--query",
            "[].{name:name,model:properties.model.name,version:properties.model.version,"
            "format:properties.model.format,sku:sku.name,capacity:sku.capacity,"
            "upgrade:properties.versionUpgradeOption,state:properties.provisioningState}",
        ]
    )
    usages = azure_json(
        [
            "cognitiveservices",
            "usage",
            "list",
            "--location",
            account_data["location"],
            "--subscription",
            resources.subscription_id,
            "--query",
            "[].{name:name.value,used:currentValue,limit:limit,unit:unit}",
        ]
    )
    models = azure_json(
        [
            "cognitiveservices",
            "account",
            "list-models",
            *common,
            "--query",
            "[?name=='gpt-5.2' || name=='gpt-5.6-sol' || name=='Microsoft-Decision-1']."
            "{name:name,version:version,format:format,skus:skus[].name,lifecycle:lifecycleStatus}",
        ]
    )
    candidates = [
        {
            "name": "decision-rag-evaluator-2",
            "model": "gpt-5.2",
            "version": "2025-12-11",
            "capacity": 60,
        },
        {
            "name": "decision-rag-evaluator-3",
            "model": "gpt-5.2",
            "version": "2025-12-11",
            "capacity": 60,
        },
        {
            "name": "decision-rag-writer-2",
            "model": "gpt-5.6-sol",
            "version": "2026-07-09",
            "capacity": 100,
        },
        {
            "name": "decision-rag-decision-2",
            "model": "Microsoft-Decision-1",
            "version": "1",
            "capacity": 100,
        },
    ]
    for candidate in candidates:
        candidate["sku"] = "GlobalStandard"
        candidate["upgrade"] = "NoAutoUpgrade"
        candidate["exact_version_listed"] = any(
            item["name"] == candidate["model"]
            and item["version"] == candidate["version"]
            and candidate["sku"] in (item["skus"] or [])
            for item in models
        )
    candidates = candidate_inventory(candidates, deployments)
    remaining = [row for row in candidates if row["deployment_status"] == "not_created"]
    return {
        "status": "read_only_preflight_not_provisioning_approval",
        "captured_utc": datetime.now(UTC).isoformat(),
        "account": account_data,
        "deployments": deployments,
        "candidate_replicas_not_created": remaining,
        "existing_candidate_replicas": [
            row for row in candidates if row["deployment_status"] != "not_created"
        ],
        "capacity_checks": replica_capacity(
            usages,
            {
                "OpenAI.GlobalStandard.gpt-5.2": sum(
                    row["capacity"] for row in remaining if row["model"] == "gpt-5.2"
                ),
                "OpenAI.GlobalStandard.gpt-5.6-sol": sum(
                    row["capacity"] for row in remaining if row["model"] == "gpt-5.6-sol"
                ),
                "AIServices.GlobalStandard.Microsoft-Decision-1": sum(
                    row["capacity"] for row in remaining if row["model"] == "Microsoft-Decision-1"
                ),
            },
        ),
        "drift_risk": [item["name"] for item in deployments if item["upgrade"] != "NoAutoUpgrade"],
        "unresolved": [
            "This read-only command neither provisions nor authorizes resources; "
            "existing deployments may include separately approved replicas.",
            "Effective rate card and token/cache meter semantics need verification.",
            "Baseline hosted nested spans/usage visibility needs an approved live pilot.",
            "Subscription billing export access and shared-traffic attribution unverified.",
            "Auto-upgrading baseline deployments cannot be changed; "
            "version drift needs monitoring.",
            "Pilot/main/retry/load call and spend caps require explicit new approval.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit read-only Azure replica preflight.")
    parser.add_argument("--resources", type=Path, default=ROOT / "manifests/azure-resources.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-azure", action="store_true")
    args = parser.parse_args()
    if not args.inspect_azure:
        parser.error("--inspect-azure is required; no Azure operations occur by default")
    if args.output.exists():
        parser.error("output exists; preserve earlier preflight evidence")
    resources = ExperimentResources.model_validate_json(args.resources.read_bytes())
    report = inspect(resources)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    persist(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
