"""Create private DRAFT pilot cost inputs from existing local evidence, without Azure calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.observed_cost_report import file_hash, write_private_report

PREFLIGHT = "data/experiments/preflight-20261104"
RESOLUTION = f"{PREFLIGHT}/billing-resolution.json"
MODERN = f"{PREFLIGHT}/modern-billing-evidence.json"
FINDINGS = f"{PREFLIGHT}/pricing/findings.json"
RETAIL = "data/comparison/phase3-full-comparison-retry1/retail-price-snapshot.json"
ACTIVE_AGENT = f"{PREFLIGHT}/baseline-active-agent.json"


def create_templates(
    root: Path, directory: Path, experiment_id: str, *, baseline_endpoint_host: str | None = None
) -> None:
    directory = directory.resolve()
    root = root.resolve()
    if not directory.is_relative_to(root):
        raise ValueError("private draft directory must be inside project evidence root")
    resolution = json.loads((root / RESOLUTION).read_bytes())["resolved"]
    resources = json.loads((root / "manifests/azure-resources.json").read_bytes())
    retail = {row["meterId"]: row for row in json.loads((root / RETAIL).read_bytes())["items"]}
    configuration_path = directory / "configuration-evidence-template.json"
    configuration_ref = str(configuration_path.relative_to(root))
    inventory_path = directory / "inventory-template.json"
    card_path = directory / "rate-card-template.json"
    for path in (configuration_path, inventory_path, card_path):
        if path.exists():
            raise FileExistsError("draft files already exist; do not overwrite frozen inputs")
    evidence = {
        "status": "UNVERIFIED_DRAFT_NOT_PILOT_APPROVAL",
        "required_parent_actions": [
            "Replace experiment_id and every zero application hash with approved frozen values.",
            "Verify and snapshot active baseline agent deployment/version/resource configuration; "
            "never infer it from response.model. Add its exact binding and meters only then.",
            "Snapshot exact deployment ARM identities/versions/SKUs, including evaluator replicas.",
            "Resolve GPT5.6 context-tier condition using authoritative per-call evidence or a "
            "documented numeric input-token threshold. No threshold is guessed in this template.",
            "Freeze configuration/pricing/semantics evidence hashes, inventory hash, rate-card "
            "file hash in Protocol.rate_card_sha256, and set both draft statuses to frozen.",
        ],
        "billing_evidence": [RESOLUTION, MODERN],
        "model_versions": {
            "gpt-5.6-sol": "2026-07-09",
            "text-embedding-3-large": "1",
            "Microsoft-Decision-1": "1",
            "gpt-5.2": "2025-12-11",
        },
        "replicas": {
            "decision-rag-evaluator": "existing; parent must verify exact configuration",
            "decision-rag-evaluator-2": "parent reports GPT5.2/2025-12-11 GlobalStandard 60",
            "decision-rag-evaluator-3": "parent reports GPT5.2/2025-12-11 GlobalStandard 60",
        },
        "scope": "Model/API usage cost, not total serving cost. Search retrieval and Defender "
        "scanned tokens excluded uniformly A-G by the frozen protocol, not zero or disabled.",
        "limitations": [
            "Baseline A requires an explicit active-agent snapshot and SDK endpoint host.",
            "GPT5.6 meters intentionally lack numeric context bounds; ambiguous tier is unpriced.",
            "No fabricated usage counters, public-meter inference, or retail embedding proxy.",
            "GPT5.2 cache writes have no additional write charge; any reported writes remain "
            "ordinary input. Only absent cache-write category is documented as zero.",
        ],
    }
    write_private_report(configuration_path, evidence)
    endpoint = resources["ai_endpoint"]
    host = endpoint.split("://", 1)[1].rstrip("/")
    bindings = []
    meters = []

    def binding(role, deployment, model, version, arms, *, tiered=False, evaluation=False):
        item = {
            "binding_id": f"{deployment}-{role}",
            "arms": list(arms),
            "call_kinds": ["evaluation.provider" if evaluation else "model.attempt"],
            "purpose": "evaluation" if evaluation else "serving",
            "application_sha256_by_arm": {} if evaluation else dict.fromkeys(arms, "0" * 64),
            "resource_id": endpoint,
            "endpoint_host": host,
            "deployment": deployment,
            "model": model,
            "model_version": version,
            "reported_models": (
                [model, f"{model}-{version}"] if len(version) == 10
                else [model, model.lower()] if model == "Microsoft-Decision-1"
                else [model]
            ),
            "role": role,
            "region": "westus3",
            "sku": "Standard" if role == "embedding" else "GlobalStandard",
            "context_tiering": tiered,
            "configuration_evidence": configuration_ref,
        }
        bindings.append(item)
        return item

    def meter(item, rates, ids, *, tier="not_applicable", cache=False, absent=(), kind=None):
        meters.append(
            {
                **{
                    key: item[key]
                    for key in (
                        "model",
                        "model_version",
                        "resource_id",
                        "deployment",
                        "role",
                        "region",
                        "sku",
                    )
                },
                "source": RESOLUTION if tier != "long" else RETAIL,
                "effective_date": "2026-11-04",
                "meter_ids": list(dict.fromkeys(ids.values())),
                "component_meter_ids": ids,
                "component_price_kinds": dict.fromkeys(ids, kind or "subscription_effective"),
                "price_kind": kind or "subscription_effective",
                "input_per_million": rates[0],
                "cached_per_million": rates[1],
                "cache_write_per_million": rates[2],
                "output_per_million": rates[3],
                "cache_semantics": "disjoint_input_subsets" if cache else "not_applicable",
                "documented_absent_zero": list(absent),
                "semantics_evidence": FINDINGS,
                "evidence_refs": [RESOLUTION, MODERN, FINDINGS, RETAIL],
                "context_tier": tier,
            }
        )

    short = resolution["writer_short_context_global_standard"]
    writer_ids = {
        counter: short[category]["meter_id"]
        for counter, category in (
            ("input_tokens", "input"),
            ("cached_tokens", "cached_input"),
            ("cache_write_tokens", "cache_write"),
            ("output_tokens", "output"),
        )
    }
    long_ids = {
        "input_tokens": "09760484-7c47-5f5a-acd7-53d7cfbff733",
        "cached_tokens": "fab49251-fa53-5aaa-84af-72eff9d3d934",
        "cache_write_tokens": "3f8397e0-567f-5dd9-ba93-2f0b15336c76",
        "output_tokens": "f3f9f3cd-eb4b-56c0-a2f6-222656d84b21",
    }
    for role, arms in (("writer", "BCDEFG"), ("semantic_decision", "G")):
        item = binding(
            role, resources["writer_deployment"], "gpt-5.6-sol", "2026-07-09", arms, tiered=True
        )
        short = resolution["writer_short_context_global_standard"]
        meter(
            item,
            [
                short[key]["effective_price_usd"]
                for key in ("input", "cached_input", "cache_write", "output")
            ],
            writer_ids,
            tier="short",
            cache=True,
        )
        meter(
            item,
            [
                str(retail[long_ids[key]]["retailPrice"])
                for key in ("input_tokens", "cached_tokens", "cache_write_tokens", "output_tokens")
            ],
            long_ids,
            tier="long",
            cache=True,
            kind="official_retail",
        )
    if baseline_endpoint_host is not None:
        active = json.loads((root / ACTIVE_AGENT).read_bytes())
        if active.get("status") != "read_only_active_agent_matches_pinned_source" or not active.get(
            "instructions_match_pinned_source"
        ):
            raise ValueError("baseline identity requires parent's verified active-agent snapshot")
        if active.get("model_deployment") != resources["writer_deployment"]:
            raise ValueError(
                "baseline model differs; supply separately verified model/rate metadata"
            )
        item = binding(
            "writer", active["model_deployment"], "gpt-5.6-sol", "2026-07-09", "A", tiered=True
        )
        item.update(
            binding_id="baseline-active-agent-writer",
            call_kinds=["baseline.sdk_create", "baseline.sdk_http_attempt"],
            endpoint_host=baseline_endpoint_host,
            allow_unobserved_deployment=True,
            configuration_evidence=ACTIVE_AGENT,
        )
    embedding = binding(
        "embedding", resources["embedding_deployment"], "text-embedding-3-large", "1", "BCDEFG"
    )
    meter(
        embedding,
        [resolution["embedding"]["equivalent_per_million_usd"], None, None, "0"],
        {"input_tokens": resolution["embedding"]["meter_id"]},
        absent=("output_tokens", "cached_tokens", "cache_write_tokens"),
    )
    decision = binding(
        "decision", resources["decision_deployment"], "Microsoft-Decision-1", "1", "CDEF"
    )
    meter(
        decision,
        [resolution["decision"]["effective_price_usd"], None, None, "0"],
        {"input_tokens": resolution["decision"]["meter_id"]},
        absent=("output_tokens", "cached_tokens", "cache_write_tokens"),
    )
    for deployment in (
        "decision-rag-evaluator",
        "decision-rag-evaluator-2",
        "decision-rag-evaluator-3",
    ):
        item = binding("grading", deployment, "gpt-5.2", "2025-12-11", "ABCDEFG", evaluation=True)
        evaluator = resolution["evaluator_global_standard"]
        cached_id = "d875aea4-23b4-54b3-9313-a93fe2595a7b"
        ids = {
            "input_tokens": evaluator["input"]["meter_id"],
            "cached_tokens": cached_id,
            "output_tokens": evaluator["output"]["meter_id"],
        }
        meter(
            item,
            [
                evaluator["input"]["effective_price_usd"],
                str(retail[cached_id]["retailPrice"]),
                evaluator["input"]["effective_price_usd"],
                evaluator["output"]["effective_price_usd"],
            ],
            ids,
            cache=True,
            absent=("cache_write_tokens",),
        )
        meters[-1]["component_price_kinds"]["cached_tokens"] = "official_retail"
    write_private_report(
        inventory_path,
        {
            "schema_version": "1",
            "status": "draft",
            "experiment_id": experiment_id,
            "bindings": bindings,
        },
    )
    sources = [configuration_ref, RESOLUTION, MODERN, FINDINGS, RETAIL]
    if baseline_endpoint_host is not None:
        sources.append(ACTIVE_AGENT)
    write_private_report(
        card_path,
        {
            "schema_version": "1",
            "status": "draft",
            "inventory_sha256": file_hash(inventory_path),
            "evidence_sha256": {path: file_hash(root / path) for path in sources},
            "meters": meters,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--baseline-endpoint-host")
    args = parser.parse_args()
    create_templates(
        args.evidence_root,
        args.output_directory,
        args.experiment_id,
        baseline_endpoint_host=args.baseline_endpoint_host,
    )


if __name__ == "__main__":
    main()
