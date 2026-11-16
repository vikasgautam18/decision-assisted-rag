from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

from app.resources import ExperimentResources
from evaluation.preflight import azure_json
from evaluation.run import ROOT, persist

API_VERSION = "2024-08-01"
FIELDS = (
    "date",
    "usageStart",
    "usageEnd",
    "consumedService",
    "meterId",
    "meterName",
    "meterCategory",
    "meterSubCategory",
    "meterRegion",
    "product",
    "productIdentifier",
    "unitOfMeasure",
    "quantity",
    "usageQuantity",
    "billableQuantity",
    "costInBillingCurrency",
    "billingCurrencyCode",
    "costInPricingCurrency",
    "pricingCurrencyCode",
    "costInUSD",
    "effectivePrice",
    "unitPrice",
    "payGPrice",
    "pretaxCost",
    "pretaxCurrency",
    "resourceLocation",
    "resourceLocationNormalized",
    "chargeType",
    "pricingModel",
)


def project_resources(resources: ExperimentResources) -> set[str]:
    prefix = (
        f"/subscriptions/{resources.subscription_id}/resourceGroups/{resources.resource_group}"
        "/providers/"
    )
    ai = urlsplit(resources.ai_endpoint).hostname.split(".")[0]
    search = urlsplit(resources.search_endpoint).hostname.split(".")[0]
    di = urlsplit(resources.di_endpoint).hostname.split(".")[0]
    return {
        (prefix + "Microsoft.CognitiveServices/accounts/" + ai).lower(),
        (prefix + "Microsoft.CognitiveServices/accounts/" + di).lower(),
        (prefix + "Microsoft.Search/searchServices/" + search).lower(),
        (prefix + "Microsoft.Storage/storageAccounts/" + resources.storage_account).lower(),
    }


def usage_row(record: dict, permitted: set[str]) -> dict | None:
    properties = record.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("raw usage detail has no properties object; do not use CLI-flattened rows")
    resource = next(
        (
            value
            for key in ("instanceName", "resourceId", "instanceId")
            if isinstance(value := properties.get(key), str)
            and value.lower().startswith("/subscriptions/")
        ),
        None,
    )
    if not isinstance(resource, str):
        return None  # Subscription-wide rows cannot be attributed to an approved resource.
    normalized = resource.rstrip("/").lower()
    if not any(normalized == target or normalized.startswith(target + "/") for target in permitted):
        return None
    result = {
        "record_id_sha256": hashlib.sha256(record["id"].encode()).hexdigest(),
        "kind": record.get("kind"),
        "resource_id": resource,
        **{key: properties[key] for key in FIELDS if key in properties},
    }
    modern = record.get("kind") == "modern"
    cost = properties.get("costInBillingCurrency" if modern else "pretaxCost")
    quantity = properties.get("quantity" if modern else "usageQuantity")
    currency = properties.get("billingCurrencyCode" if modern else "pretaxCurrency")
    result["reported_cost"] = str(Decimal(str(cost))) if cost is not None else None
    result["reported_quantity"] = str(Decimal(str(quantity))) if quantity is not None else None
    result["reported_currency"] = currency
    result["usage_status"] = (
        "reported_billing_aggregate"
        if cost is not None and quantity is not None
        else "partial_billing_aggregate"
    )
    additional = properties.get("additionalInfo")
    if isinstance(additional, str) and additional.strip():
        try:
            additional = json.loads(additional)
        except json.JSONDecodeError as error:
            raise ValueError(
                "billing additionalInfo is not valid JSON; capture incomplete"
            ) from error
    if isinstance(additional, dict):
        allowed = {
            "model",
            "modelname",
            "modelversion",
            "deployment",
            "deploymentname",
            "sku",
            "skuname",
            "cachettl",
            "promptcacheretention",
            "apitype",
            "tokenstype",
        }
        result["additional_metadata"] = {
            key: value
            for key, value in additional.items()
            if key.lower().replace("_", "").replace("-", "") in allowed
            and isinstance(value, (str, int, float, bool))
        }
    return result


def validate_page_url(url: str, subscription_id: str) -> str:
    parsed = urlsplit(url)
    expected = f"/subscriptions/{subscription_id}/providers/Microsoft.Consumption/usageDetails"
    if (
        parsed.scheme != "https"
        or parsed.hostname != "management.azure.com"
        or parsed.username
        or parsed.password
        or parsed.fragment
        or parsed.port is not None
        or parsed.path.lower() != expected.lower()
    ):
        raise ValueError("usage pagination URL is outside the configured subscription API")
    return url


def collect(resources: ExperimentResources, *, max_pages: int = 5, read=azure_json) -> dict:
    if not 1 <= max_pages <= 20:
        raise ValueError("bounded billing page limit must be between 1 and 20")
    url = (
        f"https://management.azure.com/subscriptions/{resources.subscription_id}"
        f"/providers/Microsoft.Consumption/usageDetails?api-version={API_VERSION}&$top=1000"
    )
    permitted = project_resources(resources)
    rows, seen_pages, seen_rows = [], set(), set()
    fingerprints_by_id = {}
    repeated_exact_rows = 0
    for _ in range(max_pages):
        validate_page_url(url, resources.subscription_id)
        if url in seen_pages:
            raise ValueError("billing pagination loop")
        seen_pages.add(url)
        page = read(["rest", "--method", "get", "--url", url])
        if not isinstance(page, dict) or not isinstance(page.get("value"), list):
            raise ValueError("billing API returned an invalid page")
        for raw in page["value"]:
            row = usage_row(raw, permitted)
            if row is not None:
                fingerprint = hashlib.sha256(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if fingerprint in seen_rows:
                    repeated_exact_rows += 1
                    continue
                seen_rows.add(fingerprint)
                fingerprints_by_id.setdefault(row["record_id_sha256"], set()).add(fingerprint)
                row["record_fingerprint_sha256"] = fingerprint
                rows.append(row)
        url = page.get("nextLink")
        if not url:
            break
    by_currency = {}
    for row in rows:
        if row["reported_cost"] is not None and row["reported_currency"] is not None:
            currency = row["reported_currency"]
            by_currency[currency] = str(
                Decimal(by_currency.get(currency, "0")) + Decimal(row["reported_cost"])
            )
    return {
        "status": "bounded_resource_filtered_billing_evidence_not_request_attribution",
        "source": "Microsoft.Consumption usageDetails raw ARM API",
        "api_version": API_VERSION,
        "captured_utc": datetime.now(UTC).isoformat(),
        "pages_read": len(seen_pages),
        "pagination_complete": not bool(url),
        "additional_pages_unread": bool(url),
        "permitted_resource_ids": sorted(permitted),
        "record_count": len(rows),
        "repeated_exact_rows_excluded_from_subtotal": repeated_exact_rows,
        "non_unique_record_ids": sum(len(values) > 1 for values in fingerprints_by_id.values()),
        "latest_usage_date": max(
            (row.get("date") or row.get("usageEnd") or "" for row in rows), default=None
        ),
        "reported_cost_by_currency_for_returned_records_only": by_currency,
        "records": rows,
        "limitations": [
            "Modern fields are costInBillingCurrency and quantity, "
            "not legacy pretaxCost/usageQuantity.",
            "Server resource-group filters may be ignored; "
            "only explicitly permitted resources retained.",
            "Unrelated subscription records are discarded in memory and never persisted.",
            "Billing may lag; missing recent usage is not zero cost.",
            "Resource/day/meter aggregates include shared traffic; "
            "not per-request cost attribution.",
            "These records do not authorize any paid pilot, provisioning or cap increase.",
            "The Usage Details API is deprecated; use Cost Details/exports for durable "
            "invoice reconciliation. This collector is bounded preflight evidence.",
            "Repeated identical rows are reported and excluded from observed subtotals; "
            "non-unique billing identifiers need reconciliation before an invoice-total claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read scoped raw billing evidence; no inference.")
    parser.add_argument("--resources", type=Path, default=ROOT / "manifests/azure-resources.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--read-azure", action="store_true")
    args = parser.parse_args()
    if not args.read_azure:
        parser.error("--read-azure required; no Azure call made")
    if args.output.exists():
        parser.error("output already exists; preserve billing evidence")
    resources = ExperimentResources.model_validate_json(args.resources.read_bytes())
    report = collect(resources, max_pages=args.max_pages)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    persist(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
