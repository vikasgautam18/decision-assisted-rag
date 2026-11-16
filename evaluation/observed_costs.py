from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from evaluation.experiment import CostScope, FrozenModel

COUNTERS = ("input_tokens", "output_tokens", "cached_tokens", "cache_write_tokens")


class Meter(FrozenModel):
    model: str
    model_version: str | None = None
    resource_id: str | None = None
    deployment: str
    role: str
    region: str
    sku: str
    currency: Literal["USD"] = "USD"
    source: str = Field(min_length=1)
    effective_date: str = Field(min_length=1)
    meter_ids: tuple[str, ...] = Field(min_length=1)
    price_kind: Literal["subscription_effective", "official_retail"]
    input_per_million: Decimal = Field(ge=0, allow_inf_nan=False)
    output_per_million: Decimal = Field(ge=0, allow_inf_nan=False)
    cached_per_million: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    cache_write_per_million: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    cache_semantics: Literal["disjoint_input_subsets", "not_applicable"]
    documented_absent_zero: tuple[str, ...] = ()
    semantics_evidence: str = Field(min_length=1)
    context_tier: Literal["not_applicable", "short", "long"] = "not_applicable"
    input_min_tokens: int | None = Field(default=None, ge=0, strict=True)
    input_max_tokens: int | None = Field(default=None, ge=0, strict=True)
    context_evidence: str | None = None
    component_meter_ids: dict[str, str] = Field(default_factory=dict)
    component_price_kinds: dict[str, Literal["subscription_effective", "official_retail"]] = Field(
        default_factory=dict
    )
    evidence_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid_categories(self):
        if not set(self.documented_absent_zero) <= set(COUNTERS):
            raise ValueError("undocumented usage category")
        if "input_tokens" in self.documented_absent_zero:
            raise ValueError("included input usage cannot be synthesized as zero")
        if not set(self.component_meter_ids.values()) <= set(self.meter_ids):
            raise ValueError("component meter identity is absent from meter_ids")
        if not set(self.component_meter_ids) <= set(COUNTERS):
            raise ValueError("unknown component meter category")
        if not set(self.component_price_kinds) <= set(self.component_meter_ids):
            raise ValueError("component price kind requires a matching meter identity")
        if self.cache_semantics == "disjoint_input_subsets" and (
            self.cached_per_million is None or self.cache_write_per_million is None
        ):
            raise ValueError("cache-aware pricing requires verified cache rates")
        bounds = (self.input_min_tokens, self.input_max_tokens)
        if any(value is not None for value in bounds):
            if self.context_tier == "not_applicable" or not self.context_evidence:
                raise ValueError("context bounds require a tier and documented input-token rule")
            if all(value is not None for value in bounds) and bounds[0] > bounds[1]:
                raise ValueError("context input bounds are reversed")
        return self


def counter(value: object, key: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer or unknown")
    return value


def normalize_usage(usage: dict | None) -> dict[str, int | None]:
    if usage is None:
        return dict.fromkeys(COUNTERS)
    if not isinstance(usage, dict):
        raise ValueError("usage must be an object or unknown")
    primary = usage.get("input_tokens", usage.get("prompt_tokens"))
    secondary = usage.get("output_tokens", usage.get("completion_tokens"))
    details = usage.get("input_tokens_details", usage.get("prompt_tokens_details"))
    if details is not None and not isinstance(details, dict):
        raise ValueError("usage details must be an object")
    details = details or {}
    return {
        "input_tokens": counter(primary, "input_tokens"),
        "output_tokens": counter(secondary, "output_tokens"),
        "cached_tokens": counter(details.get("cached_tokens"), "cached_tokens"),
        "cache_write_tokens": counter(
            details.get("cache_write_tokens", usage.get("cache_creation_input_tokens")),
            "cache_write_tokens",
        ),
    }


def price_usage(usage: dict | None, meter: Meter) -> dict:
    counts = normalize_usage(usage)
    provider_counts = counts.copy()
    absent_zero_applied = []
    for key in meter.documented_absent_zero:
        if usage is not None and counts[key] is None:
            counts[key] = 0
            absent_zero_applied.append(key)
    required = list(COUNTERS[:2])
    if meter.cache_semantics == "disjoint_input_subsets":
        required.extend(COUNTERS[2:])
    missing = [key for key in required if counts[key] is None]
    if missing:
        return {
            "status": "unpriced_unknown_usage",
            "cost_usd": None,
            "missing_counters": missing,
            "normalized_usage": counts,
            "provider_normalized_usage": provider_counts,
            "documented_absent_zero_applied": absent_zero_applied,
        }
    ordinary = counts["input_tokens"]
    output = counts["output_tokens"]
    cached = counts["cached_tokens"] or 0
    written = counts["cache_write_tokens"] or 0
    if meter.cache_semantics == "not_applicable" and (cached or written):
        raise ValueError("cache usage contradicts the selected non-cache meter")
    if cached + written > ordinary:
        raise ValueError("disjoint cache subsets exceed reported input tokens")
    cost = Decimal(ordinary - cached - written) * meter.input_per_million
    cost += Decimal(output) * meter.output_per_million
    if meter.cache_semantics == "disjoint_input_subsets":
        cost += Decimal(cached) * meter.cached_per_million
        cost += Decimal(written) * meter.cache_write_per_million
    return {
        "status": "priced_reported_usage_estimate_not_invoice",
        "cost_usd": str(cost / Decimal(1_000_000)),
        "normalized_usage": counts,
        "provider_normalized_usage": provider_counts,
        "documented_absent_zero_applied": absent_zero_applied,
        "price_kind": meter.price_kind,
        "component_price_kinds": meter.component_price_kinds,
        "meter_ids": list(meter.meter_ids),
        "source": meter.source,
    }


def exact_meters(call: dict, meters: list[Meter]) -> list[Meter]:
    return [
        meter
        for meter in meters
        if all(
            call.get(key) == getattr(meter, key)
            for key in ("model", "deployment", "role", "region", "sku")
        )
        and all(
            getattr(meter, key) is None or call.get(key) == getattr(meter, key)
            for key in ("resource_id", "model_version")
        )
    ]


def select_meter(call: dict, meters: list[Meter]) -> tuple[Meter | None, str]:
    matches = exact_meters(call, meters)
    if not matches:
        return None, "unpriced_missing_exact_meter"
    selected = []
    counts = normalize_usage(call.get("usage"))
    for meter in matches:
        if meter.context_tier == "not_applicable":
            selected.append(meter)
            continue
        explicit = call.get("context_tier")
        if explicit is not None:
            if explicit != meter.context_tier or not call.get("context_evidence"):
                continue
        has_bounds = meter.input_min_tokens is not None or meter.input_max_tokens is not None
        if has_bounds:
            tokens = counts["input_tokens"]
            if tokens is None:
                continue
            if meter.input_min_tokens is not None and tokens < meter.input_min_tokens:
                continue
            if meter.input_max_tokens is not None and tokens > meter.input_max_tokens:
                continue
        elif explicit is None:
            continue
        selected.append(meter)
    if len(selected) > 1:
        raise ValueError("ambiguous rate-card match")
    return (selected[0], "") if selected else (None, "unpriced_unknown_or_conflicting_context_tier")


def context_rate_bounds(call: dict, meters: list[Meter]) -> dict | None:
    """A rate envelope is not a verified context tier or an exact call cost."""
    matches = exact_meters(call, meters)
    if {meter.context_tier for meter in matches} != {"short", "long"}:
        return None
    alternatives = [price_usage(call.get("usage"), meter) for meter in matches]
    if any(item["cost_usd"] is None for item in alternatives):
        return None
    amounts = [Decimal(item["cost_usd"]) for item in alternatives]
    return {
        "status": "rate_envelope_only_context_tier_not_verified",
        "lower_usd": str(min(amounts)),
        "upper_usd": str(max(amounts)),
        "included_in_priced_subtotal": False,
        "candidate_meter_ids": [list(meter.meter_ids) for meter in matches],
    }


def subtotal(rows: list[dict]) -> dict:
    unknown = sum(row["cost_usd"] is None for row in rows)
    amount = sum(
        (Decimal(row["cost_usd"]) for row in rows if row["cost_usd"] is not None), Decimal(0)
    )
    return {
        "observable_calls": len(rows),
        "priced_calls": len(rows) - unknown,
        "unpriced_calls": unknown,
        "priced_subtotal_usd": str(amount),
        "complete_observable_usage_total_usd": str(amount) if rows and not unknown else None,
        "total_serving_cost_claim": False,
    }


def reconcile_calls(
    calls: list[dict], meters: list[Meter], *, cost_scope: CostScope | None = None
) -> dict:
    seen = set()
    rows = []
    for call in calls:
        call_id = call["call_id"]
        if call_id in seen:
            raise ValueError("duplicate billable call identity; reconcile before reporting")
        seen.add(call_id)
        if call.get("identity_status", "matched") == "matched":
            meter, reason = select_meter(call, meters)
        else:
            meter, reason = None, call["identity_status"]
        priced = (
            price_usage(call.get("usage"), meter) if meter else {"status": reason, "cost_usd": None}
        )
        if reason == "unpriced_unknown_or_conflicting_context_tier":
            priced["context_tier_cost_bounds_usd"] = context_rate_bounds(call, meters)
        rows.append(
            {
                **call,
                "reported_usage_status": (
                    "absent_unknown" if call.get("usage") is None else "service_reported"
                ),
                "normalized_usage": normalize_usage(call.get("usage")),
                **priced,
                "meter": meter.model_dump(mode="json") if meter else None,
            }
        )
    serving = [row for row in rows if row.get("purpose", "serving") == "serving"]
    evaluation = [row for row in rows if row.get("purpose") == "evaluation"]
    groups = {}
    for key in ("arm", "model", "role"):
        groups[f"by_{key}"] = {
            value: {
                purpose: subtotal(
                    [
                        row
                        for row in rows
                        if str(row.get(key) or "unknown") == value
                        and row.get("purpose", "serving") == purpose
                    ]
                )
                for purpose in ("serving", "evaluation")
            }
            for value in sorted({str(row.get(key) or "unknown") for row in rows})
        }
    grouped = {}
    for row in rows:
        key = tuple(str(row.get(field) or "unknown") for field in ("arm", "model", "role"))
        grouped.setdefault(key, []).append(row)
    return {
        "status": "usage_based_estimate_not_invoice_or_total_cost",
        "cost_scope": (cost_scope or CostScope()).report(),
        **subtotal(rows),
        "serving": subtotal(serving),
        "evaluation": subtotal(evaluation),
        **groups,
        "by_arm_model_role": [
            {
                **dict(zip(("arm", "model", "role"), key, strict=True)),
                **{
                    purpose: subtotal(
                        [row for row in values if row.get("purpose", "serving") == purpose]
                    )
                    for purpose in ("serving", "evaluation")
                },
            }
            for key, values in sorted(grouped.items())
        ],
        "calls": rows,
        "invoice_reconciliation": "pending_separate_resource_meter_billing_evidence",
        "shared_capacity": "not_allocated_to_requests",
    }
