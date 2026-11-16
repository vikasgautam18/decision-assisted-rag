import pytest

from app.resources import ExperimentResources
from evaluation.billing_evidence import collect, project_resources, usage_row, validate_page_url
from evaluation.run import ROOT


def test_modern_billing_fields_are_not_lost_by_legacy_cli_projection():
    record = {
        "id": "private-billing-record-identity",
        "kind": "modern",
        "properties": {
            "instanceName": "/subscriptions/test/project/ai",
            "costInBillingCurrency": 0.13,
            "quantity": 1,
            "billingCurrencyCode": "USD",
            "pretaxCost": None,
            "usageQuantity": None,
            "customerName": "must-not-be-persisted",
            "additionalInfo": '{"model":"test-model","authorization":"do-not-copy"}',
        },
    }
    row = usage_row(record, {"/subscriptions/test/project/ai"})
    assert row["reported_cost"] == "0.13"
    assert row["reported_quantity"] == "1"
    assert row["reported_currency"] == "USD"
    assert row["additional_metadata"] == {"model": "test-model"}
    assert "customerName" not in row
    assert "private-billing-record-identity" not in str(row)
    assert usage_row(record, {"/subscriptions/unrelated/project/ai"}) is None


def test_billing_next_link_cannot_change_host_subscription_or_scheme():
    url = (
        "https://management.azure.com/subscriptions/test/providers/"
        "Microsoft.Consumption/usageDetails?next=1"
    )
    assert validate_page_url(url, "test") == url
    for bad in (
        url.replace("test/", "foreign/"),
        url.replace("https:", "http:"),
        url.replace("management.azure.com", "untrusted.invalid"),
    ):
        with pytest.raises(ValueError):
            validate_page_url(bad, "test")


def resources():
    return ExperimentResources.model_validate_json(
        (ROOT / "manifests/azure-resources.json").read_bytes()
    )


def record(resource, *, identity="row", quantity=1, cost=0.13, date="2026-11-03"):
    return {
        "id": identity,
        "kind": "modern",
        "properties": {
            "instanceName": resource,
            "date": date,
            "quantity": quantity,
            "costInBillingCurrency": cost,
            "billingCurrencyCode": "USD",
        },
    }


def test_paginated_capture_filters_resources_even_when_server_filter_is_ignored():
    config = resources()
    permitted = sorted(project_resources(config))[0]
    next_url = (
        f"https://management.azure.com/subscriptions/{config.subscription_id}/providers/"
        "Microsoft.Consumption/usageDetails?next=1"
    )
    pages = [
        {
            "value": [
                record(permitted.upper() + "/deployments/test", identity="first"),
                record(permitted + "-unrelated", identity="foreign"),
            ],
            "nextLink": next_url,
        },
        {"value": [record(permitted, identity="second", date="2026-11-04")]},
    ]
    calls = []

    def read(arguments):
        calls.append(arguments)
        return pages[len(calls) - 1]

    report = collect(config, read=read)
    assert len(calls) == 2
    assert calls[1][-1] == next_url
    assert report["record_count"] == 2
    assert report["pagination_complete"]
    assert report["latest_usage_date"] == "2026-11-04"
    assert report["reported_cost_by_currency_for_returned_records_only"] == {"USD": "0.26"}
    assert all("-unrelated" not in row["resource_id"] for row in report["records"])
    assert report["status"].endswith("not_request_attribution")


def test_exact_repeated_rows_are_distinguished_from_nonunique_provider_identities():
    config = resources()
    permitted = sorted(project_resources(config))[0]
    first = record(permitted, identity="nonunique")
    second = record(permitted, identity="nonunique", quantity=2, cost=0.26)
    report = collect(config, read=lambda _: {"value": [first, first, second]})
    assert report["record_count"] == 2
    assert report["repeated_exact_rows_excluded_from_subtotal"] == 1
    assert report["non_unique_record_ids"] == 1
    assert report["reported_cost_by_currency_for_returned_records_only"] == {"USD": "0.39"}
    assert len({row["record_fingerprint_sha256"] for row in report["records"]}) == 2


def test_page_limit_and_billing_lag_never_imply_complete_or_zero_recent_cost():
    config = resources()
    permitted = sorted(project_resources(config))[0]
    next_url = (
        f"https://management.azure.com/subscriptions/{config.subscription_id}/providers/"
        "Microsoft.Consumption/usageDetails?next=1"
    )
    report = collect(
        config,
        max_pages=1,
        read=lambda _: {"value": [record(permitted)], "nextLink": next_url},
    )
    assert not report["pagination_complete"]
    assert report["additional_pages_unread"]
    assert report["latest_usage_date"] == "2026-11-03"
    assert any("missing recent usage is not zero" in note for note in report["limitations"])
    empty = collect(config, read=lambda _: {"value": []})
    assert empty["latest_usage_date"] is None
    assert empty["reported_cost_by_currency_for_returned_records_only"] == {}


def test_legacy_and_partial_rows_preserve_missing_values_and_real_resource_identity():
    legacy = {
        "id": "legacy-row",
        "kind": "legacy",
        "properties": {
            "instanceName": "display-name-is-not-an-arm-resource-id",
            "instanceId": "/subscriptions/test/project/ai",
            "pretaxCost": 0.26,
            "usageQuantity": 2,
            "pretaxCurrency": "USD",
        },
    }
    permitted = {"/subscriptions/test/project/ai"}
    row = usage_row(legacy, permitted)
    assert row["reported_cost"] == "0.26"
    assert row["reported_quantity"] == "2"
    assert row["reported_currency"] == "USD"
    partial = record("/subscriptions/test/project/ai", cost=None, quantity=None)
    row = usage_row(partial, permitted)
    assert row["usage_status"] == "partial_billing_aggregate"
    assert row["reported_cost"] is None
    assert row["reported_quantity"] is None


def test_malformed_billing_metadata_and_pagination_fail_explicitly():
    config = resources()
    permitted = sorted(project_resources(config))[0]
    malformed = record(permitted)
    malformed["properties"]["additionalInfo"] = "{not valid JSON"
    with pytest.raises(ValueError, match="additionalInfo is not valid JSON"):
        usage_row(malformed, {permitted})
    url = (
        f"https://management.azure.com/subscriptions/{config.subscription_id}/providers/"
        "Microsoft.Consumption/usageDetails?api-version=2024-08-01&$top=1000"
    )
    with pytest.raises(ValueError, match="pagination loop"):
        collect(config, read=lambda _: {"value": [], "nextLink": url})
    with pytest.raises(ValueError, match="invalid page"):
        collect(config, read=lambda _: {"value": {}})
    with pytest.raises(ValueError, match="between 1 and 20"):
        collect(config, max_pages=0, read=lambda _: pytest.fail("must not query Azure"))
