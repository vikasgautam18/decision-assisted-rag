"""Explicitly approved implementation walkthrough, not phase 2 benchmark evaluation."""

import argparse
import json
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--execute-approved-checkpoint", action="store_true", required=True)
    parser.add_argument(
        "--start-at",
        choices=["comparison", "scope-switch-calculation", "history", "browser"],
        default="comparison",
    )
    args = parser.parse_args()
    reports = []
    output = Path(__file__).resolve().parents[1] / "data/review/phase-one-walkthrough.json"

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(reports, indent=2) + "\n")

    with httpx.Client(base_url=args.url, headers={"Origin": args.url}, timeout=190) as client:
        client.get("/health").raise_for_status()
        conversation_id = None
        cases = [
            (
                "comparison",
                "Compare the declared combined range of NOVA-3 Dedicated CNG and VEGA-7 "
                "1.2 L Petrol MT. State the applicable variant qualifications.",
                {"NOVA-3", "VEGA-7"},
            ),
            (
                "scope-switch-calculation",
                "Now use only VOLTA-4. What is the difference in usable battery capacity "
                "between Long Range RWD and Standard Range RWD? Show the subtraction.",
                {"VOLTA-4"},
            ),
            (
                "history",
                "NOVA-3 Dedicated CNG: what range was published at Revision C, "
                "and what is the current Revision D value? Label both revisions.",
                {"NOVA-3"},
            ),
        ]
        for name, question, documents in cases:
            order = ["comparison", "scope-switch-calculation", "history", "browser"]
            if order.index(name) < order.index(args.start_at):
                continue
            started = time.monotonic()
            response = client.post(
                "/api/chat", json={"question": question, "conversation_id": conversation_id}
            )
            result = response.json()
            reports.append(
                {
                    "case": name,
                    "seconds": time.monotonic() - started,
                    "http_status": response.status_code,
                    "result": result,
                }
            )
            save()
            print(name, response.status_code, json.dumps(result), flush=True)
            assert response.status_code == 200 and result["behavior"] == "answer"
            assert result["semantic_verification"] == "passed"
            assert {item["document"] for item in result["citations"]} == documents
            if name == "comparison":
                assert "310" in result["answer"] and "792" in result["answer"]
            if name == "scope-switch-calculation":
                assert result["calculations"]
                assert result["calculations"][0]["result"] in ("14", "14.0", "14.00")
            if name == "history":
                assert "326" in result["answer"] and "310" in result["answer"]
                assert any(item["page_start"] == 13 for item in result["citations"])
            conversation_id = result["conversation_id"]
        assert client.get(f"/api/conversations/{conversation_id}").status_code == 200
        with httpx.Client(base_url=args.url, timeout=10) as other:
            assert other.get(f"/api/conversations/{conversation_id}").status_code == 404

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(args.url, wait_until="networkidle")
        page.locator("#question").fill(
            "Using the original NOVA-3 warranty claim flowchart, who approves a non-wear-part "
            "claim for INR 20,000 at 1 year and 15,000 km, and what is the settlement timeline? "
            "Show the original figure."
        )
        started = time.monotonic()
        with page.expect_response("**/api/chat", timeout=190_000) as pending:
            page.get_by_role("button", name="Ask the live pipeline").click()
        response = pending.value
        result = response.json()
        reports.append(
            {
                "case": "real-browser-original-figure",
                "seconds": time.monotonic() - started,
                "http_status": response.status,
                "result": result,
            }
        )
        save()
        print("real-browser-original-figure", response.status, json.dumps(result), flush=True)
        assert result["semantic_verification"] == "passed"
        assert "A-33" in result["answer"] and "15 working days" in result["answer"]
        assert any(item["key"] == "NOVA-3-figure-2" for item in result["figures"])
        page.wait_for_function(
            "Array.from(document.querySelectorAll('article img')).some("
            "img => img.src.includes('NOVA-3-figure-2') && img.naturalWidth > 0)"
        )
        assert page.locator("article a").count() > 0
        browser.close()
    print("Approved implementation walkthrough passed; not an evaluation result.")


if __name__ == "__main__":
    main()
