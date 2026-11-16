"""UI checks against a running pilot; intercepted chat calls do not spend Azure quota."""

import argparse
import json

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="networkidle")
        page.get_by_role("heading", name="Decision-assisted RAG", exact=True).wait_for()
        assert page.locator(".notice, [aria-label='Application status']").count() == 0
        fixture = {
            "status": "completed",
            "behavior": "answer",
            "semantic_verification": "passed",
            "trace_id": "browser-fixture-not-a-live-model-result",
            "conversation_id": "a" * 32,
            "answer": "UI fixture only: <img src=x onerror=alert(1)>",
            "citations": [
                {
                    "source_id": "NOVA-3-table-6",
                    "document": "NOVA-3",
                    "revision": "D",
                    "page_start": 4,
                    "page_end": 4,
                }
            ],
            "figures": [{"key": "NOVA-3-figure-1"}],
        }
        page.route(
            "**/api/chat",
            lambda route: route.fulfill(
                status=200, content_type="application/json", body=json.dumps(fixture)
            ),
        )
        page.get_by_role("button", name="Ask the live pipeline").click()
        page.wait_for_function("document.querySelector('#messages article') !== null")
        assert "<img src=x" in page.locator(".answer").inner_text()
        assert page.locator("article img").count() == 1
        page.wait_for_function("document.querySelector('article img').naturalWidth > 0")
        assert page.locator("article a").get_attribute("href") == "/api/documents/NOVA-3#page=4"
        page.get_by_role("button", name="New conversation").click()
        assert page.locator("article").count() == 0
        page.unroute("**/api/chat")
        page.route(
            "**/api/chat",
            lambda route: route.fulfill(
                status=503,
                content_type="application/json",
                body=json.dumps({"status": "error", "error_code": "budget_exhausted"}),
            ),
        )
        page.get_by_role("button", name="Ask the live pipeline").click()
        page.wait_for_function(
            "document.querySelector('#status').textContent.includes('budget_exhausted')"
        )
        assert page.locator("article").count() == 0
        assert not errors, errors
        browser.close()
    print("Browser UI checks passed (chat intercepted; original crop loaded from live server).")


if __name__ == "__main__":
    main()
