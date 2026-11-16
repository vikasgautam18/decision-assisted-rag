"""M8 agent runtime: the dossier agent = gpt-5.6-sol + the knowledge base attached as an MCP tool, called through
the Responses API with the caller's Entra token (fallback while the Foundry Agent Service project is unavailable;
`setup_agent.py` holds the Foundry Agent Service definition with the same instructions).

run(question, history=None, kb=C.KB_NAME) -> {answer, citations, figures, kb_calls, seconds}
CLI: python3 -m backend.agent.ask "question" [--kb dossier-kb]
"""
import json
import os
import time

import requests
from openai import APIConnectionError, APITimeoutError, BadRequestError

from backend.agent.setup_agent import AGENTS, INSTRUCTIONS, mcp_url
from common import config as C
from common.answer import cited_spans, figures_from_text

RESPONSES = f"{C.AI_ENDPOINT.replace('cognitiveservices', 'openai')}/openai/v1/responses"


def run_foundry(question, history=None, kb=C.KB_NAME):
    """Same agent, hosted in Foundry Agent Service (project connection = managed identity to the KB's MCP endpoint)."""
    from azure.ai.projects import AIProjectClient
    from azure.identity import AzureCliCredential
    t0 = time.time()
    oc = AIProjectClient(endpoint=C.project_endpoint(), credential=AzureCliCredential()).get_openai_client()
    for attempt in range(6):
        try:
            r = oc.responses.create(input=[*(history or []), {"role": "user", "content": question}],
                                    extra_body={"agent_reference": {"name": AGENTS[kb], "type": "agent_reference"}}, timeout=170)
            break
        except (APITimeoutError, APIConnectionError):
            if attempt >= 1:
                return {"answer": "ERROR the agent timed out; please retry", "citations": [], "figures": [], "kb_calls": 0,
                        "seconds": round(time.time() - t0, 1)}
        except BadRequestError as e:
            if "TooManyRequests" not in str(e) or attempt == 5:
                return {"answer": f"ERROR {str(e)[:300]}", "citations": [], "figures": [], "kb_calls": 0,
                        "seconds": round(time.time() - t0, 1)}
            time.sleep(15 * (attempt + 1))
    out = [o.model_dump() for o in r.output]
    answer = r.output_text or ""
    calls = [o for o in out if o["type"] == "mcp_call"]
    return {"answer": answer,
            "citations": [{"doc": d, "page_start": a, "page_end": b} for d, a, b in cited_spans(answer)],
            "figures": figures_from_text(answer), "kb_calls": len(calls),
            "kb_errors": [c["error"] for c in calls if c.get("error")], "seconds": round(time.time() - t0, 1),
            "tokens": {"in": getattr(r.usage, "input_tokens", 0), "out": getattr(r.usage, "output_tokens", 0)}}


def run(question, history=None, kb=C.KB_NAME, search_token=None):
    if os.getenv("AGENT_BACKEND", "foundry") == "foundry":
        return run_foundry(question, history, kb)
    return run_responses(question, history, kb, search_token)


def run_responses(question, history=None, kb=C.KB_NAME, search_token=None):
    """history: list of {"role": "user"|"assistant", "content": str} from earlier turns."""
    t0 = time.time()
    body = {"model": C.CHAT_DEPLOYMENT, "instructions": INSTRUCTIONS,
            "input": [*(history or []), {"role": "user", "content": question}],
            "tools": [{"type": "mcp", "server_label": "knowledge_base", "server_url": mcp_url(kb),
                       "authorization": search_token or C.get_token("https://search.azure.com"),
                       "allowed_tools": ["knowledge_base_retrieve"], "require_approval": "never"}],
            "max_output_tokens": 4000}
    for attempt in range(6):
        r = requests.post(RESPONSES, headers=C.cognitive_headers(), json=body, timeout=300)
        if r.ok:
            break
        time.sleep(float(r.headers.get("retry-after", 8 * (attempt + 1))))
    else:
        return {"answer": f"ERROR {r.status_code}: {r.text[:300]}", "citations": [], "figures": [], "kb_calls": 0,
                "seconds": round(time.time() - t0, 1)}
    out = r.json()["output"]
    answer = "".join(c.get("text", "") for o in out if o["type"] == "message" for c in o["content"]
                     if c["type"] == "output_text")
    calls = [o for o in out if o["type"] == "mcp_call"]
    return {"answer": answer,
            "citations": [{"doc": d, "page_start": a, "page_end": b} for d, a, b in cited_spans(answer)],
            "figures": figures_from_text(answer), "kb_calls": len(calls),
            "kb_errors": [c["error"] for c in calls if c.get("error")], "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--kb", default=C.KB_NAME)
    a = ap.parse_args()
    res = run(a.question, kb=a.kb)
    print(res["answer"])
    print("\n--", json.dumps({k: v for k, v in res.items() if k != "answer"}))
