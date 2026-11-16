# Decision-assisted RAG

A standalone dossier question-answering application and a benchmark of
decision-assisted RAG on Microsoft Foundry and Azure AI Search. A small decision
model (`Microsoft-Decision-1`) routes each question, filters the retrieved chunks
and verifies every claim, while a large model (`gpt-5.6-sol`) generates the answer once.

## Status and results

The application implements six-dossier DI ingestion, original crops and versioned
vision descriptions, an isolated hybrid/semantic Search index, multi-platform and
temporal routing, evidence assessment and one bounded expansion, deterministic
source-bound arithmetic, verification/repair, citations, images, UI, anonymous
session-owned conversations and private usage/provenance traces.

### Benchmark results

Seven configurations (A–G) answered 72 questions about six synthetic vehicle dossiers,
twice each: 144 requests per configuration and 1,008 in total (experiment
`iso7-benchmark-20261114`, run on 2026-11-14). Every answer was graded by two
independent `gpt-5.2` judge runs. A strict pass requires all deterministic checks plus
judged accuracy, completeness, supported claims and valid citations. The pass rate is the
average of the two judge runs, with their range in brackets.

| Config | Setup | Strict pass rate | Model cost | Mean latency |
|---|---|---|---|---|
| A | Foundry agent baseline | 71.9% (70.8–72.9%) | $24.14 | 21.1 s |
| B | Custom RAG, no decision model | 86.5% (85.4–87.5%) | $15.13 | 19.1 s |
| C | B + chunk filter | 88.9% (87.5–90.3%) | $13.08 | 17.2 s |
| D | B + claim verifier | 84.4% (83.3–85.4%) | $14.96 | 20.1 s |
| E | B + chunk filter + claim verifier | 89.6% (both runs) | $14.12 | 19.9 s |
| F | Full decision-assisted RAG | 95.1% (94.4–95.8%) | $13.45 | 20.0 s |
| G | Same pipeline as F, large LLM as decider | 93.8% (92.4–95.1%) | $42.82 | 81.2 s |

- F passes 95.1% of answers against 71.9% for the Foundry agent baseline (A) at 44% lower
  model cost, and the gap is above zero under both judges.
- Having the large LLM make every decision (G) matches F's accuracy within the noise of
  this test, but costs 3.2 times as much and takes about 4 times as long.
- The guardrails work best together: F is about 9 points above the custom RAG without a
  decision model (B), with intervals above zero under both judges, while a chunk filter or
  claim verifier alone (C, D, E) is not clearly better than B.

How to read these numbers:

- Each configuration was run once (two passes), so differences of a few points between close
  configurations (for example F and G, or F and E) are within the noise of this test.
- Requests were sent one at a time. Each configuration ran by itself with a 32-minute idle gap
  before the next, and a question's second pass started at least 36 minutes after its first,
  so prompt caching could not carry over. Failed requests count as failures and were not
  retried (6 in A, 2 in B, 1 in C, none in D–G).
- Costs are recorded token usage at the short-context rates confirmed by the Azure bill.
  Config A's cost is the bill for the run day minus the other six configurations, because its
  hosted agent makes model calls that its own usage report omits (524 observed in Azure Monitor
  against 138 reported). Azure AI Search and Microsoft Defender charges are not included.
- Latency is the mean time per completed request. Configs B–G wait on a client-side rate
  limiter that A does not have (2.3 s on average for F, 41 s for G); without that wait F
  averages 17.6 s.
- The figure-image download check is excluded from the strict pass for every configuration
  because it cannot be run for A.

The aggregated results are in
[`docs/benchmark-results/iso7-benchmark-20261114`](docs/benchmark-results/iso7-benchmark-20261114/summary.json):
`summary.json` has per-config quality, calls, tokens, the cost reconciliation, Azure Monitor
totals and Config F's per-step latency, and `quality.json` has judge agreement and paired
comparisons. Per-request logs, billing exports and the run scripts are not included. Earlier
runs (a first A–G run with mixed execution placement and an A/F comparison) are kept in
[`docs/benchmark-results/earlier-runs`](docs/benchmark-results/earlier-runs/) for provenance;
their scores are superseded by the benchmark above. See [evaluation/README.md](evaluation/README.md)
for the evaluation tooling.

A live implementation check is not a benchmark result. The synthetic and live entry points
are explicitly distinct.

The corpus is ATLAS-6, NOVA-3, TERRA-9, VEGA-7, VOLTA-4 and ZENITH-11. No baseline
application, documents or index are overwritten. Local hosting is single-process
and private; do not expose this development server publicly.

### Config A and the original project

Config A is the author's earlier project, `semantic_rag`: a Foundry agent with an Azure AI Search
knowledge base. That repository is private. The parts this benchmark needs from it are included:

- its runtime source, exactly as benchmarked, in [`baseline/config-a`](baseline/config-a/). See
  [`baseline/README.md`](baseline/README.md) for how to verify it;
- the six dossier PDFs in [`data/corpus`](data/corpus/) and the 72 questions in
  [`evaluation/data`](evaluation/data/), copied from commit `3c7f847` of that project. The hashes in
  `data/corpus/manifest.json` can be checked without access to it.

Not included are that project's infrastructure, ingestion and enrichment scripts, tests and docs, so
re-running Config A means provisioning its Azure resources yourself. The tools that fetch from or
copy the original repository (`ingestion/corpus.py`, `evaluation/baseline_copy.py` and
`evaluation/paired_pilot.py`) need access to it. They are not needed to read or verify the
published corpus, questions and baseline source.

## Run locally

Requires Python 3.11 or later.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `/docs` for the API, or send a synthetic lookup:

```bash
curl http://127.0.0.1:8000/api/lookup \
  -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:8000' \
  -d '{"question":"What is the declared combined range of the DEMO-1 Electric variant?"}'
```

The fixture returns 120 km with a `DEMO-1`, revision A, page 1 citation. Its
keyword matcher is test scaffolding, not real retrieval or semantic reasoning.
Other questions may abstain. No request parameter enables live services or
selects a model, index, endpoint, or experimental arm.
Mutating API calls require an explicit same-origin `Origin` or
`Sec-Fetch-Site: same-origin` signal, including CLI/benchmark calls.

`GET /health` reports synthetic execution. `/api/lookup` is an intentionally
stateless development endpoint, not the reference `/api/chat` contract. Do not
expose this development server publicly.

## Validate

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
```

Tests use only synthetic clients and injected faults. They require no Azure
credentials and never intentionally contact Azure. The parity tests that compare the baseline copy
with a checkout of the private original project run only when `SEMANTIC_RAG_REFERENCE` points to
one, so a fresh clone reports some skipped tests.

## Prepare and run the live application

`manifests/azure-resources.json` is committed with placeholder values for the subscription
ID, resource group, storage account and the three service endpoints. Before running
anything against Azure, replace them with your own resource values. Keep that file and the
generated implementation snapshot out of git so that your values are never committed:

```bash
git update-index --skip-worktree manifests/azure-resources.json data/implementation-manifest.json
```

Requires server-side Azure CLI sign-in to the configured existing resources,
the pinned source artifacts, approved data-plane permissions and paid call caps.
No end-user sign-in or Azure token is requested by the browser.

```bash
.venv/bin/python -m ingestion.full --execute-approved-checkpoint
.venv/bin/python -m app.provenance
.venv/bin/python -m uvicorn app.live:app --host 127.0.0.1 --port 8765 --no-access-log
```

Open `http://127.0.0.1:8765/` on the execution host, using private port forwarding
if needed. If already prepared, start only the server. Startup verifies the
implementation snapshot; after intentional code/source changes regenerate that
snapshot and disclose them before claiming comparable results.

Requests consume a persistent call quota recorded in a local ledger
(`data/live-checkpoint/quota.sqlite3`). By default it allows 6 Document Intelligence
analyses, 4,800 Decision-1 calls and 1,200 embedding, vision and writer model calls in
total. `/api/status` shows the remaining calls, and review chats consume the same
allowance. Per-request limits are 180 seconds, 20 decisions, five model calls, one
expansion and three repairs. Exhaustion is an explicit error. Do not reset the ledger.

Try a specific variant question, a comparison naming two platforms, an explicitly
superseded revision question, a source-bound numeric difference, or an original
flowchart request. Follow-ups inherit user intent; explicitly name a new platform
to switch scope. Original PDF citations and selected images open from local,
allowlisted corpus routes. Description-only visual limitations remain explicit.

## Limitations

The benchmark used one synthetic corpus and sequential single-user requests, so it does not
measure concurrency or load behavior. Each configuration was run once (two passes), grades
come from two automated `gpt-5.2` judge runs, and the decision thresholds (for example 0.80)
are not calibrated. Costs exclude Azure AI Search and Microsoft Defender charges.
