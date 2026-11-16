# Config A baseline source

`config-a/` is the source of **Config A**, the Foundry agent baseline in the benchmark. It is the
runtime part of the author's earlier project, `semantic_rag` (private), exactly as it was
benchmarked, under the MIT license in `config-a/LICENSE`.

## What is here

- `config-a/backend/`: the API, the Foundry agent definition and its instructions
  (`backend/agent/setup_agent.py`), the agent runtime and an Excel analyst.
- `config-a/common/`: configuration and the answer and citation helpers.
- `config-a/web/`: the single-page UI.
- `config-a/BASELINE_COPY_MANIFEST.json`: the Git blob and SHA-256 of every file.

## Why it is what ran

- The files are copied byte for byte from commit `3c7f847137db5d0b2d4680e527c39fa2173ccb10` of the
  original project. `evaluation/baseline_copy.py` builds the copy from Git objects, without edits.
- `implementation_sha256` in the manifest is
  `5d6ec6d4eced71a6d424e14d344fdbd21238422f0d0446f2f0274e6b5512a5f5`. This is the Config A
  snapshot recorded in
  [`docs/benchmark-results/earlier-runs/protocol.json`](../docs/benchmark-results/earlier-runs/protocol.json)
  (`targets.A.application_sha256`), and the value frozen for the final benchmark.
- `tests/test_published_baseline.py` checks the folder against the manifest on every test run.
- `ruff.toml` excludes the folder from linting and formatting, because any edit would break the
  hashes.

## What is not here

The original project's infrastructure, ingestion, enrichment and evaluation scripts, tests and docs
(`infra/`, `ingest/`, `enrich/`, `kb/`, `eval/`, `tests/`, `docs/`) stay private.
`config-a/README.md` is the original project's README and links to some of them, so those links do
not resolve here. To run Config A again, you must provision the Azure resources, search index and
knowledge base yourself.

## How the benchmark used it

`evaluation/baseline_serve.py` validates this folder against the manifest and serves it with
observability hooks added from outside, without changing the source. Set
`DECISION_RAG_BASELINE_COPY` to `baseline/config-a`, plus the approved baseline settings described
in [`evaluation/README.md`](../evaluation/README.md).
