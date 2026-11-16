# Copilot Instructions

## Communication
- Use clear, simple English.
- Avoid slang, idioms, and unnecessary jargon. Explain technical terms briefly when needed.
- Keep explanations direct and easy to follow.
- Do not agree with the user by default. Assess ideas critically. If you think the user is mistaken, explain why respectfully and suggest a better approach. If you are unsure, say so.

## Build, Lint, and Test Commands

### Setup and Local Server
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,browser]"
.venv/bin/python -m uvicorn app.live:app --host 127.0.0.1 --port 8765 --no-access-log
```

### Python Unit Tests (`pytest`) and Linting (`ruff`)
- Run the full offline test suite. Parity tests that need a checkout of the private original project run only when `SEMANTIC_RAG_REFERENCE` points to one, and are skipped otherwise:
  ```bash
  .venv/bin/pytest
  ```
- Run a single test file:
  ```bash
  .venv/bin/pytest tests/test_live_pipeline.py
  ```
- Run a single test function:
  ```bash
  .venv/bin/pytest tests/test_experiment.py -k test_operational_retry_uses_new_identity
  ```
- Run linter and formatter checks:
  ```bash
  .venv/bin/ruff check .
  .venv/bin/ruff format --check .
  ```

## High-Level Architecture

This repository implements the **Decision-Assisted RAG** six-dossier application and its 7-arm experimental benchmark (`Arms A–G`) on Microsoft Foundry and Azure AI Search:

1. **Application Layer (`app/`)**:
   - `app/live.py`: FastAPI live application (`create_live_app` and `create_experiment_app`), serving `/api/chat`, `/api/figures/{key}`, `/api/documents/{key}`, `/api/status`, and `/health`.
   - `app/provenance.py`: Hash-verifies all code, manifest, and prepared corpus artifacts against `data/implementation-manifest.json` before startup.
   - `app/experiment_boundary.py`: ASGI middleware that validates experiment identity headers (`X-Experiment-ID`, `X-Observation-ID`, `X-Experiment-Arm`, etc.) and records monotonic request/stage spans.

2. **Decision-Assisted RAG Pipeline (`pipelines/`)**:
   - `pipelines/arms.py`: Defines `ArmPolicy` feature toggles across custom arms `B` through `G` (`routing`, `assessment`, `verification`, `ordinary_decisions`). Arm `A` serves the isolated Foundry Agent baseline (`evaluation/baseline_serve.py`).
   - `pipelines/live.py`: Implements the 6-stage pipeline (`routing-v2` -> Hybrid Search + Section 1 injection -> `assessment-v1` -> `coverage-v1` -> `gpt-5.6-sol` writer + exact `Decimal` calculations -> `verification-v1` with up to 3 bounded repair attempts inside a 180s deadline).

3. **Clients & Telemetry (`clients/`)**:
   - `clients/models.py`, `clients/search.py`, `clients/ordinary_decisions.py`: Wrappers for `Microsoft-Decision-1` (`/providers/microsoft/v1/systemone`), `gpt-5.6-sol`, `text-embedding-3-large`, and Azure AI Search (`decision-rag-chunks-v1`).
   - `clients/rate.py` & `clients/telemetry.py`: Rolling 65-second token/request admission windows and durable JSONL span exporters.

4. **Evaluation & Cost Reconciliation (`evaluation/`)**:
   - `evaluation/dataset.py`: Loads the 72 benchmark and held-out questions (`evaluation/data/*.jsonl`).
   - `evaluation/observed_run.py`, `evaluation/observed_grade.py`, `evaluation/observed_cost_report.py`, `evaluation/observed_report.py`: Schedule generation, durable SQLite spend accounting, pooled `gpt-5.2` independent grading, and token/cache cost reconciliation.

## Key Conventions

### Python
- Follow the project's existing structure, style, and dependency choices (target Python 3.11+, Ruff line length 100, rules `E`, `F`, `I`, `UP`, `B`, `ASYNC`).
- Use clear names, type hints, and focused functions.
- Handle errors explicitly; avoid broad exception handling and silent fallbacks.
- Do not add a dependency unless it is needed and fits the project.

### AI Features & Benchmark Integrity
- Follow the project's existing AI architecture and use its configured models and services.
- Do not hard-code API keys, credentials, or other secrets.
- `baseline/config-a/` is the hash-pinned source of Config A. Never edit, reformat or lint it. `tests/test_published_baseline.py` checks it against `BASELINE_COPY_MANIFEST.json`.
- Note that `app/provenance.py` hashes source files in `app/`, `clients/`, `ingestion/`, `pipelines/`, and `pyproject.toml` against `data/implementation-manifest.json`. If you modify code in those folders, regenerate the manifest (`python -m app.provenance`) when appropriate, and never overwrite frozen experiment outputs (`write_private_report` enforces `os.O_EXCL`).
- Keep model calls easy to test; mock external AI services in unit tests rather than making live calls.
- Make tests deterministic and check important AI behavior, including relevant failure cases.

### Testing
- Always write or update unit tests for every code change.
- Run the relevant unit tests and report the results.
