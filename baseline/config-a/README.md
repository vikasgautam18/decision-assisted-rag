# Semantic RAG demo: technical dossiers on Azure AI Search and Microsoft Foundry

> **Demo / reference example only. Not production-ready.**
> This repository illustrates an approach to accurate retrieval over complex technical PDFs and spreadsheets.
> It is not hardened, load-tested or security-reviewed, and must not be deployed to production as-is. See
> [Not for production](#not-for-production).

## What it demonstrates
Technical dossiers (PDFs) are ingested into Azure AI Search and answered by a Microsoft Foundry agent with citations.
The demo targets common RAG failure modes: wrong variant, superseded revision, broken tables, uncited diagrams.

- **Layout-aware ingestion**: Azure Document Intelligence Layout, structure-aware chunking that keeps tables whole,
  and a contextual prefix (platform, revision, section) on every chunk.
- **Metadata for accuracy**: filterable fields such as `is_superseded`, so superseded appendix data is excluded by default.
- **Hybrid retrieval**: keyword + vector (`text-embedding-3-large`, 3072 dims) + semantic ranker.
- **Agentic retrieval**: Foundry IQ knowledge sources and knowledge bases, exposed to a Foundry agent.
- **Figures**: diagrams are extracted, described with a vision model, indexed and shown beside the answer.
- **Excel analyst**: a Code Interpreter agent for spreadsheet questions, with recalculation and validation steps.
- **Anonymous sandbox UI and API**: no browser sign-in; one configured knowledge base and separate browser sessions.
  Azure service authentication is retained. Localhost/private access only.
- **Evaluation**: a 57-question benchmark and a 30-case Excel suite.

## Layout
| Path | Purpose |
|---|---|
| `common/` | Central config (`.env`), shared answer logic |
| `ingest/` | SharePoint/local sync, Document Intelligence parsing, chunking, indexing |
| `enrich/` | Figure extraction and descriptions |
| `kb/` | Knowledge sources and knowledge bases |
| `backend/` | Anonymous FastAPI app, browser sessions, agent setup, Excel analyst |
| `web/` | Single-page UI |
| `infra/` | Azure deploy, smoke test, teardown scripts |
| `eval/`, `tests/` | Benchmark and offline tests |
| `docs/` | Guides, architecture diagrams, demo script |
| `track2/`, `track3/` | Experimental chunking and comparison code |

## Quick start
Prerequisites: Python 3.10+, Azure CLI 2.80+ (`az login`), an Azure subscription with resource-creation and
role-assignment permissions, Poppler tools (PDF figures/checks), and LibreOffice Calc (Excel analyst only).

```bash
pip install -r requirements.txt
cp .env.example .env             # SOURCE_MODE=local; set subscription, names and verified regions first
az login                        # operator authentication to Azure services, not browser sign-in
python3 -m common.config --check --stage base
./infra/deploy.sh preflight      # read-only; validates providers and region support
./infra/deploy.sh                # BASE ONLY: resources, RBAC and containers; billable resources
./infra/deploy.sh verify --stage base
./infra/deploy.sh discover       # read-only account model catalog and regional quota
# Select exact model/version/SKU/capacity in .env before continuing; no automatic substitutions.
./infra/deploy.sh models
./infra/deploy.sh project        # creates project/identity/roles and verifies Agent Service readiness
./infra/deploy.sh sync-env       # preview confirmed values; does not write
./infra/deploy.sh sync-env --apply
./infra/deploy.sh verify --stage all  # includes billable model calls and document analysis
# Local PDFs require no application registration, Graph consent or security groups.
./rebuild.sh                     # ingest documents, build index, figures and knowledge bases
python3 -m backend.agent.setup_agent
uvicorn backend.app:app --host 127.0.0.1 --port 8000   # one worker; open http://localhost:8000
python3 -m eval.run_benchmark --workers 2
pytest tests -q                  # offline tests
```
Full setup (anonymous sandbox, Azure roles, optional SharePoint permissions, settings and known limits):
[docs/bring-your-own.md](docs/bring-your-own.md).
See [Permissions and roles](docs/bring-your-own.md#permissions-and-roles) and the
[sandbox walkthrough](docs/bring-your-own.md#anonymous-sandbox-walkthrough) for local configuration and browser checks.

**Access boundary:** anyone who can reach this backend can query demo data, retrieve figures, upload workbooks,
and incur Azure costs. Do not expose it to the internet. For a remote host use private port forwarding or an
access-controlled tunnel; share only behind an explicit network/access boundary. Use fictional or approved data.
HTTP-only cookies keep independent browsers' conversations and workbooks separate, but do not authenticate people.
Use one worker: state and cookie signing keys live in memory and are lost on restart. Tabs sharing cookies share
the Excel workbook. Only `KB_NAME` is available through the app; trace-KB provisioning is unchanged.

The HTTP benchmark path (`python3 -m eval.run_benchmark --api http://localhost:8000`) also needs no application
token; its Azure-backed retrieval and grading still incur costs. Offline browser coverage is available with
`python3 -m pytest tests/test_ui.py -q` when Playwright and its Chromium browser are installed.

Infrastructure deployment does not host the FastAPI application. A western-US
candidate split is `LOC_SEARCH=westus2` and `LOC_FOUNDRY=westus3`; confirm regional support, model offerings and quota
before selecting it. Preflight fails closed when it cannot confirm support. Running `deploy.sh` without a phase no
longer creates model deployments. Existing resources are validated, never automatically replaced.

## Documentation
- [design.md](design.md): design and scope
- [docs/bring-your-own.md](docs/bring-your-own.md): configuring your own resources, models and SharePoint
- [docs/demo-script.md](docs/demo-script.md): 15-minute demo walkthrough
- [docs/excel-analyst.md](docs/excel-analyst.md) and [docs/excel-architecture.md](docs/excel-architecture.md): the Excel use case
- [docs/acceptance-report.md](docs/acceptance-report.md): test results

## Not for production
This code is a starting point to learn from. Before anything similar goes live, you would need at least:

- **Security review and hardening**: threat modelling, secret management (Key Vault and managed identity rather
  than `.env` and `.secrets/`), network isolation (private endpoints, no public access), and review of the Entra app
  and SharePoint permissions when used. The demo storage account may allow public network access with Entra ID auth only.
- **Authentication and authorization**: this branch intentionally removes application sign-in and group entitlement.
  It has no user identity or document-level access controls. Add appropriate authentication, authorization and
  security trimming before considering non-sandbox use.
- **Reliability and scale**: Excel sessions are held in memory, there is no persistence, queueing or autoscaling,
  and models can return intermittent refusals or 429s.
- **Data handling**: sample data is fictional ("Northwind" dossiers). Real data needs classification, retention
  and privacy controls.
- **Parameterisation**: document parsing (revision and appendix patterns), answer instructions and eval data are
  specific to the sample corpus.
- **Monitoring and operations**: logging, alerting, cost controls, CI/CD, infrastructure as code and disaster recovery.
- **Evaluation on your data**: the benchmark covers the sample corpus only.
- **Preview features**: agentic retrieval features (query planning, answer synthesis) and the Foundry IQ MCP
  connection are preview and may change.

Licensed under the [MIT License](LICENSE). Provided as is, without warranty or support. Azure usage incurs cost; run `infra/teardown.sh` to remove resources
the demo created (never against `USE_EXISTING=true` resources).
