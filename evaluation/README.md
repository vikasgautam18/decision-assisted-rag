# Standalone phase 2 evaluation

> **Final results.** The benchmark results are in
> [`docs/benchmark-results/iso7-benchmark-20261114`](../docs/benchmark-results/iso7-benchmark-20261114/summary.json)
> and summarized in the [project README](../README.md#benchmark-results). This document
> describes the evaluation tooling and **earlier development runs** (a 72-case phase 2
> evaluation, an A/F paired comparison and a first A–G run). Their scores are superseded by
> the isolated benchmark; the published outputs of the A–G and A/F runs are kept in
> `docs/benchmark-results/earlier-runs/` for provenance.

`data/benchmark.jsonl` (57), `data/heldout.jsonl` (15) and the corpus catalogue
are byte-identical copies from the original `semantic_rag` reference project, commit
`3c7f847137db5d0b2d4680e527c39fa2173ccb10`, under the accompanying upstream MIT
license. Original IDs/questions/gold labels/notes are preserved. The historical
heldout filename is not a new blindness claim. No new benchmark cases are added.

The new runner/strict Pydantic validation follow the original field schema and
document/page checks. The code is independently implemented. Legacy runner
issues are not inherited: no hidden inference retries, no grading inside app
latency, no writer-as-sole-judge, no output-shaped service-error answers.

## Protocol

Every question uses a fresh anonymous browser-equivalent client/cookie and
conversation, matching Origin and the real `/api/chat` endpoint. One worker by
default (two supported) shares durable app quotas and per-deployment token
admission. Each response and private trace is checkpointed.
Resuming skips existing attempts, including failures; an application/scorer/data
snapshot change requires a new run ID. No automatic retry of application or
grader failures. Do not mix outputs from different snapshots into one run.

Deterministic checks cover explicit behavior, error/verification state, all
citation identities and markers, a match to original gold pages, actual original
PNG bytes, and replayed source-bound calculation checks. Original gold-source
lists mix alternatives and dependencies, so all-page recall is diagnostic, not
silently treated as mandatory-all-pages. Independent grading checks actual
per-claim support and completeness; matching a page alone is never task success.

The approved evaluator is a separate fixed GPT-5.2 version 2025-12-11 deployment,
`decision-rag-evaluator`, GlobalStandard capacity 30, NoAutoUpgrade, on the shared
Azure AI account. It is not the runtime writer or Decision-1. It receives anonymous
answers, preserved gold/notes and documentary source records, not runtime model
identity, probabilities or verdicts. It is blinded to implementation identity,
not the gold labels. The grader still shares a provider/model family and can err.
Visual semantic scoring uses frozen figure descriptions; original delivery bytes
are verified separately. Human original-image review is required for visual disputes.

Global evaluator cap: 600 calls in `data/evaluation/grading-quota.sqlite3`
(originally 100, then 200, then 220, then 300, now 600); failed reservations
count. Application caps for the corrective work were DI6/Decision2300/aggregate-model550. Those increases
preserve the existing ledgers and grant 50 additional model, 300 Decision and
20 grading reservations for corrective work.
The approved evaluator deployment does not modify existing writer or baseline deployments.

The maintainer subsequently approved a fresh full 72-case rerun, including offline and
browser checks, with cumulative caps DI6/Decision3300/aggregate-model750/grader300.
All previous reservations are retained. The new run is `phase2-full-revalidation`;
its results are separate from the earlier final and targeted runs.

The initial two-worker run exposed shared token-rate throttling and is retained
unchanged. The maintainer approved behavior corrections and a paced full rerun within
remaining caps. Runtime and grader now reserve estimated input/output tokens in
65-second rolling windows at 80% of observed deployment TPM. Estimates use the
larger of o200k tokenizer counts and characters/3 plus overhead/output allowance;
cached vision inputs reserve 16,384/image instead of tokenizing base64 bytes.
Admission waits count toward the app's existing 180-second deadline. A 429 remains
an error for that attempt and puts subsequent calls into cooldown; it is not
silently retried. Other processes sharing Azure quotas can still cause throttling.
This intentionally changes latency, not answer-quality criteria or budgets.

The paced 72-case run completed with 64 application responses, eight validation
errors, 63 independent judgments and one grading error; strict provisional
passes were 52/72, with 24 human-review items. No application throttling was
observed. These are development results, not final acceptance.

Corrective validation uses separately selected original IDs (`--cases q18 ...`)
and a separate run ID/report denominator, never replacing failed cases in the
72-case run. The maintainer approved explicit source-bound Fahrenheit-to-Celsius
conversion. Percentage application/reduction/ratios, distinct-item counts and
typed references to earlier calculations remain deterministically replayed.
HALF_EVEN remains the default; HALF_UP must be explicitly declared.
Numeric operands can reference strictly increasing original source line numbers,
avoiding retyping ambiguous OCR temperature glyphs without normalizing or
loosening exact-quote checks. Cell/page coordinate metadata is not numeric evidence.
Decision-1 separately identifies baseline-specification comparisons, anchoring
their evidence in the reviewed performance sections. Verifiers receive the
permitted calculation definitions while still checking operand applicability.
The clarified grader prompt evaluates collective citation support without
inventing an all-gold-pages rule. Its reasoning effort is now low; these grader
changes are frozen and disclosed, not pooled into earlier scores.

The final frozen run, `data/evaluation/phase2-final/`, captured all 72 attempts
with no application errors: 64 deterministic passes, 71 completed independent
judgments, one grader ServiceError, and 59 strict provisional passes (81.9% of 72).
The grader error on q39 has no recorded HTTP status or detailed cause; it is
unknown and was not retried. The 20-item human-review queue contained 13
non-passing/unknown cases and seven sampled agreements; all are now marked
reviewed with the reviewer's notes.

The maintainer has now completed human review: nine items accepted, two grader errors
(q10 and q11), and nine corrections requested. q39's grader error was accepted
as an appropriate clarification. Counting the nine corrections as failures
yields 63/72 supported by human review plus the 52 non-queued strict passes; the
other 52 cases were not manually sampled. The human decisions and notes remain
in the original final queue.

The maintainer directed correction of all nine cases and approved cumulative caps
DI6/Decision2300/aggregate-model550/grader220. Existing runtime and grading
ledger reservations remain unchanged. Corrective live runs use new run IDs and
do not replace the frozen final 72-case run.

Each corrected case later passed deterministic and independent grading in a
separate targeted run: q08/q13/q55/q66 in `phase2-human-corrections`,
q26/q48/q51/q52 in `phase2-human-correction-followup`, and q28 in
`phase2-q28-revalidation-v2`. Transport failures and an earlier q28 conflict
behavior remain recorded in their original targeted reports. They are not merged
into the final 72-case score.

Six verification-budget abstentions remain on answerable cases; other failures
include missed source conflict, omitted service time limit, historical labeling,
and engine-only/system-combined ambiguity. Some grader judgments are disputable;
neither they nor original gold labels have been silently corrected. Arithmetic
passed 5/5, fleet-wide comparisons 2/2, and figures 6/7. This is exposed
development evidence, not blind confirmation or proof of stability.

New run directories copy runtime/evaluation source and resource/context manifests
alongside hashes, so uncommitted code edits cannot erase the evaluated code.
Original large PDF/prepared/index artifacts stay in their hash-verified locations.
The first exploratory run recorded hashes but did not copy complete source trees.

Strict provisional success requires all deterministic checks and independent
correctness, completeness, supported claims/citations and expected behavior,
without an unresolved gold issue or judge uncertainty. Application failures stay
in the 72-case denominator. Missing/failed grading is unknown, not a quality pass
or an automatic semantic failure. Reports distinguish these categories.

All failures, uncertainty, gold disputes and a stable approximately 10% sample of
agreements enter a human review queue. Scoring never relabels the original data.
Human adjudication and agreed functional acceptance remain required before phase
2 completion. The current corrective work does not calibrate thresholds, establish
load/stability, or authorize phase 3.

## Run

```bash
.venv/bin/python -m evaluation.dataset
# Requires approved live budgets; app must serve the current implementation manifest.
.venv/bin/python -m evaluation.run --execute-approved-evaluation \
  --run-id phase2-initial --grade
.venv/bin/python -m evaluation.report --run-id phase2-initial
```

Default API: loopback `http://127.0.0.1:8765`; no arbitrary remote evaluation URLs.
Run artifacts/snapshots/results/human queue stay under ignored `data/evaluation/`.
Grading is separate from chat latency. Prices/complete monetary cost are unknown;
reported tokens and unknown usage remain distinct.

## Coverage audit

The dataset spans 14 categories and 64 answer, one conflict, four abstain and three
clarify cases. It includes seven figure cases. Audit output records hashes and
coverage. It does not encode multi-turn ownership/scope reset, injected source
instructions, cancellation, deadline/call exhaustion or concurrency as question
labels. Reuse existing unit/fault/browser tests for those gaps first; new gold
questions require demonstrated need rather than automatically expanding the suite.

## Phase 3 preparation: approved A/F-first approach

The maintainer accepted phase 2 and approved starting with the unchanged reference A
versus the current full Decision-assisted F, followed by explanatory controls.
The reference checkout remains at `3c7f847137db5d0b2d4680e527c39fa2173ccb10`.
An external deployment's matching revision cannot be inferred from that checkout.
The reference origin and live comparison spending caps still require agreement.

`evaluation.comparison` provides offline-tested public API adapters and a seeded,
balanced, adjacent-pair A/F schedule. Each request uses a fresh cookie jar,
same-origin header, explicit wall-clock timeout, and no harness retries.
Raw answers and native citation spans are preserved. Missing usage stays unknown.
No Decision-assisted verification verdict or private calculation-trace requirement
is imposed on the external reference.

The maintainer approved one four-case paid paired pilot (q01/q28/q39/q58), eight public
chat requests and up to eight grading requests. Existing cumulative runtime and
grading caps remain unchanged. No full 72-case comparison is yet approved.
`evaluation.paired_pilot` freezes source hashes/schedule, persists intent before
every chat and grading call, skips all prior attempts on resume (including errors),
and uses the same public-output checks and anonymous grading inputs for A and F.
F's native 180-second deadline is unchanged (195-second HTTP allowance); A's
external allowance is 420 seconds with its native internal retries unchanged.
Baseline work may outlive a disconnected client; public request counts are not
a bound on its hidden internal spending.

Run the explicitly approved pilot:

```bash
.venv/bin/python -m evaluation.paired_pilot --execute-approved-pilot \
  --run-id phase3-paired-pilot
```

Original PNG delivery/page metadata is checked, but pixel identity against the
original PDF remains a human-review requirement; baseline crops may differ.
Private results stay under ignored `data/comparison/`. Shared scoring must not
silently assume the new app's private source IDs or internal verifier is available
in the reference. Reference internal retries are unchanged and must be disclosed.
The reference's native timeout/retry behavior differs from F; agree and record
the external deadline before execution rather than modifying A.

### Approved full comparison

After the pilot and a single retry, the maintainer approved all 72 paired cases
(144 initial application requests) plus at most one retry per operational
application or grading failure on either side. Current cumulative caps:
DI6/Decision4800/aggregate-model1200/grader600; no ledgers are reset.
Baseline public requests are bounded to 72 initial + at most 72 retries.
Its hidden internal calls, native retries and cost are not separately bounded
or observable through the public API.

```bash
.venv/bin/python -m evaluation.paired_run --execute-approved-comparison \
  --run-id phase3-full-comparison
```

The runner uses all unchanged questions, balanced seeded adjacent A/F pairs,
fresh sessions and the pilot's common scoring. Original attempts and single
retries stay in separate run directories. Completed incorrect answers are not
retried, nor are successful application calls repeated for a grader error.
Source/configuration changes refuse resume. Interrupted reservations remain
consumed. A pending human queue includes non-passes, disagreements, missing
grading and reproducibly sampled agreements. Earlier human decisions are not
automatically applied to new answers or to the other arm.

Paired counts are descriptive development evidence. No superiority or
significance claim follows without handling fact-family dependence and
adjudicating disagreements. Per-attempt response latency excludes grading;
retry-selected response latency is not total time-to-success.

Generate the consolidated, human-adjusted outcome without inference:

```bash
.venv/bin/python -m evaluation.outcome --run-id phase3-full-comparison
```

This writes `outcome.json` and `outcome.md` in the retry directory, verifies the
review and retry provenance hashes, and preserves original reports. Retry-inclusive
application-attempt time sums failed and successful HTTP attempts, not the
intervening grading/wait time or elapsed time until eventual success. Reported-only
grader tokens exclude unknown usage; application monetary cost remains unknown.

Partial runtime costing can recover additional evidence without inference:

```bash
.venv/bin/python -m evaluation.costs --baseline-audit /path/to/reference-audit.jsonl
.venv/bin/python -m evaluation.outcome
```

The cost command needs an archived `retail-price-snapshot.json` from the official
Azure Retail Prices API in the retry directory. It freezes matching baseline
audit-window events and F private trace calls in `cost-evidence.json`, including
failed attempts and retries but not reused records. Baseline audit events lack
request IDs: the 72-event capture-window match is observational attribution,
not per-request billing correlation. Snapshot input files must be retained.

`cost-report.json`/`.md` provide short-context Global Standard USD retail
model-only estimates, not invoices. Writer cache writes are priced separately
from ordinary input and are not double-counted. Baseline audit cache details
are absent, so its writer charges are scenarios rather than a measured subtotal.
Knowledge Base planning/reasoning, Search/semantic ranking, unknown-usage calls,
unverified embedding meters and amortized ingestion/hosting remain unpriced.
Evaluation grading is separate overhead, not application serving cost.
The outcome generator includes this partial cost section when present; no
total-cost winner can be declared from these data.

### Offline replay preparation

```bash
.venv/bin/python -m evaluation.replay --source-run phase3-full-comparison \
  --run-id phase3-verification-replay-offline
```

This freezes identical final evidence and initial factual drafts, then compares
deterministic-only eligibility with the recorded semantic verification gate at
the unchanged 0.8 cutoff. No writer, retrieval, model or grader calls are made.
The first preparation yielded 58 eligible fixtures out of 72 F response records;
14 lack a complete initial factual-draft/verification boundary (including early
terminal/error paths). All 58 passed their recorded initial verification.
This survivor-selected set cannot establish verification benefit or lack of
benefit; it is an offline mechanism/provenance check, not a scored live B/D arm.
The gate's rejection and confidence-boundary paths are tested with synthetic inputs.

Assessment replay cannot be recovered from these traces: assessment decisions
are recorded, but full pre-assessment candidate pools and their exact batching
are not. Reconstructing those from final selected evidence would bias the
comparison. Fresh capture at the assessment boundary is required before matched
B/C/E evidence-selection experiments. Routing and ordinary-LLM controls also
remain unexecuted.
# Instrumented five-pass A-G rerun

The new rerun tooling is separate from the completed A/F experiment. It does
not authorize paid calls, raise old caps, provision replicas or alter golds.

Implemented preparation includes immutable B-G stage policies, G's categorical
ordinary-model adapter, private correlated spans, complete candidate/draft
capture, an audited logging-only baseline copy, a seeded seven-arm schedule,
durable attempt/spend reservations, pooled grading, separate closed-loop load
cells and provisional offline analysis. F remains the default application.

## Preparation and freeze

1. Run offline tests and read-only deployment/quota preflight:
   `python -m evaluation.preflight --inspect-azure --output NEW_FILE.json`.
   This only reads Azure configuration. Existing output is never overwritten.
2. Resolve the baseline hosted-usage visibility, exact prices/cache semantics,
   billing access and existing cumulative usage. Freeze reviewed guidance in a
   separate overlay; do not edit the original dataset.
3. Use `evaluation.baseline_copy.prepare_copy` to prepare a new explicit
   destination from pinned git blobs. Supply approved environment settings,
   not the reference `.env`.
4. Create a `Protocol` JSON with seven distinct origins and hashes of code,
   original dataset, rate card and review overlay. The main phase requires
   72 original IDs and five passes. Pilot synthetic cases require a separate
   frozen content hash. Evaluator pool entries include actual TPM/RPM capacity.
5. Prepare without spending:
   `python -m evaluation.observed_run prepare --protocol PROTOCOL.json
   --directory NEW_EXPERIMENT_DIRECTORY`.
   Preparation creates 2,520 main schedule slots but no paid-call authorization.
6. Obtain explicit resource/spending approval. A validated `Approval` must match
   the exact protocol hash and specify attempt limits, conservative USD
   reservations, approval reference and cumulative runtime/grading limits.
   Existing reservations are preserved; default historical caps remain unchanged.

`cost_ceiling_kind` explicitly describes a conservative reservation, **not an
invoice guarantee**. Inaccessible hosted billing cannot be made known by setting
an arbitrary dollar cap. No checked-in example contains a real paid approval.

## Serving entry points

Custom B-G instances use the opt-in uvicorn factory
`app.live:create_experiment_app --factory`. Required environment:
`DECISION_RAG_EXPERIMENT`, `DECISION_RAG_ARM`, `DECISION_RAG_PROTOCOL` and
`DECISION_RAG_APPROVAL`; an optional `DECISION_RAG_RESOURCES` selects an approved
replica resource manifest. Sessions/private pipeline traces are isolated by
experiment and arm. The default `app.live:app` stays F and does not enable
experiment middleware.

Baseline A uses `evaluation.baseline_serve:create_baseline_app --factory` in a
fresh process with its frozen SDK dependencies. It additionally requires
`DECISION_RAG_BASELINE_COPY` (the benchmarked copy is published in
[`baseline/config-a`](../baseline/config-a/)) and explicit approved baseline settings. It validates
the copied-source and hook hashes. It does not configure or provision the hosted
agent. Its health endpoint reports identity, not proof of Azure readiness.

Every experiment request carries exact observation/logical request IDs, arm,
case, pass, attempt and phase. Response headers join runner and server traces
without time-window matching. Headers cannot change the configured custom arm,
resource manifest or export directory.

## Explicitly approved collection and grading

`evaluation.observed_run collect` requires `--execute-approved`, `--approval`,
`--preflight`, `--rate-card`, `--review-overlay` and `--directory`. The preflight
record must match the protocol and affirm offline parity, usage visibility,
rates/meter semantics, model/quota checks, cumulative cap approval and frozen
review guidance. Main/load phases also require pilot acceptance. Frozen input
drift prevents execution.

First attempts are durable before dispatch. Interrupted or uncertain attempts
are not automatically resubmitted. `observed_run retry` creates one new attempt
only for recorded operational failures; completed semantic failures are not
retried. Load testing has no external retries.

`evaluation.observed_grade` takes the same gates and `--execute-approved`.
It runs only after initial serving collection is accounted for, balances work
over the frozen evaluator pool, retains full usage and enforces the shared
grading ledger. `--retry-errors` records one separate operational grader retry.
Synthetic pilot cases are source-reviewed, not paid-graded.

Main serving is serial by default. Separate load cells follow the frozen arm
order at concurrency 1/2/4. Extra deployments are not automatically used to
parallelize the main latency comparison; the approved isolation experiment and
balanced placement design are still prerequisites.

## Offline analysis and visibility limits

- `python -m evaluation.observed_report --directory DIRECTORY` produces
  provisional counts, all-attempt/success-only latency, question-clustered
  comparisons and a grouped review queue. Existing human-review queues are
  preserved, not overwritten; reconcile a stale queue explicitly when adding
  later results. Load phases produce separate throughput/concurrency cells.
- `python -m evaluation.observed_spans --spans DIRECTORY --output FILE`
  checks trace parenting/boundaries, exposes incomplete spans and extracts
  observable provider attempts. Repeat `--spans` for runner/server directories.
  Exclusive duration subtracts the union of direct-child intervals; inclusive
  parent/child times must not be summed.
- `evaluation.observed_costs` provides validated exact-meter matching and Decimal
  pricing functions. Missing usage/prices stay unpriced; cache counters are not
  assumed zero. Charge estimates are not invoice reconciliation. Hidden hosted
  children remain a separate unresolved dimension.
- Process CPU describes the whole process during a trace interval; Linux memory
  is lifetime peak RSS, not a per-request allocation or Azure model CPU measure.
  Provider-side connection internals and hidden SDK backoff are unavailable
  unless supported hooks expose them. Public APIs remain non-streaming.

Read-only preflight evidence is under
`data/experiments/preflight-20261104/`. It confirms spare allocation and exact
versions listed for candidate replicas. Two pinned GPT-5.2 evaluator replicas
were subsequently created under explicit bounded-pilot approval, with the
original deployments unchanged. Raw Modern billing
subsequently resolved the exact regional embedding meter and observed model/cache
rates; see `billing-resolution.json`, which qualifies the earlier pricing research.
Hosted Search per-request usage and Defender scanned-token attribution remain
unresolved, but the maintainer explicitly excluded these charges for both
implementations. Set the frozen protocol's scope accordingly:

```json
{
  "cost_scope": {
    "excluded_categories": ["search_retrieval", "defender_ai"],
    "exclusion_reference": "Maintainer: ignore these two for both implementations"
  }
}
```

The exclusion applies uniformly to A-G and is disclosed in preparation, main/load
and cost reports. It does not erase unknown included-model usage or permit a
total-serving-cost claim. Continue capturing retrieval latency and raw billing
evidence; do not disable services or interpret excluded fees as zero.
The exclusions alone are not paid consent. A subsequent explicit approval covers
42 pilot application attempts, up to 35 grades and the two capacity60 evaluator
replicas, under a USD100 estimate-based reservation ceiling, not an invoice cap.
The frozen pilot is `instrumented-ag-pilot-20261104`. A subsequent separate approval
covers the main initial 2,520 application and up to 2,520 grading attempts, with
USD6,048 conservative reservations (USD2/application and USD0.40/grade), not an
invoice cap. Approved cumulative caps preserve prior usage:
DI6/Decision32,000/model20,000/grading3,100. Load, retries, more deployments and
ingestion remain unapproved. Serving is serial; grading follows serving using the
three-deployment evaluator pool. Unknown GPT-5.6 context tiers remain unpriced
with separate rate envelopes, not silently assigned short-context rates.
For the new `instrumented-ag-main-20261104`, the maintainer explicitly accepted
assumption-based estimation: published short rates are the labeled primary
scenario, with long-rate sensitivity and all missing usage still unknown.
Freeze this assumption alongside the new protocol; retain the strict unpriced
report as well as the separate assumed-price report. No numeric cutoff or
complete hosted-usage visibility is asserted.
Strict pre-frozen identity mismatches also remain unpriced; post-pilot analysis
must preserve their original evidence and disclose any later alias correction.
The pilot has now completed 42 application requests and 35 grading reservations:
33 judgments and two local admission rejections, with all119 traces complete.
Future grading placement estimates the unchanged full request before dispatch
and uses only pool members whose 80%-TPM admission window can fit it. A request
too large for every approved replica fails explicitly before any grading starts;
it is never truncated, silently skipped or sent to an unapproved deployment.
The failed pilot grades remain unknown. See the private pilot outcome for
the original frozen report and explicitly post-hoc case-alias analysis.

### Two-pass parallel amendment

The maintainer reduced the main run to two passes initially (1,008 attempts total)
and approved two same-version replicas: writer150 and Decision100. Original
resources, protocol and serial evidence remain unchanged. The run's
`two-pass-amendment/` selects the exact first1,008 schedule identities;
`parallel-amendment/` freezes replica bindings/rates, Docker/runtime provenance,
lane assignment, shared admission and a bounded validation gate.

Baseline A retains its exact host SDK runtime. Custom B-G run in two sets of
Docker containers, alternating original/replica lanes with at most two active
application requests. Containers obtain credentials through an authenticated
host broker; Azure CLI credentials are not copied into images. A non-overlapping
Docker subnet avoids routing collisions with the WSL host. Every custom
container must verify token/admission connectivity before its first paid request.
Only bounded health GETs are retried during startup, never inference POSTs.
Failed setup attempts and interrupted client deliveries retain accounting/spans
and are not resampled.

`parallel-segment-measurements.json` separates serial, validation and parallel-main
cost/latency by arm and lane. The original source remains403-test validated;
additional harness regressions cover shared quota atomicity, cooldowns, exact
two-pass identities, replica pricing and report shapes. Third-party dependencies
are pinned to the host versions; container Python patch and host-versus-container
placement differences are disclosed. Three-worker initial grading follows serving.
More repetitions, retries and the separate load sweep need another decision.

The first repaired-network validation is now complete but **failed** its frozen
continuation gate:13/14 responses completed with complete spans, but G/q49
recorded an original-writer429 and A/q04 recorded a native SDK429 before
completion. The run is paused at61 total application attempts, with no pending
reservations and no grades. Owned serving services are stopped.
`validation-result.json` and `validation-investigation.json` preserve the
decision/evidence; account-wide Monitor snapshots are not exact request
attribution. Do not overwrite that result or treat it as permission to continue.

The subsequent explicit request from the maintainer to resume authorized continuing only
unsubmitted selected attempts. `resume-revision-v6.json` records conservative
custom-writer pacing and baseline quiet intervals; a new validation file
preserves the original failed gate. Six focused regressions cover durable
pacing, unconsumed-ID selection, immutable assignment writes and allowing
in-flight peer work to finish after an error. Two v5 startup attempts remain
consumed, including one interrupted server trace with incomplete telemetry;
unknown usage/process/latency values must not be reconstructed. Reports need
revision/session strata and explicit missing-data disclosure, not a claim of
complete telemetry. Original frozen application source remains unchanged.

Capture a new, private historical billing snapshot without inference:

```bash
python -m evaluation.billing_evidence --read-azure --output NEW_BILLING_FILE.json
```

This bounded collector uses raw ARM Modern fields, not the legacy CLI consumption
projection. It filters approved resource IDs in memory because server filters
may be ignored. Distinct rows with nonunique provider IDs are retained and
diagnosed; exact repetitions are counted and excluded from returned-record
subtotals. Pagination limits and posting lag remain explicit. Missing charges
are not zero; resource/day aggregates are not per-request experiment costs.
The Usage Details API is deprecated, so final invoice reconciliation should use
Cost Details or billing exports rather than assume this preflight API is durable.
