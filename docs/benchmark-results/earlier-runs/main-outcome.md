# Instrumented A-G main run: provisional outcome

72 exposed questions, 2 passes, 144 initial requests per arm. Independent automated grades are provisional; human review is pending.

| Arm | Completed /144 | Strict provisional passes | Grading unknown | Mean HTTP seconds | Serving assumed-short USD | Long-rate sensitivity USD |
|---|---:|---:|---:|---:|---:|---:|
| A | 143 | 102 | 0 | 25.43 | 10.4189664 | 20.3026628 |
| B | 140 | 124 | 0 | 23.83 | 8.035165072 | 15.263341872 |
| C | 139 | 123 | 0 | 35.25 | 5.548596050 | 10.228223250 |
| D | 138 | 125 | 0 | 24.79 | 9.233849450 | 17.459697450 |
| E | 140 | 124 | 0 | 39.92 | 6.826349702 | 12.624377902 |
| F | 141 | 129 | 1 | 40.73 | 8.829538161 | 16.644000561 |
| G | 105 | 100 | 1 | 113.21 | 33.016275078 | 64.298445878 |

Telemetry is not fully complete: one interrupted server trace lacks terminal boundaries and a process sample. Missing metrics and usage remain unknown; see `missing-telemetry-amendment-v9.json`. The attempt remains in all denominators.

Costs cover reported observable model/API calls only. Search retrieval and Defender are excluded uniformly, not disabled. Grading is separate. Unknown usage is not zero; short/long scenarios do not bound unobserved charges or constitute invoice totals.

Detailed cache, stage, provider, process and paired exploratory measurements: `main-measurements.json`. Immutable strict pricing: `cost-frozen-strict.json`. Review queue: `reviews/human-review-queue.json`. No paid retries or load sweep executed.

Scope reduced by the user to passes 1-2 (1,008 attempts). The original five-pass protocol and completed observations are preserved. The report is an explicitly amended subset, not completion of all five passes. A coordinator-interrupted attempt retains its server usage but has no reconstructed client response or latency. Services restarted for the amendment; interpret cache/latency with this interruption disclosed. Further passes require a new user decision.


Execution changed after45 reservations: serial and Docker parallel segments use different placements and concurrency. The pooled HTTP means above are descriptive only, not a clean latency comparison. Use `parallel-segment-measurements.json` for segment/lane-specific latency and costs. Baseline A remained on its pinned host runtime; B-G ran in Docker containers with shared admission. Additional repetitions require a further user decision.
