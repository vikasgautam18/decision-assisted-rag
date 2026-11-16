# A vs F Comparison After Failed-Execution Retries (v12)

All 1,008 first-attempt artifacts and 5,905 v11 evidence hashes are preserved unchanged.
Retries were executed in isolated directory `af-failed-retries-v12` (`attempt=2`).

## 1. Clean-call retry-adjusted comparison (0 unpriced calls in both A and F)

| Arm | Completed /144 | Strict provisional passes | Pass rate | Grading unknown | Mean latency (s) | Active serving assumed-short USD | Long-rate sensitivity USD | Unpriced serving calls | Cumulative (with failed attempt 1) short USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 144 | 103 | 71.5% | 0 | 24.71 | 10.4781588 | 20.4169976 | 0 | 10.6230468 |
| F | 144 | 133 | 92.4% | 0 | 41.58 | 9.153037020 | 17.266750420 | 0 | 9.154706749 |

## 2. Failed-execution-only retry-adjusted comparison (replacing only non-completed attempt 1 responses)

| Arm | Completed /144 | Strict provisional passes | Pass rate | Grading unknown | Mean latency (s) | Active serving assumed-short USD | Unpriced serving calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 144 | 103 | 71.5% | 0 | 25.21 | 10.4856924 | 4 |
| F | 144 | 133 | 92.4% | 0 | 41.58 | 9.153037020 | 0 |

## 3. Original first-attempt baseline (before retries)

| Arm | Completed /144 | Strict provisional passes | Pass rate | Grading unknown | Mean latency (s) | Serving assumed-short USD | Unpriced serving calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 143 | 102 | 70.8% | 0 | 25.43 | 10.4189664 | 5 |
| F | 141 | 129 | 89.6% | 1 | 40.73 | 8.829538161 | 3 |
