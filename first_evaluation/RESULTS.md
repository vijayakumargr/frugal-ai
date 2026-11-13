# FRUGAL-AI results (real model runs, cli backend)

885 queries, 5 seeds, mean ± 95% CI over seeds. Quality target q* = 99% of always-large. Seeds resample the split, stream and audits over one set of logged model outputs.

## Model pool on all queries

| Model | Quality | Cost (USD / 1k requests) | Mean latency (s) |
| --- | --- | --- | --- |
| small (claude-haiku-5-5) | 0.824 | 1.53 | 14.3 |
| mid (claude-sonnet-5-5) | 0.856 | 10.71 | 9.3 |
| large (claude-opus-5-5) | 0.891 | 22.84 | 12.4 |

| Workload | n | small | mid | large |
| --- | --- | --- | --- | --- |
| aime | 90 | 0.967 | 0.967 | 1.000 |
| cnndm | 150 | 0.779 | 0.809 | 0.823 |
| codecontests | 165 | 0.770 | 0.794 | 0.818 |
| mmlupro | 400 | 0.828 | 0.873 | 0.917 |
| mtbench | 80 | 0.841 | 0.866 | 0.909 |

## Main results, replay stream (30% repeated requests)

| Method | Cost (USD / 1k requests) | Cost reduction (%) | Quality retained (%) | p95 latency (s) |
| --- | --- | --- | --- | --- |
| Always-large | 23.22 ± 3.55 | 0.0 ± 0.0 | 100.0 ± 0.0 | 41.7 ± 5.7 |
| Always-small | 1.54 ± 0.24 | 93.3 ± 0.9 | 91.5 ± 3.5 | 52.7 ± 11.3 |
| Cache-only | 16.97 ± 2.26 | 26.7 ± 6.3 | 98.9 ± 1.5 | 34.0 ± 5.0 |
| Compression-only | 22.76 ± 3.66 | 2.1 ± 0.9 | 97.7 ± 0.5 | 41.7 ± 5.7 |
| Router-only | 18.82 ± 4.37 | 19.3 ± 9.0 | 98.3 ± 2.8 | 37.5 ± 4.9 |
| Cascade-only | 24.75 ± 4.31 | -6.6 ± 10.5 | 98.9 ± 1.1 | 150.7 ± 26.4 |
| FRUGAL-AI | 11.59 ± 1.77 | 50.0 ± 3.7 | 97.7 ± 1.5 | 58.1 ± 22.1 |

## Main results, distinct queries only

| Method | Cost (USD / 1k requests) | Cost reduction (%) | Quality retained (%) | p95 latency (s) |
| --- | --- | --- | --- | --- |
| Always-large | 24.41 ± 2.82 | 0.0 ± 0.0 | 100.0 ± 0.0 | 43.2 ± 8.0 |
| Always-small | 1.63 ± 0.19 | 93.3 ± 0.7 | 93.1 ± 0.9 | 59.3 ± 5.0 |
| Cache-only | 23.88 ± 3.00 | 2.2 ± 3.8 | 98.8 ± 1.9 | 42.8 ± 8.2 |
| Compression-only | 23.97 ± 2.85 | 1.8 ± 0.4 | 97.8 ± 0.2 | 43.2 ± 8.0 |
| Router-only | 19.47 ± 3.22 | 20.2 ± 9.9 | 98.8 ± 1.4 | 39.7 ± 3.8 |
| Cascade-only | 25.85 ± 3.86 | -5.9 ± 8.7 | 98.6 ± 0.7 | 152.6 ± 21.3 |
| FRUGAL-AI | 15.78 ± 3.36 | 35.2 ± 12.2 | 97.5 ± 1.5 | 93.3 ± 43.6 |

## FRUGAL-AI by workload (distinct queries)

| Workload | Cost reduction (%) | Quality retained (%) |
| --- | --- | --- |
| aime | 57.1 ± 31.8 | 97.3 ± 2.3 |
| cnndm | 43.6 ± 25.9 | 96.6 ± 2.3 |
| codecontests | 17.9 ± 28.5 | 94.0 ± 5.9 |
| mmlupro | 8.6 ± 5.7 | 99.5 ± 0.7 |
| mtbench | 79.9 ± 19.8 | 95.5 ± 0.9 |

## Ablation (replay stream)

| Configuration | Cost (USD / 1k requests) | Cost reduction (%) | Quality retained (%) |
| --- | --- | --- | --- |
| Full FRUGAL-AI | 11.59 ± 1.77 | 50.0 ± 3.7 | 97.7 ± 1.5 |
| without cache | 16.15 ± 3.94 | 30.7 ± 9.6 | 98.9 ± 0.6 |
| without compression | 11.59 ± 1.77 | 50.0 ± 3.7 | 97.7 ± 1.5 |
| without router (cascade from smallest) | 22.82 ± 4.33 | 1.8 ± 10.7 | 96.5 ± 1.5 |
| without cascade (router only) | 9.15 ± 3.86 | 60.8 ± 14.8 | 96.6 ± 3.4 |
| without online controller (fixed lambda) | 11.00 ± 2.26 | 52.4 ± 8.2 | 97.8 ± 1.4 |

## Cost-quality frontier (distinct queries; method · tolerance)

| Method | Cost reduction (%) | Quality retained (%) |
| --- | --- | --- |
| FRUGAL-AI · 0.1 | 89.3 ± 1.4 | 93.0 ± 1.8 |
| Router-only · 0.1 | 93.1 ± 0.8 | 93.2 ± 0.9 |
| Cascade-only · 0.1 | 91.4 ± 0.8 | 93.1 ± 0.9 |
| FRUGAL-AI · 0.08 | 88.3 ± 2.4 | 93.8 ± 1.7 |
| Router-only · 0.08 | 87.8 ± 8.5 | 93.5 ± 1.2 |
| Cascade-only · 0.08 | 87.5 ± 8.6 | 93.2 ± 1.0 |
| FRUGAL-AI · 0.06 | 82.5 ± 4.4 | 95.1 ± 1.3 |
| Router-only · 0.06 | 68.3 ± 13.8 | 95.1 ± 1.4 |
| Cascade-only · 0.06 | 66.3 ± 15.6 | 94.7 ± 1.6 |
| FRUGAL-AI · 0.04 | 71.5 ± 5.8 | 97.0 ± 0.8 |
| Router-only · 0.04 | 50.5 ± 11.0 | 96.3 ± 1.1 |
| Cascade-only · 0.04 | 52.1 ± 10.1 | 95.9 ± 0.8 |
| FRUGAL-AI · 0.02 | 48.3 ± 22.2 | 98.0 ± 0.9 |
| Router-only · 0.02 | 28.4 ± 9.8 | 98.5 ± 1.5 |
| Cascade-only · 0.02 | 13.6 ± 9.3 | 97.7 ± 0.4 |
| FRUGAL-AI · 0.01 | 32.9 ± 13.8 | 98.7 ± 0.7 |
| Router-only · 0.01 | 20.2 ± 9.9 | 98.8 ± 1.4 |
| Cascade-only · 0.01 | -5.9 ± 8.7 | 98.6 ± 0.7 |
| FRUGAL-AI · 0.005 | 18.2 ± 24.0 | 98.6 ± 1.1 |
| Router-only · 0.005 | 11.7 ± 8.7 | 99.5 ± 0.5 |
| Cascade-only · 0.005 | -27.1 ± 12.5 | 99.4 ± 0.6 |
| FRUGAL-AI · 0.0 | -1.9 ± 4.7 | 99.6 ± 0.4 |
| Router-only · 0.0 | 6.5 ± 8.1 | 99.6 ± 0.5 |
| Cascade-only · 0.0 | -45.7 ± 6.8 | 99.7 ± 0.3 |

## Sensitivity (replay stream)

| Setting | Cost reduction (%) | Quality retained (%) | Cache hit rate (%) | False-hit rate (%) |
| --- | --- | --- | --- | --- |
| rho=0.005 | 53.7 ± 7.8 | 97.7 ± 1.4 | 30.4 ± 1.5 | 1.17 ± 1.42 |
| rho=0.01 | 52.6 ± 7.2 | 97.7 ± 1.4 | 30.4 ± 1.5 | 1.17 ± 1.42 |
| rho=0.02 | 50.0 ± 3.7 | 97.7 ± 1.5 | 30.4 ± 1.5 | 1.17 ± 1.42 |
| rho=0.05 | 47.4 ± 2.4 | 97.7 ± 1.4 | 30.4 ± 1.5 | 1.17 ± 1.42 |
| eta=0.01 | 49.5 ± 4.2 | 97.7 ± 1.4 | 30.4 ± 1.5 | 1.17 ± 1.42 |
| eta=0.05 | 50.0 ± 3.7 | 97.7 ± 1.5 | 30.4 ± 1.5 | 1.17 ± 1.42 |
| eta=0.2 | 51.9 ± 3.9 | 97.4 ± 2.1 | 30.4 ± 1.5 | 1.17 ± 1.42 |
| eta=0.5 | 55.5 ± 10.6 | 97.1 ± 2.4 | 30.4 ± 1.5 | 1.17 ± 1.42 |
| tau_cache=0.8 | 50.3 ± 3.9 | 97.1 ± 1.3 | 30.9 ± 1.0 | 1.74 ± 1.21 |
| tau_cache=0.85 | 48.6 ± 5.5 | 98.2 ± 1.1 | 29.7 ± 0.4 | 0.63 ± 0.67 |
| tau_cache=0.9 | 48.0 ± 5.9 | 98.4 ± 1.1 | 28.6 ± 0.3 | 0.41 ± 0.67 |
| tau_cache=0.95 | 47.6 ± 5.9 | 98.7 ± 0.6 | 27.1 ± 0.2 | 0.13 ± 0.26 |
| repeat_share=0.0 | 34.3 ± 13.0 | 97.4 ± 1.6 | 1.3 ± 2.0 | 1.31 ± 1.96 |
| repeat_share=0.15 | 45.5 ± 10.7 | 97.4 ± 1.4 | 15.8 ± 1.8 | 1.38 ± 1.69 |
| repeat_share=0.3 | 55.2 ± 11.3 | 97.8 ± 1.6 | 30.6 ± 1.7 | 1.20 ± 2.15 |
| repeat_share=0.5 | 68.2 ± 9.9 | 98.2 ± 1.3 | 50.0 ± 1.4 | 0.88 ± 1.51 |

## Drift: easy share falls from 80% to 50%

| Method · phase | Cost reduction (%) | Quality retained (%) |
| --- | --- | --- |
| FRUGAL-AI (online) · before | 31.6 ± 13.9 | 98.6 ± 0.4 |
| FRUGAL-AI (online) · after | 23.6 ± 20.3 | 95.5 ± 1.4 |
| FRUGAL-AI (fixed lambda) · before | 32.9 ± 15.1 | 98.6 ± 0.6 |
| FRUGAL-AI (fixed lambda) · after | 25.3 ± 20.3 | 95.4 ± 1.8 |
| Cascade-only (fixed) · before | -6.4 ± 7.5 | 98.6 ± 0.9 |
| Cascade-only (fixed) · after | -21.6 ± 8.1 | 96.4 ± 1.8 |

| Method | Windows > 1 pt below target (%) | Worst window quality retained (%) |
| --- | --- | --- |
| FRUGAL-AI (online) · windows | 60.0 ± 4.4 | 93.3 ± 2.0 |
| FRUGAL-AI (fixed lambda) · windows | 57.0 ± 8.3 | 93.3 ± 2.1 |
| Cascade-only (fixed) · windows | 47.0 ± 28.0 | 94.8 ± 1.9 |

## Budget control after drift

| Policy | Windows over budget (%) | Worst overshoot (%) | Quality retained (%) |
| --- | --- | --- | --- |
| fixed | 80.0 ± 55.5 | 77.1 ± 61.9 | 95.4 ± 1.8 |
| online budget control | 12.0 ± 26.9 | -0.2 ± 6.6 | 91.5 ± 3.8 |

## Tuned parameters and overhead

- tau_cache: 0.834 ± 0.044
- lambda0: 24.800 ± 12.367
- alpha: 0.056 ± 0.017
- tau_small: 0.690 ± 0.119
- tau_mid: 0.474 ± 0.345
- embed: 4.391 ± 0.000
- route: 0.003 ± 0.000
- verifier_call: 3693.548 ± 0.000

## Significance (paired t-test across seeds, replay stream, cost per 1k requests)

| FRUGAL-AI vs. | Mean cost difference (USD / 1k) | p-value |
| --- | --- | --- |
| Cache-only | -5.38 | 0.0009 |
| Compression-only | -11.17 | 0.0002 |
| Router-only | -7.23 | 0.0022 |
| Cascade-only | -13.15 | 0.0005 |
