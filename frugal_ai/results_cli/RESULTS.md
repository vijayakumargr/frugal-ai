# FRUGAL-AI results (real model runs, cli backend)

885 queries, 10 seeds, mean ± 95% CI over seeds. Seeds resample the split, stream and audits over one set of logged model outputs. Every tuned method keeps a setting only if its quality on the calibration split clears the target by 1 standard error(s).

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

## Main results, replay stream (30% repeated requests), target 98% of always-large

| Method | Cost (USD / 1k requests) | Cost reduction (%) | Quality retained (%) | p95 latency (s) | Seeds meeting target (%) |
| --- | --- | --- | --- | --- | --- |
| Always-large | 22.81 ± 1.84 | 0.0 ± 0.0 | 100.0 ± 0.0 | 40.6 ± 3.0 | 100 |
| Always-small | 1.52 ± 0.13 | 93.3 ± 0.5 | 92.4 ± 1.6 | 52.3 ± 7.3 | 0 |
| Cache-only | 16.88 ± 1.29 | 25.9 ± 2.8 | 99.3 ± 0.4 | 34.1 ± 3.1 | 100 |
| Compression-only | 22.35 ± 1.88 | 2.1 ± 0.4 | 97.6 ± 0.4 | 40.6 ± 3.0 | 40 |
| Router-only | 18.19 ± 2.17 | 20.5 ± 4.4 | 98.7 ± 1.2 | 37.3 ± 3.5 | 80 |
| Cascade-only | 21.41 ± 3.29 | 6.9 ± 9.8 | 98.2 ± 0.9 | 134.8 ± 17.7 | 60 |
| FRUGAL-AI | 11.20 ± 1.48 | 50.8 ± 5.2 | 98.5 ± 0.7 | 57.6 ± 12.7 | 70 |
| Router-only (per-workload floors) | 18.59 ± 2.67 | 19.0 ± 7.5 | 99.4 ± 0.5 | 38.5 ± 5.1 | 100 |
| FRUGAL-AI (per-workload floors) | 13.15 ± 1.57 | 42.2 ± 5.8 | 97.7 ± 0.8 | 65.6 ± 10.3 | 30 |

## Main results, distinct queries only, target 98% of always-large

| Method | Cost (USD / 1k requests) | Cost reduction (%) | Quality retained (%) | p95 latency (s) | Seeds meeting target (%) |
| --- | --- | --- | --- | --- | --- |
| Always-large | 23.71 ± 1.70 | 0.0 ± 0.0 | 100.0 ± 0.0 | 42.7 ± 4.5 | 100 |
| Always-small | 1.58 ± 0.09 | 93.3 ± 0.5 | 92.8 ± 0.5 | 57.0 ± 3.7 | 0 |
| Cache-only | 23.49 ± 1.74 | 0.9 ± 0.7 | 99.3 ± 0.5 | 42.5 ± 4.6 | 90 |
| Compression-only | 23.28 ± 1.71 | 1.8 ± 0.2 | 97.9 ± 0.1 | 42.7 ± 4.5 | 20 |
| Router-only | 18.73 ± 1.89 | 21.1 ± 4.8 | 98.9 ± 0.6 | 40.1 ± 4.4 | 90 |
| Cascade-only | 22.29 ± 3.27 | 6.7 ± 8.9 | 98.0 ± 0.5 | 142.6 ± 15.2 | 50 |
| FRUGAL-AI | 15.71 ± 2.08 | 33.8 ± 7.0 | 98.1 ± 0.5 | 86.7 ± 21.1 | 50 |
| Router-only (per-workload floors) | 19.29 ± 2.32 | 18.9 ± 6.9 | 99.1 ± 0.4 | 41.5 ± 5.4 | 100 |
| FRUGAL-AI (per-workload floors) | 18.30 ± 2.14 | 22.8 ± 7.1 | 97.6 ± 0.6 | 91.1 ± 17.3 | 50 |

## Main results, replay stream (30% repeated requests), target 99% of always-large

| Method | Cost (USD / 1k requests) | Cost reduction (%) | Quality retained (%) | p95 latency (s) | Seeds meeting target (%) |
| --- | --- | --- | --- | --- | --- |
| Always-large | 22.81 ± 1.84 | 0.0 ± 0.0 | 100.0 ± 0.0 | 40.6 ± 3.0 | 100 |
| Always-small | 1.52 ± 0.13 | 93.3 ± 0.5 | 92.4 ± 1.6 | 52.3 ± 7.3 | 0 |
| Cache-only | 17.82 ± 1.65 | 22.0 ± 3.1 | 99.8 ± 0.2 | 35.2 ± 3.2 | 100 |
| Compression-only | 22.35 ± 1.88 | 2.1 ± 0.4 | 97.6 ± 0.4 | 40.6 ± 3.0 | 0 |
| Router-only | 19.38 ± 2.19 | 15.3 ± 4.2 | 99.6 ± 0.3 | 39.3 ± 4.2 | 90 |
| Cascade-only | 27.93 ± 3.68 | -22.0 ± 10.1 | 99.3 ± 0.4 | 149.8 ± 15.2 | 80 |
| FRUGAL-AI | 14.75 ± 1.97 | 35.4 ± 5.9 | 99.1 ± 0.7 | 53.8 ± 17.0 | 50 |
| Router-only (per-workload floors) | 19.83 ± 2.86 | 13.6 ± 8.1 | 99.7 ± 0.3 | 41.1 ± 5.2 | 100 |
| FRUGAL-AI (per-workload floors) | 15.64 ± 1.88 | 31.5 ± 6.0 | 99.1 ± 0.6 | 69.0 ± 11.6 | 40 |

## Main results, distinct queries only, target 99% of always-large

| Method | Cost (USD / 1k requests) | Cost reduction (%) | Quality retained (%) | p95 latency (s) | Seeds meeting target (%) |
| --- | --- | --- | --- | --- | --- |
| Always-large | 23.71 ± 1.70 | 0.0 ± 0.0 | 100.0 ± 0.0 | 42.7 ± 4.5 | 100 |
| Always-small | 1.58 ± 0.09 | 93.3 ± 0.5 | 92.8 ± 0.5 | 57.0 ± 3.7 | 0 |
| Cache-only | 23.67 ± 1.75 | 0.2 ± 0.3 | 99.8 ± 0.3 | 42.7 ± 4.5 | 90 |
| Compression-only | 23.28 ± 1.71 | 1.8 ± 0.2 | 97.9 ± 0.1 | 42.7 ± 4.5 | 0 |
| Router-only | 20.04 ± 1.74 | 15.5 ± 3.8 | 99.4 ± 0.2 | 41.9 ± 4.0 | 90 |
| Cascade-only | 28.90 ± 3.23 | -21.8 ± 10.2 | 99.1 ± 0.5 | 154.9 ± 10.9 | 60 |
| FRUGAL-AI | 19.65 ± 2.21 | 17.1 ± 6.6 | 98.9 ± 0.4 | 72.8 ± 21.4 | 40 |
| Router-only (per-workload floors) | 20.59 ± 2.58 | 13.6 ± 7.3 | 99.3 ± 0.4 | 43.4 ± 5.6 | 80 |
| FRUGAL-AI (per-workload floors) | 20.91 ± 2.39 | 11.9 ± 7.2 | 98.8 ± 0.4 | 93.1 ± 17.7 | 30 |

## By workload (distinct queries, target 98%; method · workload)

| Method | Cost reduction (%) | Quality retained (%) |
| --- | --- | --- |
| Router-only · aime | 90.2 ± 2.9 | 95.8 ± 1.4 |
| Router-only · cnndm | 5.3 ± 4.6 | 99.5 ± 0.4 |
| Router-only · codecontests | 1.5 ± 2.8 | 99.9 ± 0.3 |
| Router-only · mmlupro | 32.4 ± 10.9 | 99.2 ± 1.0 |
| Router-only · mtbench | 31.8 ± 10.6 | 98.1 ± 0.8 |
| FRUGAL-AI · aime | 64.4 ± 16.9 | 97.6 ± 1.4 |
| FRUGAL-AI · cnndm | 34.5 ± 14.3 | 98.1 ± 1.1 |
| FRUGAL-AI · codecontests | 14.4 ± 17.6 | 96.6 ± 2.1 |
| FRUGAL-AI · mmlupro | 11.7 ± 5.1 | 99.2 ± 0.7 |
| FRUGAL-AI · mtbench | 78.8 ± 10.3 | 96.1 ± 1.6 |
| Router-only (per-workload floors) · aime | 26.5 ± 27.7 | 98.0 ± 2.0 |
| Router-only (per-workload floors) · cnndm | 23.9 ± 10.6 | 98.1 ± 1.2 |
| Router-only (per-workload floors) · codecontests | 11.7 ± 16.1 | 98.6 ± 1.7 |
| Router-only (per-workload floors) · mmlupro | 24.4 ± 6.6 | 100.1 ± 0.7 |
| Router-only (per-workload floors) · mtbench | 26.9 ± 10.1 | 98.8 ± 0.8 |
| FRUGAL-AI (per-workload floors) · aime | 71.0 ± 17.7 | 97.1 ± 1.3 |
| FRUGAL-AI (per-workload floors) · cnndm | 19.6 ± 11.1 | 98.9 ± 0.6 |
| FRUGAL-AI (per-workload floors) · codecontests | 8.9 ± 16.5 | 96.6 ± 2.3 |
| FRUGAL-AI (per-workload floors) · mmlupro | 30.2 ± 8.5 | 97.6 ± 1.0 |
| FRUGAL-AI (per-workload floors) · mtbench | 25.8 ± 12.8 | 98.3 ± 1.2 |

## Ablation (replay stream, target 98%)

| Configuration | Cost (USD / 1k requests) | Cost reduction (%) | Quality retained (%) | Seeds meeting target (%) |
| --- | --- | --- | --- | --- |
| Full FRUGAL-AI | 11.20 ± 1.48 | 50.8 ± 5.2 | 98.5 ± 0.7 | 70 |
| without cache | 13.70 ± 2.67 | 40.4 ± 8.5 | 98.7 ± 0.8 | 90 |
| with prompt compression | 11.20 ± 1.48 | 50.8 ± 5.2 | 98.5 ± 0.7 | 70 |
| without router (cascade from smallest) | 20.00 ± 2.85 | 12.8 ± 7.5 | 98.5 ± 0.7 | 80 |
| without cascade (three-way router) | 14.45 ± 1.69 | 36.0 ± 8.4 | 98.6 ± 0.6 | 80 |
| with online quality controller | 11.69 ± 1.54 | 48.8 ± 4.9 | 98.5 ± 0.7 | 70 |

## Cost reduction at matched quality retained (distinct queries; method · quality retained)

| Method | Cost reduction (%) |
| --- | --- |
| FRUGAL-AI · 97.0 | 69.5 ± 4.2 |
| Router-only · 97.0 | 41.0 ± 5.7 |
| Cascade-only · 97.0 | 31.1 ± 8.2 |
| FRUGAL-AI minus Router-only · 97.0 | 28.5 ± 7.8 |
| FRUGAL-AI · 98.0 | 53.5 ± 9.1 |
| Router-only · 98.0 | 32.4 ± 4.8 |
| Cascade-only · 98.0 | 12.0 ± 5.9 |
| FRUGAL-AI minus Router-only · 98.0 | 21.1 ± 11.7 |
| FRUGAL-AI · 98.5 | 41.1 ± 8.7 |
| Router-only · 98.5 | 28.8 ± 4.6 |
| Cascade-only · 98.5 | 7.4 ± 3.9 |
| FRUGAL-AI minus Router-only · 98.5 | 12.3 ± 11.7 |
| FRUGAL-AI · 99.0 | 28.9 ± 5.8 |
| Router-only · 99.0 | 24.7 ± 4.7 |
| Cascade-only · 99.0 | 4.8 ± 2.7 |
| FRUGAL-AI minus Router-only · 99.0 | 4.2 ± 9.8 |

## Cost-quality frontier (distinct queries; method · tolerance)

| Method | Cost reduction (%) | Quality retained (%) |
| --- | --- | --- |
| FRUGAL-AI · 0.1 | 91.2 ± 0.6 | 92.8 ± 0.5 |
| Router-only · 0.1 | 92.3 ± 1.8 | 92.8 ± 0.5 |
| Cascade-only · 0.1 | 90.7 ± 0.8 | 92.9 ± 0.5 |
| FRUGAL-AI · 0.08 | 89.1 ± 1.2 | 93.4 ± 0.7 |
| Router-only · 0.08 | 79.0 ± 7.9 | 94.0 ± 0.8 |
| Cascade-only · 0.08 | 75.9 ± 3.8 | 93.9 ± 0.4 |
| FRUGAL-AI · 0.06 | 81.0 ± 2.7 | 95.2 ± 0.5 |
| Router-only · 0.06 | 59.2 ± 9.3 | 95.6 ± 0.7 |
| Cascade-only · 0.06 | 64.4 ± 1.6 | 95.3 ± 0.4 |
| FRUGAL-AI · 0.05 | 76.2 ± 3.4 | 96.1 ± 0.7 |
| Router-only · 0.05 | 46.3 ± 5.8 | 96.7 ± 0.6 |
| Cascade-only · 0.05 | 50.2 ± 5.7 | 95.8 ± 0.6 |
| FRUGAL-AI · 0.04 | 72.7 ± 3.3 | 96.9 ± 0.6 |
| Router-only · 0.04 | 36.4 ± 6.3 | 97.6 ± 0.8 |
| Cascade-only · 0.04 | 36.3 ± 5.6 | 96.5 ± 0.7 |
| FRUGAL-AI · 0.03 | 63.7 ± 5.2 | 97.5 ± 0.4 |
| Router-only · 0.03 | 27.8 ± 4.7 | 98.5 ± 0.8 |
| Cascade-only · 0.03 | 21.5 ± 7.3 | 97.1 ± 0.6 |
| FRUGAL-AI · 0.025 | 51.8 ± 8.6 | 97.9 ± 0.4 |
| Router-only · 0.025 | 24.5 ± 4.9 | 98.8 ± 0.7 |
| Cascade-only · 0.025 | 13.2 ± 8.1 | 97.6 ± 0.5 |
| FRUGAL-AI · 0.02 | 40.2 ± 8.2 | 98.4 ± 0.4 |
| Router-only · 0.02 | 21.1 ± 4.8 | 98.9 ± 0.6 |
| Cascade-only · 0.02 | 6.7 ± 8.9 | 98.0 ± 0.5 |
| FRUGAL-AI · 0.015 | 33.1 ± 7.4 | 98.8 ± 0.5 |
| Router-only · 0.015 | 17.8 ± 3.2 | 99.3 ± 0.4 |
| Cascade-only · 0.015 | -3.5 ± 8.6 | 98.3 ± 0.6 |
| FRUGAL-AI · 0.01 | 25.8 ± 6.7 | 99.1 ± 0.4 |
| Router-only · 0.01 | 15.5 ± 3.8 | 99.4 ± 0.2 |
| Cascade-only · 0.01 | -21.8 ± 10.2 | 99.1 ± 0.5 |
| FRUGAL-AI · 0.0075 | 17.1 ± 6.6 | 99.1 ± 0.5 |
| Router-only · 0.0075 | 11.8 ± 4.1 | 99.5 ± 0.2 |
| Cascade-only · 0.0075 | -23.8 ± 10.6 | 99.1 ± 0.5 |
| FRUGAL-AI · 0.005 | 10.9 ± 8.4 | 99.5 ± 0.4 |
| Router-only · 0.005 | 3.7 ± 4.1 | 99.8 ± 0.2 |
| Cascade-only · 0.005 | -39.6 ± 5.1 | 99.6 ± 0.2 |
| FRUGAL-AI · 0.0025 | 9.0 ± 8.4 | 99.5 ± 0.4 |
| Router-only · 0.0025 | 3.6 ± 4.0 | 99.8 ± 0.2 |
| Cascade-only · 0.0025 | -40.8 ± 5.5 | 99.6 ± 0.2 |
| FRUGAL-AI · 0.0 | 5.9 ± 6.6 | 99.5 ± 0.4 |
| Router-only · 0.0 | 3.5 ± 3.9 | 99.8 ± 0.2 |
| Cascade-only · 0.0 | -44.9 ± 5.4 | 99.7 ± 0.2 |

## Sensitivity (replay stream)

| Setting | Cost reduction (%) | Quality retained (%) | Cache hit rate (%) | False-hit rate (%) |
| --- | --- | --- | --- | --- |
| tau_cache=0.8 | 52.6 ± 4.9 | 97.6 ± 0.9 | 30.9 ± 0.4 | 1.53 ± 0.53 |
| tau_cache=0.85 | 51.3 ± 5.2 | 98.5 ± 0.9 | 29.8 ± 0.2 | 0.62 ± 0.27 |
| tau_cache=0.9 | 50.6 ± 5.4 | 98.7 ± 0.9 | 28.5 ± 0.2 | 0.36 ± 0.27 |
| tau_cache=0.95 | 50.1 ± 5.4 | 99.0 ± 0.8 | 26.8 ± 0.4 | 0.16 ± 0.11 |
| online controller, rho=0.005 | 51.5 ± 6.1 | 98.5 ± 0.7 | 28.5 ± 1.8 | 0.60 ± 0.40 |
| online controller, rho=0.02 | 48.8 ± 4.9 | 98.5 ± 0.7 | 28.5 ± 1.8 | 0.60 ± 0.40 |
| online controller, rho=0.05 | 46.8 ± 4.8 | 98.5 ± 0.7 | 28.5 ± 1.8 | 0.60 ± 0.40 |
| online controller, eta=0.01 | 48.6 ± 4.9 | 98.5 ± 0.7 | 28.5 ± 1.8 | 0.60 ± 0.40 |
| online controller, eta=0.2 | 49.9 ± 5.6 | 98.4 ± 0.8 | 28.5 ± 1.8 | 0.60 ± 0.40 |
| online controller, eta=0.5 | 52.6 ± 8.5 | 97.6 ± 1.3 | 28.5 ± 1.8 | 0.60 ± 0.40 |
| repeat_share=0.0 | 33.4 ± 7.2 | 98.2 ± 0.5 | 0.7 ± 0.5 | 0.70 ± 0.47 |
| repeat_share=0.15 | 43.2 ± 6.5 | 98.3 ± 0.6 | 14.4 ± 1.3 | 0.67 ± 0.43 |
| repeat_share=0.3 | 53.6 ± 6.7 | 98.5 ± 0.6 | 28.3 ± 2.3 | 0.63 ± 0.53 |
| repeat_share=0.5 | 66.9 ± 5.9 | 98.6 ± 0.5 | 47.7 ± 2.6 | 0.45 ± 0.34 |

## Drift: easy share falls from 80% to 50%

| Method · phase | Cost reduction (%) | Quality retained (%) |
| --- | --- | --- |
| FRUGAL-AI (fixed multiplier) · before | 32.8 ± 7.5 | 98.8 ± 0.5 |
| FRUGAL-AI (fixed multiplier) · after | 21.9 ± 12.1 | 95.6 ± 1.5 |
| FRUGAL-AI + online quality controller · before | 32.1 ± 7.4 | 98.7 ± 0.5 |
| FRUGAL-AI + online quality controller · after | 20.5 ± 11.9 | 95.4 ± 1.5 |
| Cascade-only (fixed) · before | 6.7 ± 8.9 | 97.9 ± 0.6 |
| Cascade-only (fixed) · after | -12.3 ± 8.4 | 93.7 ± 1.7 |

| Method | Windows > 1 pt below target (%) | Worst window quality retained (%) |
| --- | --- | --- |
| FRUGAL-AI (fixed multiplier) · windows | 40.5 ± 14.1 | 93.8 ± 1.7 |
| FRUGAL-AI + online quality controller · windows | 42.0 ± 14.7 | 93.6 ± 1.8 |
| Cascade-only (fixed) · windows | 55.0 ± 11.8 | 92.0 ± 2.0 |

## Budget control after drift

| Policy | Windows over budget (%) | Worst overshoot (%) | Quality retained (%) |
| --- | --- | --- | --- |
| fixed | 86.0 ± 21.1 | 68.5 ± 27.6 | 95.6 ± 1.5 |
| online budget control | 9.0 ± 10.4 | -2.0 ± 5.0 | 91.5 ± 2.3 |

## Tuned parameters and overhead

- tau_cache: 0.890 ± 0.045
- lambda0: 23.200 ± 11.509
- alpha: 0.055 ± 0.014
- tau_small: 0.655 ± 0.096
- tau_mid: 0.449 ± 0.228
- embed: 4.391 ± 0.000
- route: 0.004 ± 0.000
- verifier_call: 3693.548 ± 0.000

## Significance (paired t-tests across seeds)

Cost per 1k requests on the replay stream, target 98%:

| FRUGAL-AI vs. | Mean cost difference (USD / 1k) | Mean quality-retained difference (points) | p-value (cost) |
| --- | --- | --- | --- |
| Cache-only | -5.67 | -0.9 | 0.0000 |
| Compression-only | -11.15 | +0.9 | 0.0000 |
| Router-only | -6.99 | -0.2 | 0.0000 |
| Cascade-only | -10.20 | +0.3 | 0.0001 |

Cost reduction at matched quality, FRUGAL-AI minus Router-only (distinct queries):

| Quality retained (%) | Difference (points) | p-value |
| --- | --- | --- |
| 97 | +28.5 | 0.0000 |
| 98 | +21.1 | 0.0027 |
| 98.5 | +12.3 | 0.0413 |
| 99 | +4.2 | 0.3616 |
