# FRUGAL-AI

Code and logged model outputs for the paper *FRUGAL-AI: A Cost-Aware Adaptive
Framework for Reducing LLM Inference Cost*.

FRUGAL-AI is a serving pipeline in front of a pool of language models: a
semantic cache, a difficulty router and a verifier-gated cascade, with all
thresholds calibrated together against one quality target. The model pool in
the experiments is Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5.

Every model call was logged once. All tables in the paper are produced by
replaying those logs, so they can be reproduced without calling any model.

## Reproduce the paper's tables

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Rebuild the 885-query set from the public benchmarks (downloads them)
.venv/bin/python -m frugal_ai.data --set full

# 2. Replay the logged outcomes through every method (no model calls)
.venv/bin/python -m frugal_ai.simulate --backend cli
```

Step 2 writes `frugal_ai/results_cli/RESULTS.md` and `summary.json`. Apart from
the two timing lines (embedding and routing overhead) the output should match
the committed files exactly.

## What is in the repository

| Path | Contents |
| --- | --- |
| `frugal_ai/data.py` | Builds the query set: MMLU-Pro 400, Codeforces (CodeContests) 165, AIME 90, CNN/DailyMail 150, MT-Bench 80 |
| `frugal_ai/collect.py`, `llm.py` | Calls each model once per query, scores the answers and logs cost and latency |
| `frugal_ai/simulate.py` | Calibration, baselines, ablations and all reported experiments, over 10 seeds |
| `frugal_ai/results_cli/outcomes.jsonl` | One row per query, model and prompt variant: quality, cost, latency, verifier confidence |
| `frugal_ai/results_cli/calls.jsonl` | Every logged model call with the model's answer and token counts |
| `frugal_ai/results_cli/paraphrases.json`, `emb.npz` | Paraphrased questions and cached embeddings used by the cache experiments |
| `frugal_ai/results_cli/RESULTS.md`, `summary.json` | The results reported in the paper |
| `first_evaluation/` | The first evaluation and the code that produced it (see below) |
| `paper/` | The paper as PDF, with its LaTeX source and figures |

The benchmark questions themselves are not redistributed here; step 1 rebuilds
them from their original sources with a fixed seed.

## Collecting new outputs (optional)

Only needed to change the models or the query set. It calls the models and
costs money or plan usage.

```bash
# through the Claude Code command line (uses the Claude Code login)
.venv/bin/python -m frugal_ai.collect --backend cli --max-usd 80 --until-done

# through the Anthropic API (needs ANTHROPIC_API_KEY in frugal_ai/.env or the shell)
.venv/bin/python -m frugal_ai.collect --backend api --max-usd 80
```

Every call is appended to `calls.jsonl` in the results directory, so an
interrupted run resumes without repeating calls. `--mock` runs the pipeline
offline on made-up outcomes, for testing the code only.

## Notes on the method and the data

- Costs are logged tokens times list prices. With the `cli` backend the roughly
  550 input tokens that Claude Code adds to each call are measured once per
  model and subtracted.
- Codeforces scoring runs model-written Python in a subprocess, 10 s per test,
  and requires every test to pass. It is slightly strict: 2 of 13 reference
  solutions failed it.
- A call that runs past 20 minutes is logged as a timeout with quality 0.
  Refusals score 0. Neither occurred in the logged run.
- Seeds redraw the calibration/test split, the request stream and the audit
  sample over one set of logged outputs. They are not independent model samples.
- Tuning rule (all tuned methods): walk the calibration cost-quality front from
  the safest setting and stop at the first setting whose quality does not clear
  the target by one standard error. A quarter of the quality tolerance is
  reserved for wrong cache hits.
- `calls.jsonl` also holds the calls of a 100-query pilot (MMLU, GSM8K,
  HumanEval) and a 60-query probe that were used to choose the benchmarks.
  Session identifiers and local file paths were removed from it.

## First evaluation

The first evaluation tuned every method by taking the cheapest setting that met
the target on the calibration split. On the test split FRUGAL-AI then retained
97.7% of the reference quality against a 99% target. The tuning rule above
replaced it. The first results and the simulator that produced them are kept in
`first_evaluation/` so the change is visible.

## Citing

See `CITATION.cff`, or use the "Cite this repository" button on GitHub.

## Licence

MIT, see `LICENSE`. The benchmarks keep their own licences.
