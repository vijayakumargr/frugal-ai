"""Build the query set: five public workloads sampled into one JSONL file.

Usage: python -m frugal_ai.data [--set full|pilot|probe]
"""
import argparse
import json
import random
import re

import numpy as np
import requests

from .config import DATA_DIR, DATA_SEED, EMBED_MODEL, KEEP_RATIO, PILOT_SIZES, PROBE_SIZES, SIZES

MTBENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/"
LETTERS = "ABCDEFGHIJ"


def _mmlu(n, rng):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    out = []
    for i in rng.sample(range(len(ds)), n):
        r = ds[i]
        opts = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(r["choices"]))
        prompt = (
            "Answer the following multiple-choice question.\n\n"
            f"{r['question']}\n{opts}\n\n"
            "End your reply with a final line of the form 'ANSWER: X' where X is A, B, C or D."
        )
        out.append({"qid": f"mmlu-{i}", "workload": "mmlu", "turns": [prompt],
                    "gold": LETTERS[r["answer"]], "meta": {"subject": r["subject"]}})
    return out


def _gsm8k(n, rng):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i in rng.sample(range(len(ds)), n):
        r = ds[i]
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        prompt = (
            "Solve the following math problem.\n\n"
            f"{r['question']}\n\n"
            "End your reply with a final line of the form 'ANSWER: <number>'."
        )
        out.append({"qid": f"gsm8k-{i}", "workload": "gsm8k", "turns": [prompt], "gold": gold, "meta": {}})
    return out


def _humaneval(n, rng):
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    out = []
    for i in sorted(rng.sample(range(len(ds)), min(n, len(ds)))):
        r = ds[i]
        prompt = (
            "Complete the following Python function. Reply with the complete function, "
            "including its signature and any imports it needs, in a single ```python code block.\n\n"
            f"{r['prompt']}"
        )
        out.append({"qid": f"humaneval-{i}", "workload": "humaneval", "turns": [prompt], "gold": None,
                    "meta": {"prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"]}})
    return out


def _compress(article, encoder):
    """Keep the most salient sentences (similarity to the document centroid), in order."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", article.strip()) if s]
    if len(sents) < 4:
        return article
    emb = encoder.encode(sents, normalize_embeddings=True)
    salience = emb @ emb.mean(axis=0)
    budget = KEEP_RATIO * sum(len(s.split()) for s in sents)
    keep, used = set(), 0
    for j in np.argsort(-salience):
        if used >= budget:
            break
        keep.add(int(j))
        used += len(sents[j].split())
    return " ".join(s for j, s in enumerate(sents) if j in keep)


def _cnndm(n, rng):
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(EMBED_MODEL)
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", streaming=True)
    ds = ds.shuffle(seed=DATA_SEED, buffer_size=2000)
    head = "Summarize the following news article in three to four sentences.\n\nArticle:\n"
    out = []
    for r in ds:
        if len(out) >= n:
            break
        out.append({"qid": f"cnndm-{r['id'][:12]}", "workload": "cnndm", "turns": [head + r["article"]],
                    "turns_compressed": [head + _compress(r["article"], encoder)],
                    "gold": r["highlights"], "meta": {"article": r["article"]}})
    return out


def _fetch_jsonl(name):
    r = requests.get(MTBENCH_URL + name, timeout=60)
    r.raise_for_status()
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


def _mtbench(n, rng):
    questions = _fetch_jsonl("question.jsonl")
    refs = {r["question_id"]: r["choices"][0]["turns"] for r in _fetch_jsonl("reference_answer/gpt-4.jsonl")}
    out = []
    for r in sorted(rng.sample(questions, min(n, len(questions))), key=lambda r: r["question_id"]):
        out.append({"qid": f"mtbench-{r['question_id']}", "workload": "mtbench", "turns": r["turns"], "gold": None,
                    "meta": {"category": r["category"], "reference": refs.get(r["question_id"])}})
    return out


def _aime(n, rng):
    from datasets import load_dataset

    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    out = []
    for i in sorted(rng.sample(range(len(ds)), min(n, len(ds)))):
        r = ds[i]
        prompt = (
            "Solve the following math problem.\n\n"
            f"{r['problem']}\n\n"
            "End your reply with a final line of the form 'ANSWER: <integer>'."
        )
        out.append({"qid": f"aime-{i}", "workload": "aime", "turns": [prompt], "gold": str(r["answer"]).strip(),
                    "meta": {}})
    return out


def _mmlupro(n, rng):
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    out = []
    for i in rng.sample(range(len(ds)), n):
        r = ds[i]
        opts = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(r["options"]))
        prompt = (
            "Answer the following multiple-choice question.\n\n"
            f"{r['question']}\n{opts}\n\n"
            f"End your reply with a final line of the form 'ANSWER: X' where X is one of A to {LETTERS[len(r['options']) - 1]}."
        )
        out.append({"qid": f"mmlupro-{r['question_id']}", "workload": "mmlupro", "turns": [prompt],
                    "gold": r["answer"], "meta": {"category": r["category"]}})
    return out


def _codecontests(n, rng, min_rating=0, max_tests=15):
    """Codeforces problems; scored on public, private and some generated tests."""
    from datasets import load_dataset

    ds = load_dataset("deepmind/code_contests", split="test", streaming=True)
    hard = [r for r in ds if r["cf_rating"] >= min_rating]
    out = []
    for r in sorted(rng.sample(hard, min(n, len(hard))), key=lambda r: r["name"]):
        tests = []
        for part in ("public_tests", "private_tests", "generated_tests"):
            tests += [[i, o] for i, o in zip(r[part]["input"], r[part]["output"]) if len(i) < 200_000]
        prompt = (
            "Solve the following competitive programming problem in Python 3. The program must read from "
            "standard input and write to standard output. Reply with the complete program in a single "
            f"```python code block.\n\n{r['description']}"
        )
        out.append({"qid": f"cc-{r['name'].split('.')[0].replace(' ', '')}", "workload": "codecontests",
                    "turns": [prompt], "gold": None,
                    "meta": {"tests": tests[:max_tests], "rating": r["cf_rating"]}})
    return out


LOADERS = {"mmlu": _mmlu, "gsm8k": _gsm8k, "humaneval": _humaneval, "cnndm": _cnndm, "mtbench": _mtbench,
           "aime": _aime, "mmlupro": _mmlupro, "codecontests": _codecontests}
SETS = {"full": ("queries.jsonl", SIZES), "pilot": ("queries_pilot.jsonl", PILOT_SIZES),
        "probe": ("queries_probe.jsonl", PROBE_SIZES)}


def queries_path(name):
    return DATA_DIR / SETS[name][0]


def load_queries(name):
    with open(queries_path(name)) as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), default="full")
    args = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    queries = []
    for w, n in SETS[args.set][1].items():
        # The probe drew its Codeforces sample from problems rated 1600 and above.
        kw = {"min_rating": 1600} if (args.set, w) == ("probe", "codecontests") else {}
        rows = LOADERS[w](n, random.Random(DATA_SEED), **kw)
        print(f"{w}: {len(rows)} queries")
        queries += rows
    with open(queries_path(args.set), "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    print(f"wrote {len(queries)} queries to {queries_path(args.set)}")


if __name__ == "__main__":
    main()
