"""Run every pool model on every query and log cost, latency and quality.

Usage: python -m frugal_ai.collect [--pilot] [--mock] [--max-usd 60]

Per (query, model, prompt variant) this makes one generation call per turn,
one cheap verifier call (small and mid models only) and, for summaries and
dialogue, one judge call. The simulator replays these logged outcomes.
"""
import argparse
import asyncio
import hashlib
import json
import datetime
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rouge_score import rouge_scorer

from .config import (GEN_EFFORT, JUDGE_EFFORT, JUDGE_MODEL, K, MODELS, PARAPHRASE_MODEL, VERIFIER_EFFORT,
                     VERIFIER_MODEL, results_dir)
from .data import load_queries
from .llm import LLM, BudgetExceeded, UsageLimit

ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
PARAPHRASED = {"mmlu", "gsm8k", "mmlupro", "aime", "mtbench"}
LETTER = {"mmlu", "mmlupro"}
NUMERIC = {"gsm8k", "aime"}
LETTERS = "ABCDEFGHIJ"
JUDGED = {"cnndm", "mtbench"}
MOCK_SKILL = [0.55, 0.80, 0.92]


def _unit(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def mock_quality(q, k, variant):
    """Mock mode only: a made-up quality so the pipeline can run offline."""
    d = 0.85 * _unit(q["qid"]) + 0.15 * _unit(f"{q['qid']}|{k}") + 0.03 * variant
    if q["workload"] in JUDGED:
        return round(max(0.1, min(1.0, MOCK_SKILL[k] + 0.25 - 0.4 * d)), 2)
    return float(d < MOCK_SKILL[k])


def mock_answer(q, k, variant):
    mq, w = mock_quality(q, k, variant), q["workload"]
    if w in LETTER:
        return f"ANSWER: {q['gold'] if mq else LETTERS[(LETTERS.index(q['gold']) + 1) % 4]}"
    if w in NUMERIC:
        return f"ANSWER: {q['gold'] if mq else float(q['gold']) + 1}"
    if w in ("humaneval", "codecontests"):
        return "MOCK_PASS" if mq else "MOCK_FAIL"
    return f"MOCK_Q={mq} mock text"


def run_humaneval(q, text):
    if text.startswith("MOCK_"):
        return float(text == "MOCK_PASS")
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    code = blocks[-1] if blocks else text
    m = q["meta"]
    if f"def {m['entry_point']}" not in code:
        code = m["prompt"] + code
    program = f"{code}\n\n{m['test']}\n\ncheck({m['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prog.py"
        path.write_text(program)
        try:
            r = subprocess.run([sys.executable, "-I", str(path)], cwd=d, capture_output=True, timeout=20)
            return float(r.returncode == 0)
        except subprocess.TimeoutExpired:
            return 0.0


def run_stdio_tests(q, text):
    """Pass only if the program prints the expected output on every test."""
    if text.startswith("MOCK_"):
        return float(text == "MOCK_PASS")
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    if not blocks:
        return 0.0
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prog.py"
        path.write_text(blocks[-1])
        for stdin, expected in q["meta"]["tests"]:
            try:
                r = subprocess.run([sys.executable, "-I", str(path)], cwd=d, input=stdin.encode(),
                                   capture_output=True, timeout=10)
            except subprocess.TimeoutExpired:
                return 0.0
            if r.returncode != 0 or r.stdout.decode(errors="replace").split() != expected.split():
                return 0.0
    return 1.0


def score_exact(q, text):
    """Return (quality, answer_key) for workloads with a gold answer."""
    if q["workload"] in LETTER:
        found = re.findall(r"ANSWER:\s*\(?([A-J])\b", text)
        key = found[-1] if found else None
        return float(key == q["gold"]), key
    found = re.findall(r"ANSWER:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not found:
        return 0.0, None
    key = found[-1].replace(",", "")
    try:
        return float(abs(float(key) - float(q["gold"])) < 1e-6), key
    except ValueError:
        return 0.0, None


def request_text(q):
    return "\n\n".join(f"[Turn {i + 1}] {t}" for i, t in enumerate(q["turns"])) if len(q["turns"]) > 1 else q["turns"][0]


def answer_text(answers):
    return "\n\n".join(f"[Answer {i + 1}] {a}" for i, a in enumerate(answers)) if len(answers) > 1 else answers[0]


def verifier_prompt(q, answers):
    return (
        "You are checking an AI assistant's answer.\n\n"
        f"<request>\n{request_text(q)}\n</request>\n\n<answer>\n{answer_text(answers)}\n</answer>\n\n"
        "How likely is it that this answer is correct and fully satisfies the request? "
        "Reply with a single line of the form 'CONFIDENCE: <integer 0-100>'."
    )


def judge_prompt(q, answers):
    if q["workload"] == "cnndm":
        return (
            "Rate the candidate summary of the news article below on a scale of 1 to 10. Judge faithfulness to "
            "the article, coverage of the key points in the reference summary, and conciseness.\n\n"
            f"<article>\n{q['meta']['article']}\n</article>\n\n<reference_summary>\n{q['gold']}\n</reference_summary>\n\n"
            f"<candidate_summary>\n{answers[0]}\n</candidate_summary>\n\n"
            "Give one or two sentences of reasoning, then a final line of the form 'SCORE: <1-10>'."
        )
    ref = q["meta"].get("reference")
    ref_part = f"<reference_answers>\n{answer_text(ref)}\n</reference_answers>\n\n" if ref else ""
    return (
        "Rate the assistant's answers in the conversation below on a scale of 1 to 10. Judge helpfulness, "
        "relevance, accuracy, depth and level of detail. If reference answers are given, use them to judge "
        "correctness.\n\n"
        f"<user_turns>\n{request_text(q)}\n</user_turns>\n\n{ref_part}"
        f"<assistant_answers>\n{answer_text(answers)}\n</assistant_answers>\n\n"
        "Give one or two sentences of reasoning, then a final line of the form 'SCORE: <1-10>'."
    )


async def outcome(llm, q, k, variant):
    """Generate, score and verify one (query, model, prompt variant)."""
    model = MODELS[k]["id"]
    turns = q["turns_compressed"] if variant else q["turns"]
    mock = lambda: mock_answer(q, k, variant)
    messages, answers, calls = [], [], []
    for t, turn in enumerate(turns):
        messages.append({"role": "user", "content": turn})
        rec = await llm.call("gen", f"{q['qid']}|v{variant}|t{t}", model, messages, GEN_EFFORT, mock,
                             resume=calls[-1].get("session_id") if calls else None, persist=len(turns) > 1)
        calls.append(rec)
        answers.append(rec["text"])
        # An empty turn (refusal or error) ends the conversation; it scores 0.
        if not rec["text"]:
            break
        messages.append({"role": "assistant", "content": rec["text"]})
    row = {
        "qid": q["qid"], "workload": q["workload"], "model": k, "variant": variant,
        "cost": sum(r["cost"] for r in calls), "latency": sum(r["latency"] for r in calls),
        "input_tokens": sum(r["input_tokens"] for r in calls),
        "output_tokens": sum(r["output_tokens"] for r in calls),
        "stop_reason": calls[-1]["stop_reason"], "answer_key": None, "rouge_l": None,
        "judge_cost": 0.0, "judge_parse_failed": False, "ver_conf": None, "ver_cost": 0.0, "ver_latency": 0.0,
    }
    complete = len(answers) == len(turns) and all(answers)
    w = q["workload"]
    if not complete:
        row["quality"] = 0.0
    elif w in LETTER | NUMERIC:
        row["quality"], row["answer_key"] = score_exact(q, answers[0])
    elif w == "humaneval":
        row["quality"] = await asyncio.to_thread(run_humaneval, q, answers[0])
    elif w == "codecontests":
        row["quality"] = await asyncio.to_thread(run_stdio_tests, q, answers[0])
    else:
        mq = mock_quality(q, k, variant)
        rec = await llm.call("judge", f"{q['qid']}|v{variant}|m{k}", JUDGE_MODEL,
                             [{"role": "user", "content": judge_prompt(q, answers)}], JUDGE_EFFORT,
                             lambda: f"SCORE: {round(10 * mq)}")
        found = re.findall(r"SCORE:\s*(\d+(?:\.\d+)?)", rec["text"])
        row["judge_cost"] = rec["cost"]
        row["judge_parse_failed"] = not found
        row["quality"] = min(1.0, float(found[-1]) / 10) if found else 0.5
        if w == "cnndm":
            row["rouge_l"] = ROUGE.score(q["gold"], answers[0])["rougeL"].fmeasure
    if k < K - 1 and complete:
        mq = mock_quality(q, k, variant)
        conf = lambda: f"CONFIDENCE: {round(max(0, min(100, 100 * (0.2 + 0.6 * mq) + random.gauss(0, 15))))}"
        rec = await llm.call("ver", f"{q['qid']}|v{variant}|m{k}", VERIFIER_MODEL,
                             [{"role": "user", "content": verifier_prompt(q, answers)}], VERIFIER_EFFORT, conf)
        found = re.findall(r"CONFIDENCE:\s*(\d+)", rec["text"])
        row["ver_conf"] = min(100, int(found[-1])) / 100 if found else 0.5
        row["ver_cost"], row["ver_latency"] = rec["cost"], rec["latency"]
    elif k < K - 1:
        row["ver_conf"] = 0.0
    return row


async def paraphrase(llm, q):
    prompt = (
        "Rewrite the user request below in different words. Keep its exact meaning and keep every number, "
        "name, answer option and formatting instruction unchanged. Output only the rewritten request.\n\n"
        f"<request>\n{q['turns'][0]}\n</request>"
    )
    rec = await llm.call("para", q["qid"], PARAPHRASE_MODEL, [{"role": "user", "content": prompt}], "low",
                         lambda: "Reworded: " + q["turns"][0])
    return q["qid"], rec["text"].strip() or None


async def run(args):
    """One collection attempt. Returns "done", "budget" or a usage-limit message."""
    backend = "mock" if args.mock else args.backend
    out_dir = results_dir(backend + (args.tag or ("_probe" if args.probe else "")))
    queries = load_queries("probe" if args.probe else "pilot" if args.pilot else "full")
    llm = LLM(out_dir, backend=backend, max_usd=args.max_usd, concurrency=args.concurrency)
    combos = [(q, k, v) for q in queries for k in range(K) for v in ((0, 1) if "turns_compressed" in q else (0,))]
    if args.dry_run:
        need = [f"gen|{MODELS[k]['id']}|{q['qid']}|v{v}|t{t}" for q, k, v in combos for t in range(len(q["turns"]))]
        have = sum(c in llm.calls for c in need)
        print(f"{len(queries)} queries; answer calls already saved: {have} of {len(need)}; "
              f"logged spend ${llm.spent:.2f}, cap ${args.max_usd:.2f}")
        return "done"
    jobs = [outcome(llm, q, k, v) for q, k, v in combos]
    para_jobs = [paraphrase(llm, q) for q in queries if q["workload"] in PARAPHRASED]
    print(f"{len(queries)} queries, {len(jobs)} outcomes, {len(para_jobs)} paraphrases; "
          f"already spent ${llm.spent:.2f}, cap ${args.max_usd:.2f}", flush=True)
    results = await asyncio.gather(*jobs, *para_jobs, return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    print("spend by call type:", {k: round(v, 3) for k, v in sorted(llm.spend_by_kind().items())})
    print(f"total spend ${llm.spent:.2f} (this run ${llm.new_spend:.2f}); at list prices, notional with --backend cli")
    if errors:
        other = [e for e in errors if not isinstance(e, (BudgetExceeded, UsageLimit))]
        print(f"INCOMPLETE: {len(errors)} of {len(results)} jobs unfinished: {errors[0]}")
        print("Finished calls are saved; re-run the same command to resume.", flush=True)
        if other:
            raise other[0]
        return llm.limit_msg or "budget"
    rows = results[: len(jobs)]
    with open(out_dir / "outcomes.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out_dir / "paraphrases.json").write_text(json.dumps(dict(results[len(jobs):]), indent=0))
    print(f"wrote {len(rows)} outcomes to {out_dir / 'outcomes.jsonl'}")
    for k in range(K):
        mine = [r for r in rows if r["model"] == k and r["variant"] == 0]
        print(f"  {MODELS[k]['name']:5s} quality {sum(r['quality'] for r in mine) / len(mine):.3f}  "
              f"cost/1k ${1000 * sum(r['cost'] for r in mine) / len(mine):.2f}  "
              f"refusals {sum(r['stop_reason'] == 'refusal' for r in mine)}  "
              f"truncated {sum(r['stop_reason'] == 'max_tokens' for r in mine)}  "
              f"timeouts {sum(r['stop_reason'] == 'timeout' for r in mine)}")
    print("  judge parse failures:", sum(r["judge_parse_failed"] for r in rows), flush=True)
    return "done"


def seconds_until_reset(message, now=None, poll=1200):
    """Seconds to wait after a usage-limit message such as 'resets 2:30am'.

    Falls back to polling every 20 minutes when no time of day can be read (a
    retry while the limit is still in place uses no tokens).
    """
    now = now or datetime.datetime.now()
    m = re.search(r"resets\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)", message, re.I)
    if not m:
        return poll
    hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
    target = now.replace(hour=hour, minute=int(m.group(2) or 0), second=0, microsecond=0)
    target += datetime.timedelta(minutes=3)
    if target <= now:
        target += datetime.timedelta(days=1)
    wait = (target - now).total_seconds()
    # A 5-hour window cannot reset later than this; anything else is polled.
    return wait if wait <= 5.25 * 3600 else poll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="first pilot set: 20 queries from each original workload")
    ap.add_argument("--probe", action="store_true", help="20 queries from each harder candidate workload")
    ap.add_argument("--mock", action="store_true", help="no model calls; made-up outcomes in results_mock/")
    ap.add_argument("--backend", choices=["api", "cli"], default="api",
                    help="api: Anthropic SDK with an API key; cli: `claude -p` on the Claude Code login")
    ap.add_argument("--max-usd", type=float, default=60.0, help="hard cap on total spend at list prices")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for the results directory")
    ap.add_argument("--dry-run", action="store_true", help="count saved and missing calls; call nothing")
    ap.add_argument("--until-done", action="store_true",
                    help="after a usage limit, wait for the reset and resume, for up to 7 days")
    args = ap.parse_args()
    deadline = time.time() + 7 * 86400
    while True:
        status = asyncio.run(run(args))
        if status in ("done", "budget") or not args.until_done or time.time() > deadline:
            break
        wait = seconds_until_reset(status)
        print(f"[{datetime.datetime.now():%b %d %H:%M}] usage limit: {status!r}; retrying in {wait / 60:.0f} min", flush=True)
        time.sleep(wait)
    print(f"finished with status: {status}", flush=True)


if __name__ == "__main__":
    main()
