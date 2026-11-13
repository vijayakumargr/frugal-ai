"""Async Claude client with an on-disk call log, a spend cap and a mock mode."""
import asyncio
import json
import os
import random
import shutil
import time

import anthropic

from .config import MAX_TOKENS, ROOT, cost_usd


class BudgetExceeded(RuntimeError):
    pass


class UsageLimit(RuntimeError):
    """The Claude Code plan's usage limit was hit; the message names the reset time."""


CALL_TIMEOUT = float(os.environ.get("FRUGAL_CALL_TIMEOUT", 1200))  # seconds per claude -p call


def load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


class LLM:
    """Every call is appended to calls.jsonl, so a re-run never pays twice."""

    def __init__(self, out_dir, backend="api", max_usd=60.0, concurrency=8):
        self.backend = backend
        self.cwd = out_dir / "cwd"
        self.max_usd = max_usd
        self.sem = asyncio.Semaphore(concurrency)
        self.log_path = out_dir / "calls.jsonl"
        self.calls = {}
        if self.log_path.exists():
            with open(self.log_path) as f:
                for line in f:
                    r = json.loads(line)
                    self.calls[r["id"]] = r
        self.spent = sum(r["cost"] for r in self.calls.values())
        self.new_spend = 0.0
        self._calib = {}
        self.limit_msg = None
        if backend == "api":
            load_dotenv()
            self.client = anthropic.AsyncAnthropic(max_retries=5, timeout=600.0)
        elif backend == "cli":
            assert shutil.which("claude"), "the claude command was not found"
            self.cwd.mkdir(exist_ok=True)

    async def call(self, kind, key, model, messages, effort, mock_text=None, resume=None, persist=False):
        """resume/persist are for the cli backend: continue or keep a multi-turn session."""
        cid = f"{kind}|{model}|{key}"
        if cid in self.calls:
            return self.calls[cid]
        if self.backend == "cli" and kind != "calib":
            overhead = await self._cli_overhead(model)
        async with self.sem:
            if self.spent >= self.max_usd:
                raise BudgetExceeded(f"spend cap ${self.max_usd:.2f} reached (${self.spent:.2f} spent)")
            if self.limit_msg:
                # Once the limit is hit, queued calls stop without starting a process.
                raise UsageLimit(self.limit_msg)
            if self.backend == "mock":
                rec = await self._mock(model, messages, mock_text)
            elif self.backend == "cli":
                rec = await self._cli(model, messages, effort, resume, persist)
                if kind != "calib":
                    rec["input_tokens"] = max(0, rec["input_tokens"] - overhead)
            else:
                rec = await self._real(model, messages, effort)
        rec.update(id=cid, kind=kind, model=model)
        rec["cost"] = cost_usd(model, rec["input_tokens"], rec["output_tokens"])
        self.calls[cid] = rec
        self.spent += rec["cost"]
        self.new_spend += rec["cost"]
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    async def _real(self, model, messages, effort):
        for attempt in range(6):
            t0 = time.perf_counter()
            try:
                # No refusal fallback on purpose: an answer from another model
                # would corrupt this model's cost and quality measurements.
                resp = await self.client.messages.create(
                    model=model, max_tokens=MAX_TOKENS, messages=messages,
                    output_config={"effort": effort},
                )
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError):
                raise
            except anthropic.BadRequestError as e:
                return {"text": "", "stop_reason": "bad_request", "error": e.message,
                        "input_tokens": 0, "output_tokens": 0, "latency": time.perf_counter() - t0}
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIStatusError) as e:
                if isinstance(e, anthropic.APIStatusError) and e.status_code < 500 and e.status_code != 429:
                    raise
                if attempt == 5:
                    raise
                await asyncio.sleep(min(60, 5 * 2 ** attempt))
                continue
            return {
                "text": "".join(b.text for b in resp.content if b.type == "text"),
                "stop_reason": resp.stop_reason,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "latency": time.perf_counter() - t0,
            }

    async def _cli_overhead(self, model):
        """Input tokens Claude Code adds to every call, measured once per model."""
        if model not in self._calib:
            # One shared task per model, so concurrent callers do not each pay for it.
            self._calib[model] = asyncio.ensure_future(
                self.call("calib", "overhead", model, [{"role": "user", "content": "."}], "low"))
        return (await self._calib[model])["input_tokens"] - 1

    async def _cli(self, model, messages, effort, resume, persist):
        """One turn through `claude -p`, billed to the Claude Code login."""
        cmd = ["claude", "-p", "--model", model, "--effort", effort, "--system-prompt", "", "--tools", "",
               "--output-format", "json", "--setting-sources", "", "--strict-mcp-config"]
        if resume:
            cmd += ["--resume", resume]
        elif not persist:
            cmd.append("--no-session-persistence")
        prompt = messages[-1]["content"]
        for attempt in range(3):
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=self.cwd, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), CALL_TIMEOUT)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                # Logged as an empty answer (quality 0); its token use is unknown.
                return {"text": "", "stop_reason": "timeout", "input_tokens": 0, "output_tokens": 0,
                        "latency": CALL_TIMEOUT, "session_id": None}
            try:
                d = json.loads(out)
            except json.JSONDecodeError:
                d = {"is_error": True, "result": (err or out).decode()[-500:]}
            if d.get("stop_reason") == "refusal" or not d.get("is_error"):
                break
            message = str(d.get("result"))
            if "limit" in message.lower():
                self.limit_msg = message[:300]
                raise UsageLimit(self.limit_msg)
            if attempt == 2:
                raise RuntimeError(f"claude -p failed: {message[:300]}")
            await asyncio.sleep(10 * (attempt + 1))
        if model not in d.get("modelUsage", {model: 0}):
            raise RuntimeError(f"asked for {model} but got {list(d['modelUsage'])}")
        u = d["usage"]
        return {
            "text": "" if d.get("stop_reason") == "refusal" else (d.get("result") or ""),
            "stop_reason": d.get("stop_reason"),
            # Cached and uncached input are all priced as plain input tokens.
            "input_tokens": u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"],
            "output_tokens": u["output_tokens"],
            "latency": d["duration_api_ms"] / 1000,
            "session_id": d.get("session_id"),
        }

    async def _mock(self, model, messages, mock_text):
        text = mock_text() if mock_text else "mock"
        n_in = sum(len(str(m["content"])) for m in messages) // 4
        speed = {"claude-haiku-5-5": 0.6, "claude-sonnet-5-5": 1.0, "claude-opus-5-5": 1.5}[model]
        return {"text": text, "stop_reason": "end_turn", "input_tokens": n_in,
                "output_tokens": random.randint(150, 600), "latency": speed * random.uniform(2, 8)}

    def spend_by_kind(self):
        out = {}
        for r in self.calls.values():
            out[r["kind"]] = out.get(r["kind"], 0.0) + r["cost"]
        return out
