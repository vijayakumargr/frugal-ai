"""Shared settings for the FRUGAL-AI experiments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# Model pool, cheapest first. Prices are USD per 1M tokens (Anthropic list
# prices, cached 2026-10-06); the last entry is the reference model m_K.
MODELS = [
    {"name": "small", "id": "claude-haiku-5-5", "p_in": 0.10, "p_out": 0.50},
    {"name": "mid", "id": "claude-sonnet-5-5", "p_in": 2.00, "p_out": 10.00},
    {"name": "large", "id": "claude-opus-5-5", "p_in": 4.00, "p_out": 20.00},
]
K = len(MODELS)
PRICE = {m["id"]: (m["p_in"], m["p_out"]) for m in MODELS}

GEN_EFFORT = "medium"  # same effort for every pool model
VERIFIER_MODEL = MODELS[0]["id"]  # cheap judge used inside the cascade
VERIFIER_EFFORT = "low"
JUDGE_MODEL = MODELS[-1]["id"]  # offline quality judge (summaries, dialogue)
JUDGE_EFFORT = "medium"
PARAPHRASE_MODEL = MODELS[0]["id"]
MAX_TOKENS = 8000

# Queries sampled per workload. The full mix uses the harder benchmarks: the
# pilot showed MMLU, GSM8K and HumanEval do not separate the three models.
SIZES = {"mmlupro": 400, "codecontests": 165, "aime": 90, "cnndm": 150, "mtbench": 80}
PILOT_SIZES = {"mmlu": 20, "gsm8k": 20, "humaneval": 20, "cnndm": 20, "mtbench": 20}
PROBE_SIZES = {"aime": 20, "mmlupro": 20, "codecontests": 20}
DATA_SEED = 0
KEEP_RATIO = 0.5  # prompt compression keep-ratio r for long documents
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def results_dir(backend: str) -> Path:
    # One directory per backend (api, cli, mock) so their outputs never mix.
    d = ROOT / ("results" if backend == "api" else f"results_{backend}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = PRICE[model_id]
    return (input_tokens * p_in + output_tokens * p_out) / 1e6
