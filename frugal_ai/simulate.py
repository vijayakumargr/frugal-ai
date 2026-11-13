"""Replay logged model outcomes through FRUGAL-AI, its baselines and ablations.

Usage: python -m frugal_ai.simulate --backend cli [--pilot] [--mock] [--seeds 10]

No API calls are made here. Each seed draws a new train/test split, request
stream and audit sample over the same logged outcomes from collect.py.
"""
import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge

from .config import EMBED_MODEL, K, MODELS, results_dir
from .data import load_queries

NF = 5  # cross-fitting folds
EPS = 0.02  # main quality tolerance: q* = (1 - EPS) x always-large quality
MAIN_EPS = (0.02, 0.01)  # tolerances reported in the main tables
Z_MARGIN = 1.0  # tuning keeps a candidate only if its quality clears the target by Z standard errors
CACHE_SHARE = 0.25  # share of the quality tolerance reserved for wrong cache hits
SWEEP_EPS = [0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.0025, 0.0]
LEVELS = [97.0, 98.0, 98.5, 99.0]  # quality-retained levels for the matched-quality comparison
RHO, ETA, ETA_MU = 0.02, 0.05, 0.05  # audit rate, dual step sizes
REPEAT_SHARE = 0.30  # share of repeated requests in the replay stream
ALPHAS = [0.003, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12]
LAMS = [0.0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128]
TAU_GRID = np.round(np.arange(0.0, 1.011, 0.01), 2)
NEVER = 1.01  # threshold that never accepts
L = K - 1  # index of the reference model


# ---------------------------------------------------------------- data

def load(backend, pilot):
    out = results_dir(backend)
    queries = load_queries("pilot" if pilot else "full")
    paras = json.loads((out / "paraphrases.json").read_text())
    n = len(queries)
    pos = {q["qid"]: i for i, q in enumerate(queries)}
    D = SimpleNamespace(n=n, out=out, queries=queries)
    for name in ("Q", "C", "LAT", "VS", "VC", "VL"):
        setattr(D, name, np.full((n, K, 2), np.nan))
    D.keys = {}
    with open(out / "outcomes.jsonl") as f:
        for line in f:
            r = json.loads(line)
            i, k, v = pos[r["qid"]], r["model"], r["variant"]
            D.Q[i, k, v], D.C[i, k, v], D.LAT[i, k, v] = r["quality"], r["cost"], r["latency"]
            if k < L:
                D.VS[i, k, v], D.VC[i, k, v], D.VL[i, k, v] = r["ver_conf"], r["ver_cost"], r["ver_latency"]
            D.keys[i, k, v] = r["answer_key"]
    assert not np.isnan(D.Q[:, :, 0]).any(), "outcomes.jsonl is incomplete; re-run collect"
    D.wnames = sorted({q["workload"] for q in queries})
    D.W = np.array([D.wnames.index(q["workload"]) for q in queries])
    D.has_v1 = ~np.isnan(D.Q[:, 0, 1])
    D.gold = [q["gold"] for q in queries]
    texts0 = [q["turns"][0] for q in queries]
    texts1 = [paras.get(q["qid"]) for q in queries]
    D.has_para = np.array([t is not None for t in texts1])
    emb_path = out / "emb.npz"
    if emb_path.exists() and np.load(emb_path)["E0"].shape[0] == n:
        z = np.load(emb_path)
        D.E0, D.E1, D.embed_time = z["E0"], z["E1"], float(z["embed_time"])
    else:
        from sentence_transformers import SentenceTransformer

        enc = SentenceTransformer(EMBED_MODEL)
        D.E0 = enc.encode(texts0, normalize_embeddings=True, batch_size=64)
        D.E1 = enc.encode([t or texts0[i] for i, t in enumerate(texts1)], normalize_embeddings=True, batch_size=64)
        t0 = time.perf_counter()
        for t in texts0[:50]:
            enc.encode([t], normalize_embeddings=True)
        D.embed_time = (time.perf_counter() - t0) / 50
        np.savez(emb_path, E0=D.E0, E1=D.E1, embed_time=D.embed_time)
    onehot = np.eye(len(D.wnames))[D.W]
    length = np.log1p([len(t.split()) for t in texts0])[:, None] / 10
    D.X = np.hstack([D.E0, onehot, length])
    D.onehot = onehot
    return D


def false_hit_quality(D, i, src, k, v):
    """Quality when query i is served the cached answer written for query src."""
    key, gold = D.keys.get((src, k, v)), D.gold[i]
    name = D.wnames[D.W[i]]
    if key is None or gold is None or D.W[i] != D.W[src] or name not in ("mmlu", "mmlupro", "gsm8k", "aime"):
        return 0.0
    if name in ("mmlu", "mmlupro"):
        return float(key == gold)
    return float(abs(float(key) - float(gold)) < 1e-6)


# ---------------------------------------------------------------- learned parts

def fit_soft(F, y):
    """Logistic regression on soft labels in [0, 1]."""
    y = np.clip(y, 0.02, 0.98)
    m = LogisticRegression(C=1.0, max_iter=3000)
    m.fit(np.vstack([F, F]), np.r_[np.ones(len(F)), np.zeros(len(F))], sample_weight=np.r_[y, 1 - y])
    return m


def cross_fit(F, y, qidx, is_train, fold, fitter, predictor):
    """Out-of-fold predictions for train rows, full-train predictions for test rows."""
    pred = np.zeros(len(y))
    tr = is_train[qidx]
    for f in range(NF):
        fit_m, held = tr & (fold[qidx] != f), tr & (fold[qidx] == f)
        if held.any():
            pred[held] = predictor(fitter(F[fit_m], y[fit_m]), F[held])
    pred[~tr] = predictor(fitter(F[tr], y[tr]), F[~tr])
    return pred


def prepare(D, seed):
    rng = np.random.default_rng(seed)
    is_train = np.zeros(D.n, bool)
    fold = np.full(D.n, -1)
    for w in range(len(D.wnames)):
        idx = rng.permutation(np.flatnonzero(D.W == w))
        tr = idx[: len(idx) // 2]
        is_train[tr] = True
        fold[tr] = np.arange(len(tr)) % NF
    P = SimpleNamespace(train=np.flatnonzero(is_train), test=np.flatnonzero(~is_train), is_train=is_train)
    proba = lambda m, F: m.predict_proba(F)[:, 1]
    allq = np.arange(D.n)
    v1 = np.flatnonzero(D.has_v1)
    tr_v1 = is_train & D.has_v1

    def shift(base, full, compressed):
        d = np.nanmean(compressed[tr_v1] - full[tr_v1]) if tr_v1.any() else 0.0
        return np.clip(base + d, 0, 1)

    # Router: predicted quality of each model per query.
    P.qhat = np.zeros((D.n, K, 2))
    for k in range(K):
        P.qhat[:, k, 0] = cross_fit(D.X, D.Q[:, k, 0], allq, is_train, fold, fit_soft, proba)
        P.qhat[:, k, 1] = shift(P.qhat[:, k, 0], D.Q[:, k, 0], D.Q[:, k, 1])
    # Verifier: calibrated probability that a cheap model's answer is good.
    P.s = np.full((D.n, K, 2), np.nan)
    P.shat = np.zeros((D.n, K, 2))
    P.sig = np.zeros(K)
    qidx = np.r_[allq, v1]
    var = np.r_[np.zeros(D.n, int), np.ones(len(v1), int)]
    for k in range(L):
        F = np.column_stack([D.VS[qidx, k, var], P.qhat[qidx, k, var], D.onehot[qidx]])
        s = cross_fit(F, D.Q[qidx, k, var], qidx, is_train, fold, fit_soft, proba)
        P.s[qidx, k, var] = s
        # Router-side forecast of the verifier score, to predict acceptance.
        ridge = lambda F, y: Ridge(alpha=1.0).fit(F, y)
        P.shat[:, k, 0] = cross_fit(D.X, P.s[:, k, 0], allq, is_train, fold, ridge, lambda m, F: m.predict(F))
        P.shat[:, k, 1] = shift(P.shat[:, k, 0], P.s[:, k, 0], P.s[:, k, 1])
        P.sig[k] = max(0.02, np.std(P.s[is_train, k, 0] - P.shat[is_train, k, 0]))
    # Cost model: train mean per workload, model and prompt variant.
    P.chat = np.zeros((D.n, K, 2))
    P.vchat = np.zeros((D.n, K, 2))
    for w in range(len(D.wnames)):
        g, gt = D.W == w, (D.W == w) & is_train
        for k in range(K):
            for v in range(2):
                if not np.isnan(D.C[gt, k, v]).all():
                    P.chat[g, k, v] = np.nanmean(D.C[gt, k, v])
                    if k < L:
                        P.vchat[g, k, v] = np.nanmean(D.VC[gt, k, v])
    P.cbar = D.C[P.train, L, 0].mean()
    return P


def crc_tau(D, P, alpha):
    """Conformal risk control: smallest thresholds whose accepted-answer loss is <= alpha."""
    tau = []
    for k in range(L):
        rows = [(P.train, 0), (P.train[D.has_v1[P.train]], 1)]
        s = np.concatenate([P.s[i, k, v] for i, v in rows])
        loss = np.concatenate([np.maximum(0, D.Q[i, L, 0] - D.Q[i, k, v]) for i, v in rows])
        n, pick = len(s), NEVER
        for t in TAU_GRID:
            if (n * np.mean(loss * (s >= t)) + 1) / (n + 1) <= alpha:
                pick = float(t)
                break
        tau.append(pick)
    return tau


# ---------------------------------------------------------------- policy

def realised(D, P, tau, cascade):
    """Logged cost, quality and latency of starting at model k with variant v."""
    shape = (D.n, K, 2)
    OC, OQ, OL = np.zeros(shape), np.full(shape, np.nan), np.zeros(shape)
    for v in range(2):
        for start in range(K):
            done = np.zeros(D.n, bool)
            for j in range(start, K):
                act = ~done
                OC[act, start, v] += D.C[act, j, v]
                OL[act, start, v] += D.LAT[act, j, v]
                if not cascade or j == L:
                    acc = np.ones(D.n, bool)
                else:
                    OC[act, start, v] += D.VC[act, j, v]
                    OL[act, start, v] += D.VL[act, j, v]
                    acc = P.s[:, j, v] >= tau[:, j]
                new = act & acc
                OQ[new, start, v] = D.Q[new, j, v]
                done |= new
    return OC, OQ, OL


def expected(P, tau, cascade):
    """Router's forecast of cascade cost and quality for each start model."""
    EC, EQ = P.chat.copy(), P.qhat.copy()
    if cascade:
        for k in range(L - 1, -1, -1):
            z = (tau[:, k, None] - P.shat[:, k, :]) / P.sig[k]
            a = 1 - stats.norm.cdf(z)
            q_acc = np.clip(P.shat[:, k, :] + P.sig[k] * stats.norm.pdf(z) / np.maximum(a, 1e-6), 0, 1)
            EC[:, k, :] = P.chat[:, k, :] + P.vchat[:, k, :] + (1 - a) * EC[:, k + 1, :]
            EQ[:, k, :] = a * q_acc + (1 - a) * EQ[:, k + 1, :]
    return EC, EQ


def make(D, P, tau=(NEVER,) * L, cascade=True, starts=None, compress=False, lam=0.0, fixed=None, cache=False,
         tau_cache=1.0, online=False, eta=ETA, rho=RHO, eps=EPS, budget=None, overhead=True):
    """Bundle a policy: thresholds, allowed actions, multipliers and precomputed tables.

    tau and lam are either one setting for all queries or one row per query.
    """
    tau = np.broadcast_to(np.asarray(tau, float), (D.n, L))
    lam = np.broadcast_to(np.asarray(lam, float), (D.n,)).copy()
    valid = np.zeros((D.n, K, 2), bool)
    for k in (range(K) if starts is None else starts):
        valid[:, k, 0] = True
        valid[:, k, 1] = D.has_v1 & compress
    OC, OQ, OL = realised(D, P, tau, cascade)
    EC, EQ = expected(P, tau, cascade)
    return SimpleNamespace(tau=tau, cascade=cascade, valid=valid, lam=lam, fixed=fixed, cache=cache,
                           tau_cache=tau_cache, online=online, eta=eta, rho=rho, eps=eps, budget=budget,
                           overhead=overhead, OC=OC, OQ=OQ, OL=OL, EC=EC, EQ=EQ)


def choose(P, cfg, idx, lam, mu=0.0):
    if cfg.fixed is not None:
        return cfg.fixed[idx]
    score = (1 + mu) * cfg.EC[idx] / P.cbar - np.reshape(lam, (-1, 1, 1)) * cfg.EQ[idx]
    score = np.where(cfg.valid[idx], score, np.inf)
    return score.reshape(len(idx), -1).argmin(axis=1)  # action = 2 * k + v


def static_eval(D, P, cfg):
    """Per-query cost and quality on distinct queries with no cache and fixed multipliers."""
    idx = np.arange(D.n)
    a = choose(P, cfg, idx, cfg.lam)
    return cfg.OC[idx, a // 2, a % 2], cfg.OQ[idx, a // 2, a % 2]


def serve(D, P, cfg, i, k, v, relax=1.0):
    """Run the cascade for one request from start model k; thresholds are divided by relax."""
    cost = lat = 0.0
    while True:
        cost += D.C[i, k, v]
        lat += D.LAT[i, k, v]
        if not cfg.cascade or k == L:
            break
        cost += D.VC[i, k, v]
        lat += D.VL[i, k, v]
        if P.s[i, k, v] >= cfg.tau[i, k] / relax:
            break
        k += 1
    return cost, D.Q[i, k, v], lat, k


def simulate(D, P, cfg, stream, rng):
    """Serve a request stream one by one; returns per-request arrays."""
    T = len(stream)
    cost, qual, lat = np.zeros(T), np.zeros(T), np.zeros(T)
    hit, false_hit, lam_trace = np.zeros(T, bool), np.zeros(T, bool), np.zeros(T)
    dlam, mu = 0.0, 0.0  # dlam: online correction added to the tuned multiplier
    M, meta, m = np.zeros((T, D.E0.shape[1]), np.float32), [], 0
    over = D.embed_time if cfg.overhead else 0.0
    for t, (i, para) in enumerate(stream):
        e = D.E1[i] if para else D.E0[i]
        served_ref = False
        if cfg.cache and m:
            sims = M[:m] @ e
            j = int(sims.argmax())
            if sims[j] >= cfg.tau_cache:
                src, k, v = meta[j]
                hit[t], false_hit[t] = True, src != i
                qual[t] = D.Q[i, k, v] if src == i else false_hit_quality(D, i, src, k, v)
                lat[t] = over
        if not hit[t]:
            a = int(choose(P, cfg, np.array([i]), max(0.0, cfg.lam[i] + dlam), mu)[0])
            k, v = a // 2, a % 2
            # Budget pressure (mu > 0) lowers the acceptance thresholds.
            cost[t], qual[t], lat[t], fin = serve(D, P, cfg, i, k, v, 1 + mu)
            lat[t] += over
            served_ref = fin == L and v == 0
            if cfg.cache:
                M[m], m = e, m + 1
                meta.append((i, fin, v))
        if cfg.online and rng.random() < cfg.rho:
            # Audit: the reference model answers too; its call is paid for.
            if not served_ref:
                cost[t] += D.C[i, L, 0]
            dlam += cfg.eta * ((1 - cfg.eps) * D.Q[i, L, 0] - qual[t]) / cfg.rho
        if cfg.budget is not None:
            mu = max(0.0, mu + ETA_MU * (cost[t] / P.cbar - cfg.budget))
        lam_trace[t] = dlam
    return SimpleNamespace(cost=cost, qual=qual, lat=lat, hit=hit, false_hit=false_hit, lam=lam_trace)


def metrics(D, run, stream, sel=None, eps=None):
    i = np.array([s[0] for s in stream])
    sel = np.ones(len(i), bool) if sel is None else sel
    ref_c, ref_q = D.C[i[sel], L, 0].mean(), D.Q[i[sel], L, 0].mean()
    met = {} if eps is None else {"met": 100.0 * (run.qual[sel].mean() >= (1 - eps) * ref_q)}
    return met | {
        "cost_per_1k": 1000 * run.cost[sel].mean(),
        "cost_red": 100 * (1 - run.cost[sel].mean() / ref_c),
        "q_ret": 100 * run.qual[sel].mean() / ref_q,
        "p95_lat": float(np.percentile(run.lat[sel], 95)),
        "hit_rate": 100 * run.hit[sel].mean(),
        "false_hit_rate": 100 * run.false_hit[sel].mean(),
    }


# ---------------------------------------------------------------- streams

def unique_stream(P, rng):
    return [(int(i), False) for i in rng.permutation(P.test)]


def replay_stream(D, idx, rng, repeat_share=REPEAT_SHARE):
    """Every query once, plus Zipf-distributed repeats; repeats may be paraphrased."""
    n_rep = int(round(len(idx) * repeat_share / (1 - repeat_share)))
    order = rng.permutation(idx)
    w = 1 / np.arange(1, len(order) + 1)
    reps = rng.choice(order, size=n_rep, p=w / w.sum())
    items = [(int(i), False) for i in order] + [(int(i), bool(D.has_para[i] and rng.random() < 0.5)) for i in reps]
    return [items[j] for j in rng.permutation(len(items))]


def drift_stream(D, idx, rng, T, shares=(0.8, 0.5)):
    """Share of easy queries (small model succeeds) drops halfway through."""
    easy, hard = idx[D.Q[idx, 0, 0] >= 0.8], idx[D.Q[idx, 0, 0] < 0.8]
    out = []
    for share in shares:
        pick_easy = rng.random(T // 2) < share
        out += [(int(rng.choice(easy if e else hard)), False) for e in pick_easy]
    return out


# ---------------------------------------------------------------- tuning (train split only)

def pick(cands, idx, ref, eps, z=Z_MARGIN):
    """Walk the calibration cost-quality front from the safest setting towards cheaper ones.

    A setting is accepted while its quality on idx clears the target by z standard
    errors; the walk stops at the first failure, so a cheap setting that passes by
    luck beyond a failing one is never chosen.
    """
    rows = sorted(((c[idx].mean(), q[idx].mean(), j) for j, (c, q, _) in enumerate(cands)), key=lambda r: (r[0], -r[1]))
    front, best_q = [], -np.inf
    for cost, qual, j in rows:
        if qual > best_q:
            front.append(j)
            best_q = qual
    chosen = front[-1]
    for j in reversed(front):
        d = cands[j][1][idx] - (1 - eps) * ref[idx]
        if d.mean() - z * d.std(ddof=1) / np.sqrt(len(idx)) < 0:
            break
        chosen = j
    return cands[chosen][2]


def pick_per_workload(D, P, cands, eps):
    """One setting per workload, so each workload meets the target on its own."""
    return [pick(cands, P.train[D.W[P.train] == w], D.Q[:, L, 0], eps) for w in range(len(D.wnames))]


def frugal_cands(D, P, **opts):
    """Per-query cost and quality of every (alpha, lambda) setting, plus always-large as a safe fallback."""
    cands = []
    use_lams = LAMS if opts.get("starts") is None else [0.0]
    for alpha in (ALPHAS if opts.get("cascade", True) else [None]):
        tau = crc_tau(D, P, alpha) if alpha else (NEVER,) * L
        for lam in use_lams:
            c, q = static_eval(D, P, make(D, P, tau=tau, lam=lam, **opts))
            cands.append((c, q, {"tau": tau, "lam": lam, "alpha": alpha}))
    if opts.get("starts") is None:
        cands.append((D.C[:, L, 0], D.Q[:, L, 0], {"tau": (NEVER,) * L, "lam": 0.0, "alpha": None, "starts": [L]}))
    return cands


def build(D, P, p, **kw):
    """Policy from tuned settings p: one dict for all traffic, or a list with one dict per workload."""
    if isinstance(p, dict):
        return make(D, P, tau=p["tau"], lam=p["lam"], **{"starts": p.get("starts"), **kw})
    parts = [build(D, P, pw, **kw) for pw in p]
    cfg = parts[0]
    for name in ("tau", "lam", "valid", "OC", "OQ", "OL", "EC", "EQ"):
        arr = getattr(cfg, name).copy()
        for w, part in enumerate(parts):
            arr[D.W == w] = getattr(part, name)[D.W == w]
        setattr(cfg, name, arr)
    return cfg


def cascade_cands(D, P):
    grid = np.r_[np.arange(0.30, 1.0, 0.04), NEVER]
    cands = []
    for t0 in grid:
        for t1 in grid:
            c, q = static_eval(D, P, make(D, P, tau=(t0, t1), starts=[0]))
            cands.append((c, q, (float(t0), float(t1))))
    return cands


def router_only_fixed(D, P, theta):
    """Binary router: small model if its predicted quality is at least theta (one value or one per query)."""
    return np.where(P.qhat[:, 0, 0] >= theta, 0, 2 * L)


def router_cands(D, P):
    cands = []
    for theta in np.r_[np.quantile(P.qhat[P.train, 0, 0], np.linspace(0, 1, 41)), 2.0]:
        c, q = static_eval(D, P, make(D, P, cascade=False, fixed=router_only_fixed(D, P, theta)))
        cands.append((c, q, float(theta)))
    return cands


def cache_false_hits(D, P, rng):
    """Wrong cache hits on a calibration replay stream, for each candidate threshold."""
    stream = replay_stream(D, P.train, rng)
    fixed = np.full(D.n, 2 * L)
    grid = np.round(np.arange(0.70, 1.0, 0.01), 2)
    hits = [simulate(D, P, make(D, P, cascade=False, fixed=fixed, cache=True, tau_cache=t), stream, rng).false_hit.sum()
            for t in grid]
    return grid, np.array(hits), len(stream)


def tune_tau_cache(table, eps):
    """Lowest threshold whose conformal bound on the false-hit rate fits the cache's share of the tolerance."""
    grid, hits, n = table
    for t, h in zip(grid, hits):
        if (h + 1) / (n + 1) <= CACHE_SHARE * eps:
            return float(t)
    return 1.0


def at_level(points, level):
    """Cost reduction at a given quality retained, interpolated along the Pareto front of (quality, saving) points."""
    front, best = [], -np.inf
    for q, r in sorted(points, key=lambda p: -p[0]):
        if r > best:
            front.append((q, r))
            best = r
    front.reverse()  # quality ascending
    return float(np.interp(level, [p[0] for p in front], [p[1] for p in front]))


# ---------------------------------------------------------------- experiments

def run_seed(D, seed):
    P = prepare(D, seed)
    rng = np.random.default_rng(1000 + seed)
    R = {}
    ref = D.Q[:, L, 0]
    cache_table = cache_false_hits(D, P, rng)
    tau_cache = tune_tau_cache(cache_table, EPS)
    FC, RC, CC = frugal_cands(D, P), router_cands(D, P), cascade_cands(D, P)
    fixed_large = np.full(D.n, 2 * L)
    large = lambda **kw: make(D, P, cascade=False, fixed=fixed_large, overhead=False, **kw)
    router = lambda theta: make(D, P, cascade=False, fixed=router_only_fixed(D, P, theta))
    theta_w = lambda eps: np.array(pick_per_workload(D, P, RC, eps))[D.W]
    frugal = lambda p, **kw: build(D, P, p, **{**dict(cache=True, tau_cache=tau_cache), **kw})
    route_eps = lambda eps: (1 - CACHE_SHARE) * eps  # tolerance left for routing once the cache has its share
    streams = {"replay": replay_stream(D, P.test, rng), "unique": unique_stream(P, rng)}

    def methods_at(eps):
        tc = tune_tau_cache(cache_table, eps)
        return {
            "Always-large": large(),
            "Always-small": make(D, P, cascade=False, fixed=np.zeros(D.n, int), overhead=False),
            "Cache-only": large(cache=True, tau_cache=tc),
            "Compression-only": make(D, P, cascade=False, starts=[L], compress=True, overhead=False),
            "Router-only": router(pick(RC, P.train, ref, eps)),
            "Cascade-only": make(D, P, tau=pick(CC, P.train, ref, eps), starts=[0], overhead=False),
            "FRUGAL-AI": frugal(pick(FC, P.train, ref, route_eps(eps)), eps=eps, tau_cache=tc),
            "Router-only (per-workload floors)": router(theta_w(eps)),
            "FRUGAL-AI (per-workload floors)": frugal(pick_per_workload(D, P, FC, route_eps(eps)), eps=eps, tau_cache=tc),
        }

    for eps in MAIN_EPS:
        for sname, stream in streams.items():
            for name, cfg in methods_at(eps).items():
                run = simulate(D, P, cfg, stream, np.random.default_rng(seed))
                R[f"main_{sname}_{eps}|{name}"] = metrics(D, run, stream, eps=eps)
                if eps == EPS and sname == "unique" and name.split(" (")[0] in ("FRUGAL-AI", "Router-only"):
                    w = D.W[[s[0] for s in stream]]
                    for wi, wname in enumerate(D.wnames):
                        R[f"workload|{name}|{wname}"] = metrics(D, run, stream, w == wi)
    fz = pick(FC, P.train, ref, route_eps(EPS))
    tuned = lambda **opts: pick(frugal_cands(D, P, **opts), P.train, ref, route_eps(EPS))
    methods = methods_at(EPS)

    # Ablations on the replay stream.
    ablations = {
        "Full FRUGAL-AI": methods["FRUGAL-AI"],
        "without cache": build(D, P, pick(FC, P.train, ref, EPS)),
        "with prompt compression": frugal(tuned(compress=True), compress=True),
        "without router (cascade from smallest)": frugal(tuned(starts=[0]), starts=[0]),
        "without cascade (three-way router)": frugal(tuned(cascade=False), cascade=False),
        "with online quality controller": frugal(fz, online=True),
    }
    for name, cfg in ablations.items():
        run = simulate(D, P, cfg, streams["replay"], np.random.default_rng(seed))
        R[f"ablation|{name}"] = metrics(D, run, streams["replay"], eps=EPS)

    # Cost-quality frontier on distinct test queries (no cache), and savings at matched quality.
    t = P.test
    point = lambda c, q: (100 * q[t].mean() / ref[t].mean(), 100 * (1 - c[t].mean() / D.C[t, L, 0].mean()))
    anchors = [(100.0, 0.0), point(D.C[:, 0, 0], D.Q[:, 0, 0])]
    sweep = {"FRUGAL-AI": [], "Router-only": [], "Cascade-only": []}
    for eps in SWEEP_EPS:
        tuned = {
            "FRUGAL-AI": build(D, P, pick(FC, P.train, ref, eps)),
            "Router-only": router(pick(RC, P.train, ref, eps)),
            "Cascade-only": make(D, P, tau=pick(CC, P.train, ref, eps), starts=[0]),
        }
        for name, cfg in tuned.items():
            q_ret, red = point(*static_eval(D, P, cfg))
            sweep[name].append((q_ret, red))
            R[f"pareto|{name}|{eps}"] = {"cost_red": red, "q_ret": q_ret}
    for level in LEVELS:
        at = {name: at_level(pts + anchors, level) for name, pts in sweep.items()}
        for name, v in at.items():
            R[f"matched|{name}|{level}"] = {"cost_red": v}
        R[f"matched|FRUGAL-AI minus Router-only|{level}"] = {"cost_red": at["FRUGAL-AI"] - at["Router-only"]}

    # Drift: the share of easy queries falls from 80% to 50% halfway through.
    T, win = 10000, 500
    dstream = drift_stream(D, P.test, rng, T)
    half = np.arange(T) >= T // 2
    drift = {
        "FRUGAL-AI (fixed multiplier)": frugal(fz, cache=False),
        "FRUGAL-AI + online quality controller": frugal(fz, cache=False, online=True),
        "Cascade-only (fixed)": methods["Cascade-only"],
    }
    ref_i = np.array([s[0] for s in dstream])
    for name, cfg in drift.items():
        run = simulate(D, P, cfg, dstream, np.random.default_rng(seed))
        for phase, sel in (("before", ~half), ("after", half)):
            R[f"drift|{name}|{phase}"] = metrics(D, run, dstream, sel)
        wq = run.qual.reshape(-1, win).mean(1) / D.Q[ref_i, L, 0].reshape(-1, win).mean(1)
        R[f"drift|{name}|windows"] = {"below_target": 100 * np.mean(wq < 1 - EPS - 0.01), "min_q_ret": 100 * wq.min()}
        if seed == 0:
            R[f"trace|{name}"] = {"q_ret": (100 * wq).tolist(),
                                  "cost_per_1k": (1000 * run.cost.reshape(-1, win).mean(1)).tolist()}

    # Budget mode: a per-request budget set 15% above the pre-drift spend of the fixed policy.
    pre = drift_stream(D, P.train, rng, 4000, shares=(0.8, 0.8))
    fixed_cfg = frugal(fz, cache=False)
    budget = 1.15 * simulate(D, P, fixed_cfg, pre, rng).cost.mean() / P.cbar
    for name, cfg in {"fixed": fixed_cfg, "online budget control": frugal(fz, cache=False, budget=budget)}.items():
        run = simulate(D, P, cfg, dstream, np.random.default_rng(seed))
        wc = run.cost.reshape(-1, win).mean(1) / (budget * P.cbar)
        after = wc[len(wc) // 2:]
        R[f"budget|{name}"] = {"windows_over": 100 * np.mean(after > 1.0), "worst_overshoot": 100 * (after.max() - 1),
                               "q_ret_after": metrics(D, run, dstream, half)["q_ret"]}
        if seed == 0:
            R[f"trace|budget {name}"] = {"spend_vs_budget": (100 * wc).tolist()}

    # Sensitivity on the replay stream.
    sens = {f"tau_cache={t}": frugal(fz, tau_cache=t) for t in (0.80, 0.85, 0.90, 0.95)}
    sens |= {f"online controller, rho={r}": frugal(fz, online=True, rho=r) for r in (0.005, 0.02, 0.05)}
    sens |= {f"online controller, eta={e}": frugal(fz, online=True, eta=e) for e in (0.01, 0.2, 0.5)}
    for name, cfg in sens.items():
        run = simulate(D, P, cfg, streams["replay"], np.random.default_rng(seed))
        R[f"sens|{name}"] = metrics(D, run, streams["replay"])
    for share in (0.0, 0.15, 0.30, 0.50):
        stream = replay_stream(D, P.test, np.random.default_rng(seed), share)
        run = simulate(D, P, methods["FRUGAL-AI"], stream, np.random.default_rng(seed))
        R[f"sens|repeat_share={share}"] = metrics(D, run, stream)

    R["params|tuned"] = {"tau_cache": tau_cache, "lambda0": fz["lam"], "alpha": fz["alpha"] or np.nan,
                         "tau_small": fz["tau"][0], "tau_mid": fz["tau"][1]}
    t0 = time.perf_counter()
    cfg = methods["FRUGAL-AI"]
    for i in P.test[:200]:
        choose(P, cfg, np.array([i]), cfg.lam[i])
    R["params|overhead_ms"] = {"embed": 1000 * D.embed_time, "route": 1000 * (time.perf_counter() - t0) / 200,
                               "verifier_call": 1000 * float(np.nanmean(D.VL[:, :L, 0]))}
    return R


# ---------------------------------------------------------------- report

def aggregate(seed_results):
    out = {}
    for key in seed_results[0]:
        if key.startswith("trace|"):
            out[key] = seed_results[0][key]
            continue
        out[key] = {}
        for m in seed_results[0][key]:
            x = np.array([r[key][m] for r in seed_results], float)
            x = x[~np.isnan(x)]
            ci = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
            out[key][m] = {"mean": float(x.mean()) if len(x) else float("nan"), "ci": float(ci)}
    return out


def table(agg, prefix, cols, first="Method"):
    """cols: (metric, title, format[, show_ci])."""
    rows = [k for k in agg if k.startswith(prefix + "|")]
    lines = ["| " + " | ".join([first] + [c[1] for c in cols]) + " |", "|" + " --- |" * (len(cols) + 1)]
    for k in rows:
        cells = [f"{agg[k][c[0]]['mean']:{c[2]}}" + (f" ± {agg[k][c[0]]['ci']:{c[2]}}" if len(c) < 4 or c[3] else "")
                 for c in cols]
        lines.append("| " + " | ".join([k[len(prefix) + 1:].replace("|", " · ")] + cells) + " |")
    return "\n".join(lines)


MAIN_COLS = [("cost_per_1k", "Cost (USD / 1k requests)", ".2f"), ("cost_red", "Cost reduction (%)", ".1f"),
             ("q_ret", "Quality retained (%)", ".1f"), ("p95_lat", "p95 latency (s)", ".1f")]
MET = ("met", "Seeds meeting target (%)", ".0f", False)
CACHE_COLS = MAIN_COLS[1:3] + [("hit_rate", "Cache hit rate (%)", ".1f"), ("false_hit_rate", "False-hit rate (%)", ".2f")]


def report(D, agg, seed_results, args):
    md = [f"# FRUGAL-AI results ({'MOCK DATA - not real model outputs' if args.mock else f'real model runs, {args.backend} backend'})", ""]
    md.append(f"{D.n} queries, {len(seed_results)} seeds, mean ± 95% CI over seeds. Seeds resample the split, stream "
              "and audits over one set of logged model outputs. Every tuned method keeps a setting only if its "
              f"quality on the calibration split clears the target by {Z_MARGIN:g} standard error(s).\n")
    md.append("## Model pool on all queries\n")
    md.append("| Model | Quality | Cost (USD / 1k requests) | Mean latency (s) |\n| --- | --- | --- | --- |")
    for k in range(K):
        md.append(f"| {MODELS[k]['name']} ({MODELS[k]['id']}) | {D.Q[:, k, 0].mean():.3f} | "
                  f"{1000 * D.C[:, k, 0].mean():.2f} | {D.LAT[:, k, 0].mean():.1f} |")
    md.append("\n| Workload | n | small | mid | large |\n| --- | --- | --- | --- | --- |")
    for wi, w in enumerate(D.wnames):
        g = D.W == wi
        md.append(f"| {w} | {g.sum()} | " + " | ".join(f"{D.Q[g, k, 0].mean():.3f}" for k in range(K)) + " |")
    sections = []
    for eps in MAIN_EPS:
        tgt = f"target {100 * (1 - eps):.0f}% of always-large"
        sections += [
            (f"Main results, replay stream ({100 * REPEAT_SHARE:.0f}% repeated requests), {tgt}", f"main_replay_{eps}", MAIN_COLS + [MET], "Method"),
            (f"Main results, distinct queries only, {tgt}", f"main_unique_{eps}", MAIN_COLS + [MET], "Method"),
        ]
    sections += [
        (f"By workload (distinct queries, target {100 * (1 - EPS):.0f}%; method · workload)", "workload", MAIN_COLS[1:3], "Method"),
        (f"Ablation (replay stream, target {100 * (1 - EPS):.0f}%)", "ablation", MAIN_COLS[:3] + [MET], "Configuration"),
        ("Cost reduction at matched quality retained (distinct queries; method · quality retained)", "matched", [MAIN_COLS[1]], "Method"),
        ("Cost-quality frontier (distinct queries; method · tolerance)", "pareto", MAIN_COLS[1:3], "Method"),
        ("Sensitivity (replay stream)", "sens", CACHE_COLS, "Setting"),
    ]
    for title, prefix, cols, first in sections:
        md += [f"\n## {title}\n", table(agg, prefix, cols, first)]
    md.append("\n## Drift: easy share falls from 80% to 50%\n")
    md.append(table({k: v for k, v in agg.items() if not k.endswith("windows")}, "drift", MAIN_COLS[1:3], "Method · phase"))
    md.append("")
    md.append(table({k: v for k, v in agg.items() if k.endswith("windows")}, "drift",
                    [("below_target", "Windows > 1 pt below target (%)", ".1f"), ("min_q_ret", "Worst window quality retained (%)", ".1f")]))
    md += ["\n## Budget control after drift\n",
           table(agg, "budget", [("windows_over", "Windows over budget (%)", ".1f"), ("worst_overshoot", "Worst overshoot (%)", ".1f"),
                                 ("q_ret_after", "Quality retained (%)", ".1f")], "Policy")]
    md += ["\n## Tuned parameters and overhead\n",
           "\n".join(f"- {m}: {v['mean']:.3f} ± {v['ci']:.3f}" for key in ("params|tuned", "params|overhead_ms")
                     for m, v in agg[key].items())]
    md.append("\n## Significance (paired t-tests across seeds)\n")
    md.append(f"Cost per 1k requests on the replay stream, target {100 * (1 - EPS):.0f}%:\n")
    md.append("| FRUGAL-AI vs. | Mean cost difference (USD / 1k) | Mean quality-retained difference (points) | p-value (cost) |\n| --- | --- | --- | --- |")
    get = lambda name, m: np.array([r[f"main_replay_{EPS}|{name}"][m] for r in seed_results])
    for name in ("Cache-only", "Compression-only", "Router-only", "Cascade-only"):
        dc, dq = get("FRUGAL-AI", "cost_per_1k") - get(name, "cost_per_1k"), get("FRUGAL-AI", "q_ret") - get(name, "q_ret")
        p = stats.ttest_1samp(dc, 0).pvalue if len(dc) > 1 else float("nan")
        md.append(f"| {name} | {dc.mean():+.2f} | {dq.mean():+.1f} | {p:.4f} |")
    md.append("\nCost reduction at matched quality, FRUGAL-AI minus Router-only (distinct queries):\n")
    md.append("| Quality retained (%) | Difference (points) | p-value |\n| --- | --- | --- |")
    for level in LEVELS:
        d = np.array([r[f"matched|FRUGAL-AI minus Router-only|{level}"]["cost_red"] for r in seed_results])
        p = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 and d.std() > 0 else float("nan")
        md.append(f"| {level:g} | {d.mean():+.1f} | {p:.4f} |")
    (D.out / "RESULTS.md").write_text("\n".join(md) + "\n")
    (D.out / "summary.json").write_text(json.dumps(agg, indent=1))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--backend", choices=["api", "cli"], default="api")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    D = load("mock" if args.mock else args.backend, args.pilot)
    seed_results = [run_seed(D, s) for s in range(args.seeds)]
    report(D, aggregate(seed_results), seed_results, args)


if __name__ == "__main__":
    main()
