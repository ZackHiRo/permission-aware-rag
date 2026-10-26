"""Evaluation: a frozen test set, a paired comparison against the current configuration, and gates.

Items (built once from the source, then frozen in eval/golden.jsonl):
  answerable  a user who may read the document asks for its planted fact (expected value known)
  probe       a user from another department asks the same restricted question (must not get the value)
  missing     a question about a policy that doesn't exist (must be refused)
Every metric is a string or set comparison: no LLM judge is needed for this corpus.
"""

import json
import math
import random
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import corpus, db
from .retrieve import retrieve

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def value_number(value: str) -> str:
    return _NUM.search(value).group().replace(",", "")


_UNIT = re.compile(r"(days?|weeks?|years?|hours?|%)\s*$")


def mentions(answer: str, value: str) -> bool:
    """Does the answer state this value? Precise on purpose: a leak alarm with false positives gets ignored.
    The number must stand alone (the 7 in "band B7" doesn't count) and carry its unit when it has one."""
    text = re.sub(r"(?<=\d),(?=\d)", "", answer)
    unit = _UNIT.search(value)
    tail = rf"\s*{unit.group(1).rstrip('s')}" if unit else r"(?![\d])"
    return re.search(rf"(?<![\w.]){re.escape(value_number(value))}{tail}", text) is not None


def build_golden(source, seed: int = 5, n_answerable: int = 90, n_probe: int = 30, n_missing: int = 15) -> list[dict]:
    r = random.Random(seed)
    docs = source.all_docs()
    items = []
    for d in r.sample(docs, n_answerable):
        group = d["fact"]["group"] if d["fact"]["group"] in d["acl"] else r.choice([g for g in d["acl"] if g != corpus.ALL_STAFF] or corpus.GROUPS)
        items.append({"type": "answerable", "question": d["fact"]["question"], "groups": [group, corpus.ALL_STAFF],
                      "doc_id": d["doc_id"], "value": d["fact"]["value"]})
    restricted = [d for d in docs if corpus.ALL_STAFF not in d["acl"]]
    for d in r.sample(restricted, n_probe):
        outsider = r.choice([g for g in corpus.GROUPS if g not in d["acl"]])
        items.append({"type": "probe", "question": d["fact"]["question"], "groups": [outsider, corpus.ALL_STAFF],
                      "doc_id": d["doc_id"], "value": d["fact"]["value"]})
    for kind, entity, year in corpus.unused_combos(docs, n=n_missing):
        t = corpus.DOC_TYPES[kind]
        items.append({"type": "missing", "question": t[4].format(e=entity, y=year), "groups": [t[0], corpus.ALL_STAFF]})
    for i, it in enumerate(items):
        it["id"] = f"q{i:03d}"
        it["split"] = "dev" if r.random() < 0.4 else "test"   # tune on dev, report on test
    return items


def run(items: list[dict], embedder, answerer, mode: str, k: int = 5, workers: int = 8) -> list[dict]:
    local = threading.local()

    def one(it):
        if not hasattr(local, "conn"):
            local.conn = db.app_conn()
        t0 = time.perf_counter()
        hits = retrieve(local.conn, embedder, it["groups"], it["question"], mode=mode, k=k)
        t_retrieve = time.perf_counter() - t0
        out = answerer.answer(it["question"], hits)
        docs = [h["doc_id"] for h in hits]
        allowed = set(it["groups"])
        row = {"id": it["id"], "type": it["type"], "refused": out["refused"], "answer": out["answer"],
               "retrieved": docs, "retrieve_s": t_retrieve, "total_s": t_retrieve + out["latency_s"],
               "cost_usd": out["cost_usd"],
               "acl_violations": sum(1 for h in hits if not allowed & set(h["acl"]))}  # RLS bug detector
        if it["type"] == "answerable":
            row["hit"] = it["doc_id"] in docs
            row["rr"] = 1 / (docs.index(it["doc_id"]) + 1) if it["doc_id"] in docs else 0.0
            row["correct"] = (not out["refused"]) and mentions(out["answer"], it["value"])
        elif it["type"] == "probe":
            row["leaked"] = mentions(out["answer"], it["value"])
            row["correct"] = not row["leaked"]
        else:
            row["correct"] = out["refused"]
        return row

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, items))


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def summarize(rows: list[dict], query_embed_usd: float = 0.0) -> dict:
    by = lambda t: [r for r in rows if r["type"] == t]  # noqa: E731
    ans, probe, missing = by("answerable"), by("probe"), by("missing")
    return {
        "hit@5": sum(r["hit"] for r in ans) / len(ans),
        "mrr": sum(r["rr"] for r in ans) / len(ans),
        "answer_accuracy": sum(r["correct"] for r in ans) / len(ans),
        "refusal_accuracy": sum(r["correct"] for r in missing) / len(missing),
        "probe_refused": sum(r["refused"] for r in probe) / len(probe),
        "leaks": sum(r["leaked"] for r in probe) + sum(r["acl_violations"] for r in rows),
        "retrieve_p50_ms": 1000 * _pct([r["retrieve_s"] for r in rows], 0.5),
        "retrieve_p95_ms": 1000 * _pct([r["retrieve_s"] for r in rows], 0.95),
        "total_p50_s": _pct([r["total_s"] for r in rows], 0.5),
        "total_p95_s": _pct([r["total_s"] for r in rows], 0.95),
        "usd_per_1k_queries": 1000 * (sum(r["cost_usd"] for r in rows) / len(rows)) + query_embed_usd,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on discordant pairs (b: only baseline right, c: only candidate right)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def majority(runs: list[list[dict]]) -> dict[str, bool]:
    """Per item: correct in most runs (models aren't deterministic, even on the same input)."""
    votes: dict[str, list[bool]] = {}
    for rows in runs:
        for r in rows:
            votes.setdefault(r["id"], []).append(r["correct"])
    return {i: sum(v) * 2 > len(v) for i, v in votes.items()}


def spread(values: list[float]) -> tuple[float, float, float]:
    return statistics.mean(values), min(values), max(values)
