"""Command line for the demo.

  python cli.py reset                      # (re)create the schema and roles
  python cli.py seed                       # create the fake source system with 300 documents
  python cli.py sync | reconcile | rebuild
  python cli.py mutate --days 3            # simulate three days of edits, deletes and permission changes
  python cli.py ask --groups claims "What is the claims escalation threshold in the 2024 HomeShield claims handling manual?"
  python cli.py golden                     # freeze the evaluation set from the current source
  python cli.py eval --mode dense --runs 3 # run the evaluation for one configuration
  python cli.py compare dense hybrid       # paired comparison + release gates (exit code 1 if a gate fails)

EMBEDDER=openai uses text-embedding-3-small (needs OPENAI_API_KEY); the default is an offline hashing embedder.
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag import corpus, db, evaluate  # noqa: E402
from rag.embed import get_embedder  # noqa: E402
from rag.source import SourceSystem  # noqa: E402
from rag.sync import Syncer  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE_DB = HERE / "data" / "source.db"
GOLDEN = HERE / "eval" / "golden.jsonl"
RESULTS = HERE / "eval" / "results"

GATES = {"max_p95_latency_ratio": 1.25, "max_cost_ratio": 1.25, "alpha": 0.05, "max_category_drop": 0.05}


def syncer(embedder=None):
    return Syncer(SourceSystem(str(SOURCE_DB)), embedder or get_embedder(), db.sync_conn())


def cmd_seed(a):
    SOURCE_DB.parent.mkdir(exist_ok=True)
    SOURCE_DB.unlink(missing_ok=True)
    SourceSystem(str(SOURCE_DB)).seed(corpus.generate(n_docs=a.docs))
    print(f"source system seeded with {a.docs} documents")


def cmd_ask(a):
    from rag.answer import Answerer
    from rag.retrieve import retrieve

    groups = a.groups.split(",") + [corpus.ALL_STAFF]
    hits = retrieve(db.app_conn(), get_embedder(), groups, a.question, mode=a.mode)
    for h in hits:
        print(f"  {h['score']:.3f} {h['chunk_id']} acl={h['acl']}")
    print(json.dumps(Answerer().answer(a.question, hits), indent=2))


def cmd_golden(a):
    items = evaluate.build_golden(SourceSystem(str(SOURCE_DB)))
    GOLDEN.write_text("".join(json.dumps(i) + "\n" for i in items))
    print(f"{len(items)} items frozen in {GOLDEN}")


def cmd_eval(a):
    from rag.answer import Answerer

    items = [json.loads(x) for x in GOLDEN.read_text().splitlines()]
    items = [i for i in items if a.split == "all" or i["split"] == a.split]
    embedder = get_embedder()
    runs = []
    for n in range(a.runs):
        before = embedder.tokens
        rows = evaluate.run(items, embedder, Answerer(), a.mode)
        embed_usd = (embedder.tokens - before) / len(items) * 1000 * getattr(embedder, "usd_per_million", 0) / 1e6
        runs.append({"rows": rows, "summary": evaluate.summarize(rows, embed_usd)})
        print(f"run {n + 1}: " + json.dumps({k: round(v, 3) for k, v in runs[-1]["summary"].items()}))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{a.mode}_{a.split}.json").write_text(json.dumps({"mode": a.mode, "split": a.split, "embedder": embedder.name, "runs": runs}))


def cmd_compare(a):
    base = json.loads((RESULTS / f"{a.baseline}_{a.split}.json").read_text())
    cand = json.loads((RESULTS / f"{a.candidate}_{a.split}.json").read_text())
    keys = list(base["runs"][0]["summary"])
    print(f"\n{a.split} split, {len(base['runs'][0]['rows'])} items, mean (min–max) over {len(base['runs'])} runs\n")
    print(f"| metric | {a.baseline} | {a.candidate} |\n|---|---|---|")
    for k in keys:
        cells = []
        for res in (base, cand):
            m, lo, hi = evaluate.spread([r["summary"][k] for r in res["runs"]])
            cells.append(f"{m:.3f} ({lo:.3f}–{hi:.3f})" if hi != lo else f"{m:.3f}")
        print(f"| {k} | {cells[0]} | {cells[1]} |")

    vb, vc = evaluate.majority([r["rows"] for r in base["runs"]]), evaluate.majority([r["rows"] for r in cand["runs"]])
    b = sum(1 for i in vb if vb[i] and not vc[i])
    c = sum(1 for i in vb if vc[i] and not vb[i])
    p = evaluate.mcnemar_exact(b, c)
    print(f"\nPaired items (majority of runs): only {a.baseline} right = {b}, only {a.candidate} right = {c}, exact McNemar p = {p:.4f}")

    mean = lambda res, k: evaluate.spread([r["summary"][k] for r in res["runs"]])[0]  # noqa: E731
    # Per category, not just overall: a regression in a small, critical category (e.g. refusals)
    # can hide inside an aggregate that improves. v1 of this gate missed exactly that.
    gold = {json.loads(x)["id"]: json.loads(x)["type"] for x in GOLDEN.read_text().splitlines()}
    def share(votes, t):
        ids = [i for i in votes if gold[i] == t]
        return sum(votes[i] for i in ids) / len(ids)
    print("\nPer category (majority of runs):")
    category_ok = True
    for t in ("answerable", "probe", "missing"):
        sb, sc = share(vb, t), share(vc, t)
        ok = sc >= sb - GATES["max_category_drop"]
        category_ok &= ok
        print(f"  {t:<11} {a.baseline} {sb:.0%} → {a.candidate} {sc:.0%} {'' if ok else '  <-- regression'}")

    gates = {
        "zero leaks (every run)": all(r["summary"]["leaks"] == 0 for r in cand["runs"]),
        f"no category drops more than {GATES['max_category_drop']:.0%}": category_ok,
        "no significant quality drop": not (c < b and p < GATES["alpha"]),
        f"p95 latency <= {GATES['max_p95_latency_ratio']}x baseline": mean(cand, "total_p95_s") <= GATES["max_p95_latency_ratio"] * mean(base, "total_p95_s"),
        f"cost <= {GATES['max_cost_ratio']}x baseline": mean(cand, "usd_per_1k_queries") <= GATES["max_cost_ratio"] * mean(base, "usd_per_1k_queries"),
    }
    print("\nGates:")
    for name, ok in gates.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    sys.exit(0 if all(gates.values()) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reset")
    s = sub.add_parser("seed"); s.add_argument("--docs", type=int, default=300)
    sub.add_parser("sync"); sub.add_parser("reconcile"); sub.add_parser("rebuild")
    m = sub.add_parser("mutate"); m.add_argument("--days", type=int, default=1)
    q = sub.add_parser("ask"); q.add_argument("--groups", required=True); q.add_argument("--mode", default="hybrid"); q.add_argument("question")
    sub.add_parser("golden")
    e = sub.add_parser("eval"); e.add_argument("--mode", default="hybrid"); e.add_argument("--runs", type=int, default=3); e.add_argument("--split", default="test")
    c = sub.add_parser("compare"); c.add_argument("baseline"); c.add_argument("candidate"); c.add_argument("--split", default="test")
    a = ap.parse_args()

    if a.cmd == "reset":
        db.reset_schema(); print("schema and roles created")
    elif a.cmd == "seed":
        cmd_seed(a)
    elif a.cmd == "sync":
        print(syncer().run_incremental())
    elif a.cmd == "reconcile":
        print(syncer().reconcile())
    elif a.cmd == "rebuild":
        print(syncer().full_rebuild())
    elif a.cmd == "mutate":
        src = SourceSystem(str(SOURCE_DB))
        for d in range(1, a.days + 1):
            print(f"day {d}:", src.mutate_day(d))
    elif a.cmd == "ask":
        cmd_ask(a)
    elif a.cmd == "golden":
        cmd_golden(a)
    elif a.cmd == "eval":
        cmd_eval(a)
    elif a.cmd == "compare":
        cmd_compare(a)


if __name__ == "__main__":
    main()
