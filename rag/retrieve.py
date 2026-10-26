"""Retrieval as a given user. Row-level security filters inside the search, on both legs:
the application code below has no WHERE clause about permissions, and couldn't widen access if it tried."""

import re

from .db import as_user
from .embed import to_pgvector

_WORD = re.compile(r"[a-z0-9]+")
RRF_K = 60  # reciprocal rank fusion constant
CANDIDATES = 40  # per leg, before fusion


def _or_query(text: str) -> str:
    return " | ".join(dict.fromkeys(_WORD.findall(text.lower()))) or "''"


def dense(conn, groups: list[str], qvec: list[float], k: int = 5, iterative_scan: bool = True) -> list[dict]:
    with as_user(conn, groups, iterative_scan):
        rows = conn.execute(
            "SELECT chunk_id, doc_id, body, acl, 1 - (embedding <=> %s::vector) AS score "
            "FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
            (to_pgvector(qvec), to_pgvector(qvec), k),
        ).fetchall()
    return [dict(chunk_id=r[0], doc_id=r[1], body=r[2], acl=r[3], score=float(r[4])) for r in rows]


def hybrid(conn, groups: list[str], text: str, qvec: list[float], k: int = 5) -> list[dict]:
    """Dense + full-text, fused with reciprocal rank fusion. Exact tokens (years, product names,
    band codes) are where dense retrieval alone is weakest."""
    with as_user(conn, groups):
        rows = conn.execute(
            f"""
            WITH dense AS (
              SELECT chunk_id, embedding <=> %(v)s::vector AS dist FROM chunks
              ORDER BY embedding <=> %(v)s::vector LIMIT {CANDIDATES}),
            dense_ranked AS (SELECT chunk_id, row_number() OVER (ORDER BY dist) AS r FROM dense),
            sparse AS (
              SELECT chunk_id, ts_rank_cd(tsv, q) AS rank FROM chunks, to_tsquery('english', %(q)s) q
              WHERE tsv @@ q ORDER BY ts_rank_cd(tsv, q) DESC LIMIT {CANDIDATES}),
            sparse_ranked AS (SELECT chunk_id, row_number() OVER (ORDER BY rank DESC) AS r FROM sparse),
            fused AS (
              SELECT chunk_id, sum(1.0 / ({RRF_K} + r)) AS score
              FROM (SELECT * FROM dense_ranked UNION ALL SELECT * FROM sparse_ranked) u GROUP BY chunk_id)
            SELECT c.chunk_id, c.doc_id, c.body, c.acl, f.score
            FROM fused f JOIN chunks c USING (chunk_id) ORDER BY f.score DESC LIMIT %(k)s
            """,
            {"v": to_pgvector(qvec), "q": _or_query(text), "k": k},
        ).fetchall()
    return [dict(chunk_id=r[0], doc_id=r[1], body=r[2], acl=r[3], score=float(r[4])) for r in rows]


def retrieve(conn, embedder, groups: list[str], question: str, mode: str = "hybrid", k: int = 5) -> list[dict]:
    qvec = embedder.embed([question])[0]
    return hybrid(conn, groups, question, qvec, k) if mode == "hybrid" else dense(conn, groups, qvec, k)
