"""The filtered-ANN pitfall: with a permission filter, HNSW finds ef_search nearest candidates FIRST and
filters AFTER. A small department then silently gets fewer than k results (it looks like a quality bug,
not a security one). pgvector 0.8 iterative index scans keep scanning until k rows pass the filter."""

import random

from rag.db import as_user
from rag.embed import DIM, to_pgvector


def _unit(r):
    v = [r.gauss(0, 1) for _ in range(DIM)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def test_iterative_scan_fixes_missing_results_for_a_small_department(env):
    r = random.Random(3)
    with env.sync_conn.transaction():  # a large archive only one other group can read
        env.sync_conn.execute("INSERT INTO documents (doc_id, title, source_version, content_hash, acl) "
                              "VALUES ('archive', 'Archive', 1, 'x', '{archive}')")
        with env.sync_conn.cursor() as cur:
            cur.executemany("INSERT INTO chunks (chunk_id, doc_id, idx, body, acl, embedding) "
                            "VALUES (%s, 'archive', %s, 'archived', '{archive}', %s::vector)",
                            [(f"archive:{i}", i, to_pgvector(_unit(r))) for i in range(8000)])
        env.sync_conn.execute("ANALYZE chunks")

    q = to_pgvector(_unit(r))
    sql = "SELECT chunk_id FROM chunks ORDER BY embedding <=> %s::vector LIMIT 10"

    with as_user(env.app_conn, ["legal"], iterative_scan=False):
        plan = "\n".join(row[0] for row in env.app_conn.execute("EXPLAIN " + sql, (q,)).fetchall())
        strict = env.app_conn.execute(sql, (q,)).fetchall()
        visible = env.app_conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    with as_user(env.app_conn, ["legal"], iterative_scan=True):
        iterative = env.app_conn.execute(sql, (q,)).fetchall()

    assert "chunks_embedding_idx" in plan       # the HNSW index is used, with the RLS filter on top
    assert visible >= 10                        # the user can read plenty of chunks...
    assert len(strict) < 10                     # ...but post-filtering returned fewer than asked
    assert len(iterative) == 10                 # iterative scan returns a full page
    print(f"\nlegal can read {visible} chunks; LIMIT 10 returned {len(strict)} without iterative scan, {len(iterative)} with it")
