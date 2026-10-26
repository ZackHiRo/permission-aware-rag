"""Keep the index in sync with the source system.

Rules this module is built around:
  1. The change feed says *which* documents changed; the source is re-read for *what* they are now.
     Replaying an event is therefore harmless: applying the same version twice is a no-op.
  2. Each document is replaced in ONE Postgres transaction (vectors live in the same database), so a
     crash leaves either the old version or the new one, never a mix, and never stale extra chunks.
  3. The cursor moves only after every change it covers is applied or parked in dead_letters,
     in the same transaction that marks the run as finished. A crashed run restarts from the old cursor.
  4. Unchanged content is never re-embedded; a permission-only change updates ACLs in place.
  5. Deletes the feed never reported are found by reconcile(), which compares full listings,
     behind a guard that refuses to delete a large share of the index at once.
"""

import hashlib
import json

import psycopg

from .chunking import chunk
from .embed import to_pgvector

SOURCE = "maple-dms"


class CrashForTest(Exception):
    """Raised by crash_after to simulate the process dying mid-run."""


class MassDeleteGuard(Exception):
    """Refuse to delete a large share of the index: usually a broken listing, not real deletions."""


def content_hash(title: str, body: str) -> str:
    return hashlib.sha256(f"{title}\n\n{body}".encode()).hexdigest()


class Syncer:
    def __init__(self, source, embedder, conn: psycopg.Connection, max_delete_fraction: float = 0.2):
        self.source, self.embedder, self.conn = source, embedder, conn
        self.max_delete_fraction = max_delete_fraction

    # --- bookkeeping ---------------------------------------------------------------------------
    def cursor(self) -> int:
        with self.conn.transaction():
            row = self.conn.execute("SELECT cursor FROM sync_state WHERE source = %s", (SOURCE,)).fetchone()
        return row[0] if row else 0

    def _start_run(self, mode: str, cursor_from: int | None) -> int:
        with self.conn.transaction():
            return self.conn.execute(
                "INSERT INTO sync_runs (mode, cursor_from) VALUES (%s, %s) RETURNING run_id", (mode, cursor_from)
            ).fetchone()[0]

    def _finish_run(self, run_id: int, stats: dict, cursor_to: int | None = None):
        """Finishing a run and moving the cursor happen together, or not at all."""
        with self.conn.transaction():
            if cursor_to is not None:
                self.conn.execute(
                    "INSERT INTO sync_state (source, cursor) VALUES (%s, %s) "
                    "ON CONFLICT (source) DO UPDATE SET cursor = excluded.cursor, updated_at = now()",
                    (SOURCE, cursor_to),
                )
            self.conn.execute(
                "UPDATE sync_runs SET status = 'ok', finished_at = now(), cursor_to = %s, stats = %s WHERE run_id = %s",
                (cursor_to, json.dumps(stats), run_id),
            )

    def _dead_letter(self, doc_id: str, error: str):
        with self.conn.transaction():
            self.conn.execute(
                "INSERT INTO dead_letters (doc_id, error) VALUES (%s, %s) ON CONFLICT (doc_id) DO UPDATE "
                "SET error = excluded.error, attempts = dead_letters.attempts + 1, last_seen = now()",
                (doc_id, error[:500]),
            )

    # --- applying one document -----------------------------------------------------------------
    def apply(self, doc_id: str) -> str:
        doc = self.source.get(doc_id)
        with self.conn.transaction():
            row = self.conn.execute(
                "SELECT source_version, content_hash, acl FROM documents WHERE doc_id = %s", (doc_id,)
            ).fetchone()
        if doc is None:
            if not row:
                return "noop"
            with self.conn.transaction():
                self.conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))  # chunks cascade
            return "deleted"
        if row and row[0] >= doc["version"]:
            return "noop"  # this version (or a newer one) is already indexed: replaying an event is safe

        h = content_hash(doc["title"], doc["body"])
        if row and row[1] == h:
            with self.conn.transaction():
                self.conn.execute("UPDATE documents SET source_version = %s, acl = %s, indexed_at = now() WHERE doc_id = %s",
                                  (doc["version"], doc["acl"], doc_id))
                if sorted(row[2]) != sorted(doc["acl"]):
                    self.conn.execute("UPDATE chunks SET acl = %s WHERE doc_id = %s", (doc["acl"], doc_id))
                    return "acl_updated"
            return "unchanged"

        texts = chunk(doc["title"], doc["body"])
        vectors = self.embedder.embed(texts)  # network call: done before the transaction opens
        with self.conn.transaction():
            self.conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            self.conn.execute(
                "INSERT INTO documents (doc_id, title, source_version, content_hash, acl) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (doc_id) DO UPDATE SET title = excluded.title, source_version = excluded.source_version, "
                "content_hash = excluded.content_hash, acl = excluded.acl, indexed_at = now()",
                (doc_id, doc["title"], doc["version"], h, doc["acl"]),
            )
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (chunk_id, doc_id, idx, body, acl, embedding) VALUES (%s, %s, %s, %s, %s, %s::vector)",
                    [(f"{doc_id}:{h[:12]}:{i}", doc_id, i, t, doc["acl"], to_pgvector(v))
                     for i, (t, v) in enumerate(zip(texts, vectors))],
                )
        return "reindexed"

    # --- runs ----------------------------------------------------------------------------------
    def run_incremental(self, crash_after: int | None = None, fail_on: set[str] | None = None) -> dict:
        start = self.cursor()
        run_id = self._start_run("incremental", start)
        events, next_cursor = self.source.changes_since(start)
        with self.conn.transaction():
            retry = [r[0] for r in self.conn.execute("SELECT doc_id FROM dead_letters ORDER BY doc_id").fetchall()]
        doc_ids = list(dict.fromkeys(retry + [doc_id for _, doc_id, _ in events]))  # unique, order kept

        stats: dict[str, int] = {}
        for i, doc_id in enumerate(doc_ids):
            if crash_after is not None and i == crash_after:
                raise CrashForTest(f"simulated crash after {i} documents")
            try:
                if fail_on and doc_id in fail_on:
                    raise ValueError("simulated unparseable document")
                outcome = self.apply(doc_id)
                with self.conn.transaction():
                    self.conn.execute("DELETE FROM dead_letters WHERE doc_id = %s", (doc_id,))
            except (ValueError, psycopg.DataError) as e:  # a bad document: park it, keep going
                self._dead_letter(doc_id, str(e))
                outcome = "dead_lettered"
            stats[outcome] = stats.get(outcome, 0) + 1
        self._finish_run(run_id, stats, cursor_to=next_cursor)
        return stats

    def reconcile(self, force: bool = False) -> dict:
        """Compare full listings: catches deletions and changes the feed never reported."""
        run_id = self._start_run("reconcile", None)
        source = self.source.list_versions()
        with self.conn.transaction():
            indexed = dict(self.conn.execute("SELECT doc_id, source_version FROM documents").fetchall())
        orphans = sorted(set(indexed) - set(source))
        if len(orphans) > max(5, self.max_delete_fraction * len(indexed)) and not force:
            raise MassDeleteGuard(f"refusing to delete {len(orphans)} of {len(indexed)} documents; check the source listing")
        stale = sorted(d for d, v in source.items() if indexed.get(d, -1) < v)
        stats = {"orphans_deleted": 0, "stale_applied": 0}
        for doc_id in orphans:
            with self.conn.transaction():
                self.conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            stats["orphans_deleted"] += 1
        for doc_id in stale:
            self.apply(doc_id)
            stats["stale_applied"] += 1
        self._finish_run(run_id, stats)
        return stats

    def full_rebuild(self) -> dict:
        """Index everything from scratch: the reference an incremental sync must match."""
        with self.conn.transaction():
            self.conn.execute("DELETE FROM documents")
            self.conn.execute("DELETE FROM dead_letters")
        stats = {"reindexed": 0}
        for doc_id in sorted(self.source.list_versions()):
            self.apply(doc_id)
            stats["reindexed"] += 1
        _, latest = self.source.changes_since(0, limit=10**9)
        run_id = self._start_run("rebuild", None)
        self._finish_run(run_id, stats, cursor_to=latest)
        return stats


def snapshot(conn: psycopg.Connection) -> list[tuple]:
    """Everything that defines the index's content, in a comparable form."""
    with conn.transaction():
        return conn.execute(
            """
            SELECT d.doc_id, d.source_version, d.content_hash, d.acl,
                   array_agg(c.chunk_id ORDER BY c.idx), array_agg(md5(c.body) ORDER BY c.idx),
                   array_agg(md5(c.embedding::text) ORDER BY c.idx), bool_and(c.acl = d.acl)
            FROM documents d JOIN chunks c USING (doc_id)
            GROUP BY d.doc_id ORDER BY d.doc_id
            """
        ).fetchall()
