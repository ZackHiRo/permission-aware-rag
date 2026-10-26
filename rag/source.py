"""A stand-in for the client's source system (think SharePoint or a DMS), in SQLite.

It behaves like real change feeds do:
  - changes_since(cursor) returns events after an opaque, monotonically increasing cursor;
  - events say *that* a document changed, not *what* it is now: the sync re-reads the document;
  - some deletions never reach the feed (e.g. an admin purge), which only a reconciliation catches.
"""

import json
import random
import sqlite3

from . import corpus


class SourceSystem:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
              doc_id TEXT PRIMARY KEY, title TEXT, body TEXT, acl TEXT, fact TEXT,
              version INTEGER NOT NULL, deleted INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS log (seq INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, op TEXT);
            """
        )

    # --- read API (what the sync pipeline uses) ---------------------------------------------
    def changes_since(self, cursor: int, limit: int = 10_000) -> tuple[list[tuple[int, str, str]], int]:
        rows = self.db.execute("SELECT seq, doc_id, op FROM log WHERE seq > ? ORDER BY seq LIMIT ?", (cursor, limit)).fetchall()
        return rows, (rows[-1][0] if rows else cursor)

    def get(self, doc_id: str) -> dict | None:
        row = self.db.execute("SELECT doc_id, title, body, acl, fact, version FROM docs WHERE doc_id = ? AND deleted = 0", (doc_id,)).fetchone()
        if not row:
            return None
        return {"doc_id": row[0], "title": row[1], "body": row[2], "acl": json.loads(row[3]), "fact": json.loads(row[4]), "version": row[5]}

    def list_versions(self) -> dict[str, int]:
        """Full listing for reconciliation: every live document and its version."""
        return dict(self.db.execute("SELECT doc_id, version FROM docs WHERE deleted = 0").fetchall())

    def all_docs(self) -> list[dict]:
        return [self.get(d) for d in sorted(self.list_versions())]

    # --- write API (what users of the source system do) -------------------------------------
    def _log(self, doc_id: str, op: str):
        self.db.execute("INSERT INTO log (doc_id, op) VALUES (?, ?)", (doc_id, op))

    def put(self, doc: dict):
        cur = self.db.execute("SELECT version FROM docs WHERE doc_id = ?", (doc["doc_id"],)).fetchone()
        version = (cur[0] + 1) if cur else 1
        self.db.execute(
            "INSERT INTO docs (doc_id, title, body, acl, fact, version, deleted) VALUES (?, ?, ?, ?, ?, ?, 0) "
            "ON CONFLICT(doc_id) DO UPDATE SET title=excluded.title, body=excluded.body, acl=excluded.acl, "
            "fact=excluded.fact, version=excluded.version, deleted=0",
            (doc["doc_id"], doc["title"], doc["body"], json.dumps(doc["acl"]), json.dumps(doc["fact"]), version),
        )
        self._log(doc["doc_id"], "upsert")
        self.db.commit()

    def set_acl(self, doc_id: str, acl: list[str]):
        self.db.execute("UPDATE docs SET acl = ?, version = version + 1 WHERE doc_id = ?", (json.dumps(acl), doc_id))
        self._log(doc_id, "acl")
        self.db.commit()

    def delete(self, doc_id: str, logged: bool = True):
        self.db.execute("UPDATE docs SET deleted = 1, version = version + 1 WHERE doc_id = ?", (doc_id,))
        if logged:
            self._log(doc_id, "delete")
        self.db.commit()

    def seed(self, docs: list[dict]):
        for d in docs:
            self.put(d)

    def mutate_day(self, day: int) -> dict:
        """One day of realistic changes. Returns what happened, for logs and tests."""
        r = random.Random(1000 + day)
        live = sorted(self.list_versions())
        picks = r.sample(live, k=14)
        edited, grown, shrunk, acl_changed = picks[0:4], picks[4:7], picks[7:9], picks[9:12]
        deleted_logged, deleted_silent = picks[12], picks[13]
        for doc_id in edited:  # the fact changes: the old value must disappear from the index
            d = self.get(doc_id)
            kind = d["fact"]["kind"]
            self.put(corpus.make_document(r, doc_id, kind, d["fact"]["entity"], d["fact"]["year"], acl=d["acl"]))
        for doc_id in grown:
            d = self.get(doc_id)
            d["body"] += "\n\n" + " ".join(r.sample(corpus.FILLER[d["fact"]["group"]], k=3))
            self.put(d)
        for doc_id in shrunk:  # fewer chunks than before: the extra old chunks must not survive
            d = self.get(doc_id)
            paras = d["body"].split("\n\n")
            keep = [p for p in paras if d["fact"]["value"] in p or p.startswith("Internal reference")]
            d["body"] = "\n\n".join(keep or paras[:1])
            self.put(d)
        for doc_id in acl_changed:  # permissions only: content is the same
            d = self.get(doc_id)
            self.set_acl(doc_id, [d["fact"]["group"]] if d["acl"] == [corpus.ALL_STAFF] else [corpus.ALL_STAFF])
        self.delete(deleted_logged)
        self.delete(deleted_silent, logged=False)
        new_ids = []
        for i in range(3):
            kind = r.randrange(len(corpus.DOC_TYPES))
            doc_id = f"n{day:02d}{i}"
            self.put(corpus.make_document(r, doc_id, kind, r.choice(corpus.DOC_TYPES[kind][2]), r.choice(corpus.YEARS)))
            new_ids.append(doc_id)
        return {"edited": edited, "grown": grown, "shrunk": shrunk, "acl_changed": acl_changed,
                "deleted": deleted_logged, "deleted_silently": deleted_silent, "added": new_ids}
