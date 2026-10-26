"""The incremental sync must end in exactly the state a from-scratch rebuild produces,
whatever happens on the way: missed events, crashes, replays, bad documents."""

import pytest

from rag import corpus
from rag.chunking import chunk
from rag.db import as_user
from rag.sync import CrashForTest, MassDeleteGuard, snapshot


def rebuild_reference(env):
    """Snapshot of a full rebuild. Destroys the current index, so take other snapshots first."""
    env.syncer.full_rebuild()
    return snapshot(env.sync_conn)


def test_daily_changes_match_a_full_rebuild(env):
    for day in range(1, 4):
        env.source.mutate_day(day)
        env.syncer.run_incremental()
    env.syncer.reconcile()
    incremental = snapshot(env.sync_conn)
    assert incremental == rebuild_reference(env)


def test_deletes_the_feed_missed_are_caught_by_reconcile(env):
    changes = env.source.mutate_day(1)
    env.syncer.run_incremental()
    indexed = {r[0] for r in snapshot(env.sync_conn)}
    assert changes["deleted"] not in indexed           # reported delete: gone
    assert changes["deleted_silently"] in indexed      # never reported: still there
    env.syncer.reconcile()
    assert changes["deleted_silently"] not in {r[0] for r in snapshot(env.sync_conn)}


@pytest.mark.parametrize("crash_after", [0, 1, 5, 9, 13])
def test_crash_mid_run_then_rerun_matches_full_rebuild(env, crash_after):
    env.source.mutate_day(1)
    cursor_before = env.syncer.cursor()
    with pytest.raises(CrashForTest):
        env.syncer.run_incremental(crash_after=crash_after)
    assert env.syncer.cursor() == cursor_before        # the cursor did not move past unfinished work
    env.syncer.run_incremental()                        # the next scheduled run, nothing special
    env.syncer.reconcile()
    after_recovery = snapshot(env.sync_conn)
    assert after_recovery == rebuild_reference(env)


def test_replaying_the_whole_feed_changes_nothing(env):
    """On an up-to-date index, re-applying every event ever emitted is a no-op (at-least-once is safe)."""
    env.source.mutate_day(1)
    env.syncer.run_incremental()
    env.syncer.reconcile()
    before = snapshot(env.sync_conn)
    with env.sync_conn.transaction():
        env.sync_conn.execute("UPDATE sync_state SET cursor = 0")   # e.g. a lost or reset checkpoint
    stats = env.syncer.run_incremental()
    assert set(stats) <= {"noop"}
    assert snapshot(env.sync_conn) == before


def test_permission_change_revokes_access_without_reembedding(env):
    doc = next(d for d in env.source.all_docs() if d["acl"] == [corpus.ALL_STAFF] and d["fact"]["group"] != "legal")
    reader = ["legal", corpus.ALL_STAFF]
    seen = lambda: doc["doc_id"] in {r[0] for r in _visible(env, reader)}  # noqa: E731
    assert seen()
    env.source.set_acl(doc["doc_id"], [doc["fact"]["group"]])
    assert seen()                                       # stale until the sync runs: revocation latency is real
    tokens = env.embedder.tokens
    stats = env.syncer.run_incremental()
    assert stats == {"acl_updated": 1}
    assert env.embedder.tokens == tokens                # nothing re-embedded
    assert not seen()


def test_shrinking_document_leaves_no_stale_chunks(env):
    doc = max(env.source.all_docs(), key=lambda d: len(chunk(d["title"], d["body"])))
    before = len(chunk(doc["title"], doc["body"]))
    doc["body"] = doc["body"].split("\n\n")[0]
    env.source.put(doc)
    env.syncer.run_incremental()
    with env.sync_conn.transaction():
        n = env.sync_conn.execute("SELECT count(*) FROM chunks WHERE doc_id = %s", (doc["doc_id"],)).fetchone()[0]
    assert before > 1 and n == 1


def test_bad_document_is_parked_and_retried_without_blocking_the_rest(env):
    changes = env.source.mutate_day(1)
    bad = changes["edited"][0]
    stats = env.syncer.run_incremental(fail_on={bad})
    assert stats["dead_lettered"] == 1 and sum(stats.values()) > 1
    with env.sync_conn.transaction():
        assert env.sync_conn.execute("SELECT doc_id FROM dead_letters").fetchall() == [(bad,)]
    env.syncer.run_incremental()                        # no new events: the parked document is retried
    with env.sync_conn.transaction():
        assert env.sync_conn.execute("SELECT count(*) FROM dead_letters").fetchone()[0] == 0
    env.syncer.reconcile()
    assert snapshot(env.sync_conn) == rebuild_reference(env)


def test_broken_listing_cannot_wipe_the_index(env, monkeypatch):
    before = snapshot(env.sync_conn)
    monkeypatch.setattr(env.source, "list_versions", lambda: {})   # e.g. the source API returned an empty page
    with pytest.raises(MassDeleteGuard):
        env.syncer.reconcile()
    assert snapshot(env.sync_conn) == before


def _visible(env, groups):
    with as_user(env.app_conn, groups):
        return env.app_conn.execute("SELECT DISTINCT doc_id FROM chunks").fetchall()
