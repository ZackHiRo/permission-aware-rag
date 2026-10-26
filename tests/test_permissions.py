"""Permissions are enforced by the database, inside the search, and fail closed."""

import psycopg
import pytest

from rag import corpus, retrieve
from rag.db import as_user


def visible(conn, groups):
    with as_user(conn, groups):
        return conn.execute("SELECT doc_id, acl FROM chunks").fetchall()


def test_no_identity_sees_nothing(env):
    assert visible(env.app_conn, []) == []
    with env.app_conn.transaction():  # app.groups never set at all
        assert env.app_conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


@pytest.mark.parametrize("group", corpus.GROUPS)
def test_each_group_sees_exactly_what_its_acls_allow(env, group):
    rows = visible(env.app_conn, [group, corpus.ALL_STAFF])
    assert rows and all({group, corpus.ALL_STAFF} & set(acl) for _, acl in rows)
    expected = {d["doc_id"] for d in env.source.all_docs() if {group, corpus.ALL_STAFF} & set(d["acl"])}
    assert {doc_id for doc_id, _ in rows} == expected


def test_scope_does_not_survive_the_transaction(env):
    """A pooled connection reused for the next request must not keep the previous user's groups."""
    assert visible(env.app_conn, ["hr"])
    with env.app_conn.transaction():
        assert env.app_conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


def test_query_role_cannot_write(env):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with env.app_conn.transaction():
            env.app_conn.execute("DELETE FROM chunks")


def test_restricted_question_from_another_department_retrieves_nothing_restricted(env):
    hr_only = [d for d in env.source.all_docs() if d["acl"] == ["hr"]]
    assert hr_only
    for d in hr_only[:5]:
        for mode in ("dense", "hybrid"):
            hits = retrieve.retrieve(env.app_conn, env.embedder, ["claims", corpus.ALL_STAFF], d["fact"]["question"], mode)
            assert all({"claims", corpus.ALL_STAFF} & set(h["acl"]) for h in hits)
            assert not any("CANARY" in h["body"] and h["doc_id"] == d["doc_id"] for h in hits)
