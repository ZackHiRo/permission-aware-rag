"""Connections, one per role. The query path never uses a role that can see everything."""

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg

HOST = os.environ.get("PGHOST", "localhost")
PORT = os.environ.get("PGPORT", "54329")
DB = os.environ.get("PGDATABASE", "postgres")
ADMIN_URL = os.environ.get("ADMIN_URL", f"postgresql://postgres:postgres@{HOST}:{PORT}/{DB}")
SYNC_URL = os.environ.get("SYNC_URL", f"postgresql://rag_sync:rag_sync@{HOST}:{PORT}/{DB}")
APP_URL = os.environ.get("APP_URL", f"postgresql://rag_app:rag_app@{HOST}:{PORT}/{DB}")

SCHEMA = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"


def reset_schema() -> None:
    """Drop and recreate everything (admin only: it also creates the roles)."""
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute("SET client_min_messages = warning")
        conn.execute(SCHEMA.read_text())


def sync_conn() -> psycopg.Connection:
    return psycopg.connect(SYNC_URL)


def app_conn() -> psycopg.Connection:
    return psycopg.connect(APP_URL)


@contextmanager
def as_user(conn: psycopg.Connection, groups: list[str], iterative_scan: bool = True):
    """A read transaction scoped to one user's groups.

    Groups come from the verified identity (IdP token) on the server, never from the request body.
    set_config(..., true) is transaction-local: it can't leak into the next request on a pooled connection.
    """
    with conn.transaction():
        conn.execute("SELECT set_config('app.groups', %s, true)", (",".join(groups),))
        conn.execute(f"SET LOCAL hnsw.iterative_scan = {'relaxed_order' if iterative_scan else 'off'}")
        yield conn
