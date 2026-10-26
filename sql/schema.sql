-- Permission-aware RAG index. Run as a superuser (it creates roles); the app never connects as one.
--
-- Two roles:
--   rag_sync  writes the index (sync pipeline). Sees everything, through an explicit policy.
--   rag_app   serves queries. Sees only rows whose ACL overlaps the groups set for the current
--             transaction (SET LOCAL app.groups = 'claims,all_staff'). No groups set → no rows.
-- Superusers and BYPASSRLS roles skip RLS entirely, so neither role is one. FORCE ROW LEVEL SECURITY
-- also applies the policies to the table owner.

CREATE EXTENSION IF NOT EXISTS vector;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_sync') THEN CREATE ROLE rag_sync LOGIN PASSWORD 'rag_sync'; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_app') THEN CREATE ROLE rag_app LOGIN PASSWORD 'rag_app'; END IF;
END $$;

DROP TABLE IF EXISTS chunks, documents, sync_state, sync_runs, dead_letters CASCADE;

-- One row per indexed source document (its latest indexed version).
CREATE TABLE documents (
  doc_id          text PRIMARY KEY,
  title           text NOT NULL,
  source_version  bigint NOT NULL,          -- version reported by the source system
  content_hash    text NOT NULL,            -- sha256 of title + body: unchanged content is never re-embedded
  acl             text[] NOT NULL,          -- groups allowed to read it, mirrored from the source
  indexed_at      timestamptz NOT NULL DEFAULT now()
);

-- Chunks carry their own copy of the ACL so the filter runs inside the search, on both legs.
CREATE TABLE chunks (
  chunk_id   text PRIMARY KEY,              -- deterministic: doc_id:content_hash[:12]:idx (same content → same ids)
  doc_id     text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  idx        int  NOT NULL,
  body       text NOT NULL,
  acl        text[] NOT NULL,
  embedding  vector(256) NOT NULL,
  tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
);
CREATE INDEX chunks_doc_idx ON chunks (doc_id);
CREATE INDEX chunks_acl_idx ON chunks USING gin (acl);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

-- Sync bookkeeping: the cursor only moves after every change it covers is applied or parked.
CREATE TABLE sync_state (
  source      text PRIMARY KEY,
  cursor      bigint NOT NULL,
  updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE sync_runs (
  run_id        bigserial PRIMARY KEY,
  mode          text NOT NULL,              -- incremental | reconcile
  started_at    timestamptz NOT NULL DEFAULT now(),
  finished_at   timestamptz,
  status        text NOT NULL DEFAULT 'running',  -- running | ok | failed (a crashed run stays 'running')
  cursor_from   bigint,
  cursor_to     bigint,
  stats         jsonb NOT NULL DEFAULT '{}'
);
CREATE TABLE dead_letters (
  doc_id      text PRIMARY KEY,
  error       text NOT NULL,
  attempts    int NOT NULL DEFAULT 1,
  last_seen   timestamptz NOT NULL DEFAULT now()
);

-- Row-level security ------------------------------------------------------------
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;

-- Fail closed: an unset or empty app.groups gives NULL, and NULL never passes the policy.
CREATE POLICY app_read_documents ON documents FOR SELECT TO rag_app
  USING (acl && string_to_array(nullif(current_setting('app.groups', true), ''), ','));
CREATE POLICY app_read_chunks ON chunks FOR SELECT TO rag_app
  USING (acl && string_to_array(nullif(current_setting('app.groups', true), ''), ','));

CREATE POLICY sync_all_documents ON documents TO rag_sync USING (true) WITH CHECK (true);
CREATE POLICY sync_all_chunks ON chunks TO rag_sync USING (true) WITH CHECK (true);

GRANT USAGE ON SCHEMA public TO rag_app, rag_sync;
GRANT SELECT ON documents, chunks TO rag_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON documents, chunks, sync_state, sync_runs, dead_letters TO rag_sync;
GRANT USAGE ON SEQUENCE sync_runs_run_id_seq TO rag_sync;
