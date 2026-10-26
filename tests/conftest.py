import pytest

from rag import corpus, db
from rag.embed import HashEmbedder
from rag.source import SourceSystem
from rag.sync import Syncer


class Env:
    def __init__(self, tmp_path, n_docs: int):
        db.reset_schema()
        self.source = SourceSystem(str(tmp_path / "source.db"))
        self.source.seed(corpus.generate(n_docs=n_docs))
        self.embedder = HashEmbedder()
        self.sync_conn = db.sync_conn()
        self.app_conn = db.app_conn()
        self.syncer = Syncer(self.source, self.embedder, self.sync_conn)

    def close(self):
        self.sync_conn.close()
        self.app_conn.close()


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path, n_docs=150)
    e.syncer.run_incremental()
    yield e
    e.close()
