"""Embedders. The index stores which model produced its vectors; mixing models fails silently.

  HashEmbedder    deterministic, offline (tests and CI need no API key and give the same vectors every run)
  OpenAIEmbedder  text-embedding-3-small at 256 dimensions (the real evaluation)
"""

import hashlib
import math
import os
import re

DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")


def to_pgvector(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


class HashEmbedder:
    name = "hash-256"

    def __init__(self):
        self.tokens = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            words = _TOKEN.findall(t.lower())
            v = [0.0] * DIM
            for f in words + [a + "_" + b for a, b in zip(words, words[1:])]:
                h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
                v[h % DIM] += 1.0 if (h >> 63) else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
            self.tokens += len(words)
        return out


class OpenAIEmbedder:
    name = "text-embedding-3-small@256"
    usd_per_million = 0.02

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()
        self.tokens = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), 128):
            r = self.client.embeddings.create(model="text-embedding-3-small", input=texts[i:i + 128], dimensions=DIM)
            self.tokens += r.usage.total_tokens
            out.extend(d.embedding for d in r.data)
        return out


def get_embedder(name: str | None = None):
    name = name or os.environ.get("EMBEDDER", "hash")
    return OpenAIEmbedder() if name == "openai" else HashEmbedder()
