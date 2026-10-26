"""Paragraph-packing chunker. Each chunk starts with the document title, so a chunk that says
"the threshold is CAD 12,500" still says which product and year it belongs to."""

MAX_WORDS = 90


def chunk(title: str, body: str, max_words: int = MAX_WORDS) -> list[str]:
    chunks, cur = [], []
    for para in [p.strip() for p in body.split("\n\n") if p.strip()]:
        if cur and len(" ".join(cur).split()) + len(para.split()) > max_words:
            chunks.append(cur)
            cur = []
        cur.append(para)
    if cur:
        chunks.append(cur)
    return [f"{title}\n\n" + "\n\n".join(c) for c in chunks]
