"""Tiny retrieval engine: text chunking + BM25 ranking. No dependencies."""
import math
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text, stem_len=5):
    """Lowercase words, truncated to stem_len characters.

    Truncation is a very cheap stemmer. It helps with languages like Kazakh,
    where one word has many endings (кітап, кітапта, кітаптың -> "кітап").
    """
    return [t[:stem_len] for t in TOKEN_RE.findall(text.lower())]


def chunk_text(text, size=800, overlap=150):
    """Split text into overlapping chunks, preferring to cut at spaces."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


class BM25:
    """Okapi BM25 ranking over a list of text documents."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.n = len(docs)
        self.avgdl = sum(len(t) for t in self.tokens) / self.n if self.n else 0
        self.tf = [Counter(t) for t in self.tokens]
        self.df = Counter()
        for t in self.tokens:
            self.df.update(set(t))

    def _idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n - n + 0.5) / (n + 0.5))

    def score(self, query_tokens, i):
        dl = len(self.tokens[i]) or 1
        total = 0.0
        for term in query_tokens:
            f = self.tf[i].get(term, 0)
            if not f:
                continue
            norm = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += self._idf(term) * f * (self.k1 + 1) / norm
        return total

    def top_k(self, query, k=4):
        """Return [(score, doc_index)] for the k best matches (score > 0)."""
        q = tokenize(query)
        scored = sorted(((self.score(q, i), i) for i in range(self.n)), reverse=True)
        return [(s, i) for s, i in scored[:k] if s > 0]


def load_chunks(folder):
    """Read every .txt / .md file in folder -> list of (source_name, chunk)."""
    out = []
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            out.extend((path.name, c) for c in chunk_text(text))
    return out
