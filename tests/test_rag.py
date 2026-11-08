import unittest

from rag import BM25, chunk_text, tokenize


class RagTests(unittest.TestCase):
    def test_tokenize_stems_kazakh_endings(self):
        self.assertEqual(tokenize("кітап кітапта кітаптың"), ["кітап"] * 3)

    def test_chunks_cover_text_and_respect_size(self):
        text = " ".join(f"word{i}" for i in range(500))
        chunks = chunk_text(text, size=200, overlap=40)
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(len(c) <= 200 for c in chunks))
        self.assertTrue(chunks[0].startswith("word0"))
        self.assertTrue(chunks[-1].endswith("word499"))

    def test_chunk_empty(self):
        self.assertEqual(chunk_text("   "), [])

    def test_bm25_ranks_relevant_first(self):
        docs = [
            "cats are small animals that like milk",
            "python is a programming language for data and AI",
            "the weather today is sunny and warm",
        ]
        hits = BM25(docs).top_k("which programming language for AI", k=2)
        self.assertEqual(hits[0][1], 1)

    def test_bm25_no_match(self):
        self.assertEqual(BM25(["apples and pears"]).top_k("quantum"), [])


if __name__ == "__main__":
    unittest.main()
