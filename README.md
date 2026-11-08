# Document Q&A (RAG from scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dos1k77/doc-qa-rag/blob/main/demo.ipynb)

Ask questions about your own `.txt` / `.md` files. The program finds the most
relevant passages and asks an LLM (Google Gemini) to answer **only from those
passages**, with citations like `[1]`.

Retrieval is implemented from scratch – chunking + **BM25 ranking** in pure
Python – so you can see exactly how RAG works. No libraries to install.

## How it works

```
docs/*.md -> chunk_text() -> BM25 index
question  -> top_k() -> best passages -> prompt -> Gemini -> answer + sources
```

- `rag.py` – tokenizer, chunking, BM25 (fully unit-tested, runs offline)
- `ask.py` – command-line app and prompt building
- `llm.py` – small Gemini API client

A small stemmer (cutting words to 5 characters) helps with Kazakh, where one
word has many endings.

## Setup

1. Get a free API key at <https://aistudio.google.com/apikey>.
2. Copy `.env.example` to `.env` and paste the key.
3. Put your `.txt` / `.md` files into `docs/` (a sample is included).

```bash
python ask.py "What is RAG?"
python ask.py "RAG деген не?"
python ask.py            # interactive mode
```

## Tests

```bash
python -m unittest -v
```

## Ideas for improvement

- PDF support
- Embedding-based search (semantic) combined with BM25
- Show the matching text next to each source
- Simple web interface

> Never commit your `.env` file (it is in `.gitignore`).
