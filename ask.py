"""Ask questions about your own documents (retrieval-augmented generation)."""
import argparse

import llm
from rag import BM25, load_chunks

SYSTEM = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite passages like [1] or [2]. If the answer is not in the context, say "
    "that you could not find it. Answer in the language of the question."
)


def build_prompt(question, passages):
    context = "\n\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(passages, 1))
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(question, chunks, index, k=4):
    hits = index.top_k(question, k)
    if not hits:
        return "No relevant passages found in the documents.", []
    passages = [chunks[i] for _, i in hits]
    return llm.ask(build_prompt(question, passages), system=SYSTEM), passages


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="?", help="omit for interactive mode")
    ap.add_argument("--docs", default="docs", help="folder with .txt/.md files")
    ap.add_argument("-k", type=int, default=4, help="passages to retrieve")
    args = ap.parse_args()

    chunks = load_chunks(args.docs)
    if not chunks:
        raise SystemExit(f"No .txt/.md files found in '{args.docs}'.")
    index = BM25([c for _, c in chunks])
    print(f"Indexed {len(chunks)} chunks from '{args.docs}'.")

    def run(q):
        try:
            text, passages = answer(q, chunks, index, args.k)
        except llm.LLMError as e:
            print("Error:", e)
            return
        print("\n" + text + "\n")
        for i, (src, _) in enumerate(passages, 1):
            print(f"  [{i}] {src}")

    if args.question:
        run(args.question)
    else:
        while True:
            q = input("\nQuestion (empty to quit): ").strip()
            if not q:
                break
            run(q)


if __name__ == "__main__":
    main()
