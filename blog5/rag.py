#!/usr/bin/env python3
"""A small local Retrieval-Augmented Generation (RAG) assistant."""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "llama3.2:1b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


@dataclass
class Chunk:
    """A piece of the source document and its embedding."""

    text: str
    embedding: List[float]


def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Split text into overlapping word-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[start : start + chunk_size]) for start in range(0, len(words), step)]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compare two vectors using cosine similarity."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    if not left_length or not right_length:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_length * right_length)


class Ollama:
    """Minimal client for Ollama's local REST API."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL):
        self.base_url = base_url.rstrip("/")

    def _post(self, endpoint: str, payload: dict) -> dict:
        request = Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Is 'ollama serve' running? ({error})"
            ) from error

    def embed(self, model: str, text: str) -> List[float]:
        response = self._post("/api/embed", {"model": model, "input": text})
        embeddings = response.get("embeddings")
        if not embeddings:
            raise RuntimeError(f"Ollama returned no embedding: {response}")
        return embeddings[0]

    def chat(self, model: str, prompt: str) -> str:
        response = self._post(
            "/api/generate",
            {"model": model, "prompt": prompt, "stream": False},
        )
        answer = response.get("response")
        if answer is None:
            raise RuntimeError(f"Ollama returned no response: {response}")
        return answer.strip()


def build_index(text: str, client: Ollama, embedding_model: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """Chunk a document and create one embedding for each chunk."""
    chunks = split_into_chunks(text, chunk_size, overlap)
    return [Chunk(chunk, client.embed(embedding_model, chunk)) for chunk in chunks]


def retrieve(question: str, index: Sequence[Chunk], client: Ollama, embedding_model: str, top_k: int) -> List[Chunk]:
    """Return the chunks most similar to the question."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    question_embedding = client.embed(embedding_model, question)
    ranked = sorted(
        index,
        key=lambda chunk: cosine_similarity(question_embedding, chunk.embedding),
        reverse=True,
    )
    return list(ranked[:top_k])


def make_prompt(question: str, matches: Sequence[Chunk]) -> str:
    """Put retrieved excerpts into the model's context."""
    context = "\n\n---\n\n".join(chunk.text for chunk in matches)
    return f"""You answer questions using only the context below.
If the context does not contain the answer, say you do not know based on this document.
Keep the answer concise and do not invent facts.

Context:
{context}

Question: {question}
Answer:"""


def answer_question(question: str, index: Sequence[Chunk], client: Ollama, chat_model: str, embedding_model: str, top_k: int) -> str:
    matches = retrieve(question, index, client, embedding_model, top_k)
    return client.chat(chat_model, make_prompt(question, matches))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions about a local text document.")
    parser.add_argument("document", type=Path, help="path to a .txt or Markdown document")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL, help="Ollama chat model")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model")
    parser.add_argument("--top-k", type=int, default=3, help="number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=180, help="words per chunk")
    parser.add_argument("--overlap", type=int, default=30, help="overlapping words between chunks")
    parser.add_argument("--question", help="ask one question and exit")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama server URL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.document.is_file():
        raise SystemExit(f"Document not found: {args.document}")

    text = args.document.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit("The document is empty.")

    client = Ollama(args.ollama_url)
    print(f"Indexing {args.document} with {args.embedding_model}...")
    index = build_index(text, client, args.embedding_model, args.chunk_size, args.overlap)
    print(f"Indexed {len(index)} chunks. Type 'quit' to exit.")

    questions = [args.question] if args.question else iter(lambda: input("\nYou: ").strip(), "quit")
    try:
        for question in questions:
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            print(f"\nAssistant: {answer_question(question, index, client, args.model, args.embedding_model, args.top_k)}")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
    except RuntimeError as error:
        print(f"\nError: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
