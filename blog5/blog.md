# Build a Local RAG Assistant

Large language models are good at writing answers, but their built-in knowledge may not include your notes, documentation, or the latest version of a project. Retrieval-Augmented Generation (RAG) adds a search step before generation: find relevant excerpts in a document, then give those excerpts to the model as context.

This lesson builds that pipeline with a small command-line program. The input is a UTF-8 text or Markdown file, Ollama provides both the embeddings and the final answer, and the Python standard library handles the rest.

## The pipeline

The program performs four steps:

1. Read the document.
2. Split it into overlapping word-based chunks.
3. Create an embedding for every chunk and store the vectors in memory.
4. Embed a question, rank chunks by cosine similarity, and place the best excerpts in the prompt sent to the chat model.

The model does not receive the entire document automatically. It receives only the retrieved excerpts, which keeps the prompt focused and makes the source of the answer visible in the code.

## Setup

Install Ollama, then pull a chat model and an embedding model:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
ollama serve
```

Run the example with a text or Markdown file:

```bash
python3 rag.py notes.md
```

The `--question` option runs one query without entering the interactive loop:

```bash
python3 rag.py notes.md --question "What are the main recommendations?"
```

## Chunking

Language models have limited context windows, and a whole document may be too large or too noisy to send with every question. `split_into_chunks` turns the document into word-based pieces. By default, each piece contains 180 words and the next piece repeats 30 words from the previous one.

The overlap helps preserve meaning when an important sentence crosses a chunk boundary. Chunking by words is intentionally easy to inspect for this lesson; production systems may use token-aware or structure-aware splitting.

## Embeddings and retrieval

An embedding is a list of numbers representing the meaning of a piece of text. Texts with related meanings tend to have vectors that point in similar directions. The program asks Ollama's `/api/embed` endpoint for one vector per chunk and one vector for the question.

`cosine_similarity` compares those vectors. The index is sorted by the score, and the top three chunks are selected by default. This is an in-memory index, so the document is embedded again each time the program starts; there is no database or persistent cache in this example.

## Building the context

The selected chunks are joined with separators and inserted into a prompt:

```text
Context:
<retrieved chunk 1>
---
<retrieved chunk 2>

Question: <the user's question>
Answer:
```

The instruction tells the chat model to use only that context and to say when the document does not contain an answer. This is a useful guardrail, but it is not a guarantee that a model will never make a mistake. Retrieval quality depends on the document, chunk settings, embedding model, and question.

## What this example does not do

This lesson supports UTF-8 text and Markdown files. It does not extract text from PDFs or Word documents, persist an index, cite exact source locations, or evaluate answer quality. Those are natural next steps once the basic RAG loop is clear.

The main idea is small enough to follow end to end: retrieve relevant text first, then generate an answer grounded in that retrieved context.
