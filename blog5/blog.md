# Build a Local RAG Assistant with Python and Ollama

Retrieval-Augmented Generation (RAG) is one of the most useful patterns for building applications with large language models. Instead of asking a model to answer from memory alone, we first search a document for relevant information and then include that information in the prompt.

In this tutorial, we'll build a small command-line RAG assistant. It will read a text or Markdown document, split it into chunks, create local embeddings with Ollama, retrieve the most relevant chunks, and use those excerpts to answer a question.

📦 **[View the complete Blog 5 code on GitHub](https://github.com/ani0x53/llm-blog-series/tree/main/blog5)**

## What Is RAG?

RAG combines two capabilities:

1. **Retrieval**: Find useful pieces of information from a document.
2. **Generation**: Ask a language model to write an answer using those pieces.

This is helpful when the information lives in your own notes, documentation, or project files. The model does not need to memorize the document. The application retrieves relevant text at question time and sends that text as context.

### The RAG Pipeline

Our example follows four simple steps:

1. Read a UTF-8 text or Markdown document
2. Split the document into overlapping chunks
3. Embed the chunks and the user's question
4. Retrieve the closest chunks and pass them to a chat model

The complete implementation is in [`rag.py`](https://github.com/ani0x53/llm-blog-series/blob/main/blog5/rag.py).

## Prerequisites

You'll need:

- Python 3.8 or higher
- [Ollama](https://ollama.com/download)
- A chat model
- An embedding model

Install Ollama, then download the models used by this example:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
ollama serve
```

The Python code uses only the standard library, so there are no third-party packages to install:

```bash
cd blog5
python3 rag.py --help
```

## Running the Assistant

Pass a text or Markdown file as the first argument:

```bash
python3 rag.py notes.md
```

The program embeds the document once at startup and then opens an interactive prompt:

```text
Indexing notes.md with nomic-embed-text...
Indexed 4 chunks. Type 'quit' to exit.

You: What are the main recommendations?
Assistant: ...
```

You can also ask one question and exit:

```bash
python3 rag.py notes.md \
  --question "What are the main recommendations?"
```

## Step 1: Splitting the Document into Chunks

Sending an entire document with every question can make prompts unnecessarily large. The function `split_into_chunks` divides the document into smaller word-based pieces.

```python
def split_into_chunks(text, chunk_size=800, overlap=120):
    words = text.split()
    step = chunk_size - overlap
    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
    ]
```

The overlap is important. If a sentence crosses a chunk boundary, repeating some words in the next chunk gives both pieces a little surrounding context.

This tutorial uses word-based splitting because it is easy to understand. Production systems may use token-aware splitting or preserve structure such as headings and paragraphs.

## Step 2: Creating Embeddings

An embedding is a list of numbers that represents the meaning of text. Related pieces of text tend to have vectors pointing in similar directions.

The example sends each chunk to Ollama's local embedding endpoint:

```python
def embed(self, model, text):
    response = self._post(
        "/api/embed",
        {"model": model, "input": text},
    )
    return response["embeddings"][0]
```

When the program starts, it creates one embedding for every chunk and keeps those vectors in memory. There is no database or persistent index in this example, so the document is embedded again each time the program starts.

## Step 3: Retrieving Relevant Chunks

When a user asks a question, the question is embedded using the same embedding model. We then compare the question vector with every document vector using cosine similarity.

```python
def cosine_similarity(left, right):
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / (
        left_length * right_length
    )
```

The chunks are sorted by their similarity score, and the top three are selected by default:

```python
ranked = sorted(
    index,
    key=lambda chunk: cosine_similarity(question_embedding, chunk.embedding),
    reverse=True,
)
matches = ranked[:top_k]
```

This is the retrieval part of RAG. The application is choosing which pieces of the document should be shown to the language model.

## Step 4: Building the Model Context

The retrieved chunks are joined together and inserted into a prompt:

```text
You answer questions using only the context below.
If the context does not contain the answer, say you do not know based on this document.

Context:
<retrieved chunk 1>
---
<retrieved chunk 2>

Question: <the user's question>
Answer:
```

Finally, the prompt is sent to the Ollama chat model. The model receives the relevant excerpts rather than the whole document.

This instruction is a useful guardrail, but it is not a guarantee that the model will always be correct. Answer quality depends on the document, the chunk size, the embedding model, the retrieved results, and the question itself.

## Useful Options

The command-line program exposes a few settings so you can experiment:

```bash
python3 rag.py notes.md \
  --chunk-size 120 \
  --overlap 20 \
  --top-k 5 \
  --model llama3.2:1b \
  --embedding-model nomic-embed-text
```

- `--chunk-size`: Number of words in each chunk
- `--overlap`: Number of repeated words between chunks
- `--top-k`: Number of chunks added to the prompt
- `--model`: Ollama chat model
- `--embedding-model`: Ollama embedding model

Smaller chunks can provide more focused excerpts, while larger chunks preserve more surrounding context. Increasing `top-k` provides more information to the chat model but also makes the prompt longer.

## What This Example Does Not Do

This implementation intentionally keeps the architecture small. It supports UTF-8 text and Markdown files, but it does not:

- Extract text from PDFs or Word documents
- Persist embeddings in a database
- Cite exact page or line locations
- Re-rank results with a second model
- Evaluate answer quality automatically

Those would be useful next steps for a larger application. For now, the code shows the complete RAG loop in one file: retrieve relevant text first, then generate an answer grounded in that context.

## Conclusion

RAG does not require a large framework to understand. With a document, an embedding model, a similarity function, and a chat model, we can build a useful local assistant in a few small steps.

The main idea is simple: **retrieve before you generate**. Once this foundation is clear, you can add persistent indexes, better document loaders, citations, hybrid search, or a web interface.
