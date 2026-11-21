# Blog 5: Build a Local RAG Assistant

A small command-line Retrieval-Augmented Generation (RAG) example. It reads a text or Markdown document, splits it into overlapping chunks, embeds those chunks with Ollama, retrieves the most relevant chunks for each question, and sends those excerpts to a local chat model.

## Quick Start

Install and start [Ollama](https://ollama.com/download), then download the two models:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
ollama serve
```

Run from this directory:

```bash
python3 rag.py path/to/notes.md
```

Ask one question and exit:

```bash
python3 rag.py path/to/notes.md --question "What is this document about?"
```

Type `quit` or `exit` to leave interactive mode. Use `--model`, `--embedding-model`, `--top-k`, `--chunk-size`, and `--overlap` to experiment with the pipeline.

## Files

- `rag.py` - the complete local RAG example
- `requirements.txt` - dependency note; the Python code uses only the standard library
- `blog.md` - the tutorial and explanation

## Supported input

The example reads UTF-8 text files, including Markdown. It does not parse PDFs, Word documents, or other binary formats.

## Troubleshooting

- **Could not reach Ollama**: make sure `ollama serve` is running.
- **Model errors**: pull both models shown above, or pass different model names with the command-line options.
- **Slow indexing**: each document chunk requires one embedding request, so larger documents take longer.
