# OpenRAG

![OpenRAG Screenshot](screenshot/screenshot.png)

A lightweight, open-source Retrieval-Augmented Generation (RAG) library that you can drop your own documents into and get a fully working Q&A app — with a browser UI and a REST API — out of the box.

Built on local [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeddings, hybrid retrieval with BM25 + vector search, and cross-encoder reranking. Model-agnostic by design — bring any LLM you like.

No cloud embedding costs. No vendor lock-in for vector search. Just your documents, locally indexed, intelligently queried.

---

## Features

- **Multi-format ingestion** — supports `.txt` and `.pdf` source documents
- **Local embeddings** — uses `all-MiniLM-L6-v2`, runs entirely on your machine
- **Hybrid retrieval** — combines dense vector search and BM25 keyword search, fused with Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` reranks candidates for higher precision before sending to the LLM
- **Persistent vector store** — embeddings saved to disk via ChromaDB, no re-processing on restart
- **Model-agnostic generation** — pass any callable as your LLM; built-in support for Gemini, OpenAI, and Anthropic via `config.ini`
- **Browser UI** — reactive chat interface, no page reloads
- **REST API** — query your RAG programmatically via `POST /ask`
- **Plug and play** — configure the app name, source material, system prompt, and LLM via a single `config.ini` file

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AryvLabs/OpenRAG.git
cd OpenRAG
```

### 2. Install dependencies

For ingestion (first-time setup):
```bash
pip install -r requirements-ingest.txt
```

For running the app only (after embeddings are generated):
```bash
pip install -r requirements.txt
```

### 3. Configure your RAG instance

Edit `config.ini` in the project root:

```ini
[rag]
name = TestRAG
source = Wuthering Heights by Emily Brontë

[llm]
provider = gemini
model = gemini-2.0-flash
api_key_env = GEMINI_API_KEY

[prompt]
system = You are a literary expert. Your sole purpose is to answer questions strictly using the text provided in the reference passage...
```

The `name` and `source` fields appear in the browser UI header and API responses. The `system` prompt controls how the LLM behaves — change it to suit your domain. The `[llm]` section controls which model is used for generation (see [LLM Configuration](#llm-configuration) below).

### 4. Add your source documents

Place your `corpus.txt` or `corpus.pdf` into the `src/corpus/` directory.

### 5. Generate embeddings

Run `embed.py` from the `src/` directory:

```bash
# For plain text
cd src
python3 embed.py txt

# For PDF
cd src
python3 embed.py pdf
```

This populates `src/db/` with vector data. Re-run whenever you update the corpus.

### 6. Set your API key

Set the environment variable named in `api_key_env`:

```bash
# macOS / Linux
export LLM_API_KEY="your_api_key_here"

# Windows (Command Prompt)
set LLM_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:LLM_API_KEY="your_api_key_here"
```

### 7. Start the app

```bash
python3 app.py
```

Open `http://localhost:5000` for the UI, or use the API directly.

---

## LLM Configuration

The `[llm]` section in `config.ini` controls which model powers generation. Three providers are supported out of the box:

**Google Gemini**
```ini
[llm]
provider = gemini
model = gemini-2.0-flash
api_key_env = LLM_API_KEY
```

**OpenAI**
```ini
[llm]
provider = openai
model = gpt-4o
api_key_env = LLM_API_KEY
```

**Anthropic**
```ini
[llm]
provider = anthropic
model = claude-opus-4-5
api_key_env = LLM_API_KEY
```

`api_key_env` is the name of the environment variable that holds your API key — not the key itself.

### Using a custom LLM

The generation library is model-agnostic. `generateResponse` accepts any callable with the signature `(prompt: str) -> str`, so you can wire in any model — a local Ollama instance, a custom API, a mock for testing — by bypassing `build_llm()` entirely:

```python
from src.generate import generateResponse

# Example: local Ollama
import requests

def my_llm(prompt: str) -> str:
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
    })
    return res.json()["response"]

answer = generateResponse("Who is Heathcliff?", my_llm)
```

---

## API Reference

### POST /ask

Run a RAG query programmatically.

**Request**
```json
{
  "query": "Who is Heathcliff?"
}
```

**Response**
```json
{
  "answer": "Heathcliff is a foundling taken in by Mr Earnshaw...",
  "rag": "TestRAG",
  "source": "Wuthering Heights by Emily Brontë"
}
```

**Error response**
```json
{
  "error": "Missing or empty 'query' field."
}
```

### GET /health

Liveness check. Useful for cloud platform health check configuration.

```json
{
  "status": "ok",
  "rag": "TestRAG",
  "source": "Wuthering Heights by Emily Brontë"
}
```

---

## Retrieval Pipeline

```
query
  ├── dense vector search (all-MiniLM-L6-v2)  ─┐
  └── BM25 keyword search                      ─┴─ RRF fusion → cross-encoder rerank → top-k chunks → LLM
```

1. The query is embedded with the same model used at ingestion time and compared against stored vectors in ChromaDB.
2. BM25 scores the same query against the full corpus stored in ChromaDB.
3. Both ranked lists are merged using Reciprocal Rank Fusion.
4. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks the fused candidates by jointly scoring each `(query, document)` pair.
5. The top chunks are passed to the configured LLM with the system prompt from `config.ini`.

---

## Deploying to the Cloud

Use `requirements.txt` for deployment — it excludes the ingestion-only dependencies since embeddings are already stored in ChromaDB.

```bash
pip install -r requirements.txt
gunicorn app:app
```

Set `LLM_API_KEY` as an environment variable in your cloud platform's settings.

---

## Configuration Reference

| File | Key | Description |
|---|---|---|
| `config.ini` | `rag.name` | Display name shown in the UI and API responses |
| `config.ini` | `rag.source` | Source material label shown in the UI and API responses |
| `config.ini` | `llm.provider` | LLM provider to use (`gemini`, `openai`, `anthropic`) |
| `config.ini` | `llm.model` | Model name as expected by the provider's SDK |
| `config.ini` | `llm.api_key_env` | Name of the environment variable holding the API key |
| `config.ini` | `prompt.system` | System prompt sent to the LLM on every query |
| Environment | `LLM_API_KEY` | API key for the configured LLM provider |

---

## Contributing

Contributions are welcome. Feel free to open PRs with bug fixes and feature suggestions.

Found a bug or have a feature request? [Open an issue](https://github.com/AryvLabs/OpenRAG/issues) and include:
- A clear description of the problem or suggestion
- Steps to reproduce (for bugs)
- Your environment details (OS, Python version)

### Roadmap
- Streaming API responses
- Reranker model configurability via `config.ini`

---

## For Enterprise

If you're looking to build a RAG or other AI feature for your enterprise, check us out [here](https://aryvlabs.com). Our team can help you build your next production-ready AI workflow.
