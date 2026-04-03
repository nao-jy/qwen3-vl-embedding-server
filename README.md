# Qwen3-VL-Embedding Server

> OpenAI-compatible multimodal embedding server for Qwen3-VL-Embedding-2B.
> Embed text, images, or both — via a simple REST API.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why This Exists

Qwen3-VL-Embedding is one of the most capable open multimodal embedding models available, but it ships without a ready-to-use server. This project fills that gap — a production-ready FastAPI server with an OpenAI-compatible API, full multimodal support (text + images), and a clean interface for building vision memory and RAG systems on top of local hardware.

---

## Features

| Feature | Status |
|---------|--------|
| Text embeddings (single/batch) | ✅ |
| Image embeddings (base64) | ✅ |
| Image embeddings (URL) | ✅ |
| Image embeddings (local path) | ✅ |
| Text + image together | ✅ |
| OpenAI-compatible `/v1/embeddings` | ✅ |
| Mixed batch support | ✅ |
| Instruction-based embeddings | ✅ |
| FP16 on CUDA | ✅ |

---

## Quick Start

### Requirements

- Python 3.9+
- CUDA-capable GPU (CPU fallback supported, but slow)
- ~4GB VRAM for the 2B model

### Install

```bash
git clone https://github.com/metik2009/qwen3-vl-embedding-server.git
cd qwen3-vl-embedding-server
```

**Option A — direct run (simplest):**
```bash
pip install -r requirements.txt
python server.py
```

**Option B — install as a package (enables the `qvle-server` CLI):**
```bash
pip install -e .
qvle-server --host 0.0.0.0 --port 8000
```

**Option C — with [uv](https://github.com/astral-sh/uv):**
```bash
uv sync
python server.py
```

The model downloads on first run (~60 seconds). Subsequent starts are faster once cached by HuggingFace.

### Verify It's Running

```bash
curl http://localhost:8000/health
```

---

## API Reference

### `POST /v1/embeddings`

**OpenAI compatibility notes:** The `model` field is accepted and echoed back but not used for routing (there is only one model). Only `encoding_format: "float"` is supported; other values return a 400.

#### Text only

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "A cat sitting on a couch"}'
```

#### Batch text

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["text one", "text two", "text three"]}'
```

#### Image via base64

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "What is in this image?",
      "image": {"type": "base64", "data": "<base64-encoded-image>"}
    }
  }'
```

#### Image via URL

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Describe this",
      "image": {"type": "url", "data": "https://example.com/image.png"}
    }
  }'
```

#### Image via local file path

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "What is this?",
      "image": {"type": "path", "data": "/path/to/your/image.png"}
    }
  }'
```

#### Mixed batch (text-only and multimodal in same request)

Use the object form for all items — omit `image` for text-only entries:

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      {"text": "Text-only item"},
      {"text": "Item with image", "image": {"type": "base64", "data": "<base64>"}},
      {"text": "Another text-only item"}
    ]
  }'
```

#### Instruction-based embeddings

Use the `instruction` field to steer embeddings for specific tasks:

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "A sunset over the mountains",
    "instruction": "Represent this for retrieval"
  }'
```

### `GET /health`

Returns model status and device info.

---

## Running Tests

```bash
# Start the server first, then in another terminal:
python test_multimodal.py
```

The test suite covers text-only, base64, URL, local path, mixed batch, error handling, cosine similarity sanity checks, and instruction-based embedding behavior.

---

## Performance

Tested on RTX 5070 Laptop (sm_120 Blackwell, 8GB VRAM):

| Metric | Value |
|--------|-------|
| Model load | ~60s (first run) |
| VRAM usage | ~4GB |
| Text embedding | ~100ms |
| Image + text | ~300ms |

---

## Architecture

```
Client Request
    ↓
FastAPI Server
    ├── /v1/embeddings (POST)
    │   ├── Normalize inputs
    │   ├── Load images (base64/url/path)
    │   ├── Qwen3-VL processor
    │   └── Return embeddings (last-token pooling)
    └── /health (GET)
    ↓
Qwen3-VL-Embedding-2B (GPU)
```

Pooling: **last-token** (`outputs.last_hidden_state[:, -1, :]`), which is the correct method for this model family.

---

## Security Note

The `path` image type allows loading images from the local filesystem. **This server is intended for local use only — do not expose it to the public internet without adding an authentication layer.**

---

## License

MIT — see [LICENSE](LICENSE).
