# Paw — Swift MLX LLM Server Design

**Date:** 2026-03-12
**Status:** Approved

## Overview

Paw is a Swift-native LLM inference server for Apple Silicon. It downloads HuggingFace models and serves them as OpenAI-compatible API endpoints. Zero Python dependency.

## Goals

- **Simple**: `paw pull <model>` then `paw serve` — done
- **Compatible**: Full OpenAI API compatibility (chat completions, embeddings, tool use, vision)
- **Native**: Pure Swift on Apple Silicon via MLX, maximum performance
- **Dynamic**: Auto-detect downloaded models, hot-switchable via API requests

## Non-Goals

- Multi-GPU / distributed inference
- Model fine-tuning
- Organization/admin API endpoints
- Assistants / Threads API (server-side state management)

## Architecture

```
Paw
├── CLI Layer (ArgumentParser)
│   ├── paw serve [--port] [--host] [--model]
│   ├── paw pull <model-id>
│   ├── paw models
│   └── paw remove <model-id>
│
├── API Layer (Vapor)
│   ├── POST /v1/chat/completions
│   ├── POST /v1/completions
│   ├── POST /v1/embeddings
│   ├── GET  /v1/models
│   ├── GET  /v1/models/{model}
│   └── GET  /health
│
├── Inference Engine (mlx-swift-lm)
│   ├── ModelManager    — scan/load/switch models
│   ├── ChatHandler     — chat + tool use + vision
│   └── StreamAdapter   — MLX AsyncStream → SSE
│
└── references/
    ├── swift-mlx-server/   ← fork base (Vapor + MLXLLM)
    ├── swama/              ← advanced features reference
    └── maclocal-api/       ← dual backend reference
```

## Approach

Fork `mzbac/swift-mlx-server` as the foundation. It already provides:
- Vapor HTTP server with routing
- `/v1/chat/completions` (streaming + non-streaming)
- `/v1/models` listing
- MLX model loading via `mlx-swift-lm`
- SSE streaming adapter
- KV cache quantization support

We extend with:
- CLI commands (`pull`, `models`, `remove`) via ArgumentParser
- Full OpenAI response format compliance
- Tool use / function calling support
- Vision (VLM) support via MLXVLM
- Embeddings via MLXEmbedders
- `/v1/completions` endpoint
- Dynamic model scanning and hot-switching

## API Endpoints

### Phase 1 — MVP

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming + non-streaming) |
| `/v1/models` | GET | List available models |
| `/v1/models/{model}` | GET | Single model info |
| `/health` | GET | Health check |

Supported parameters:
- `model`, `messages`, `temperature`, `top_p`, `max_tokens`
- `stream`, `stop`, `presence_penalty`, `frequency_penalty`
- `seed`, `n` (single choice only)

### Phase 2 — Tool Use + Vision

| Feature | Description |
|---------|-------------|
| `tools` | Function calling via chat template injection |
| `tool_choice` | `auto` / `none` / specific function |
| Image input | `image_url` content type (requires VLM model) |
| `logprobs` | Token probability distribution |

### Phase 3 — Embeddings + Completions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/embeddings` | POST | Vector embeddings (MLXEmbedders) |
| `/v1/completions` | POST | Text completion (non-chat format) |

## CLI Design

```bash
# Download a model from HuggingFace
paw pull mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit

# List downloaded models
paw models

# Start server (auto-loads all downloaded models)
paw serve
paw serve --port 8080 --host 0.0.0.0
paw serve --model mlx-community/Qwen3.5-27B  # load specific model only

# Remove a downloaded model
paw remove mlx-community/Qwen3.5-27B
```

## Model Manager

- Scans HuggingFace cache (`~/.cache/huggingface/hub/`) on startup
- Filters for MLX-compatible models (looks for `config.json` with MLX markers)
- Loads default model on startup, lazy-loads others on first request
- Thread-safe via `ModelContainer.perform` (MLX built-in)
- Model specified in API request `model` field selects which model to use

## Dependencies

| Package | Purpose |
|---------|---------|
| `ml-explore/mlx-swift-lm` | LLM/VLM inference + HF download |
| `vapor/vapor` | HTTP server framework |
| `apple/swift-argument-parser` | CLI commands |

## Target Platform

- macOS 15.0+ (Apple Silicon only)
- Swift 6.0+
- Single binary deployment
