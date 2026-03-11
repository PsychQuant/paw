# Paw

Swift-native LLM inference server for Apple Silicon. Downloads HuggingFace models and serves them as OpenAI-compatible API.

## Quick Start

```bash
# Build
swift build

# Download a model
swift run Paw pull mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit

# Start server
swift run Paw serve --port 8080 --host 0.0.0.0

# List models
swift run Paw models
```

## Architecture

- CLI: swift-argument-parser (subcommands: serve, pull, models, remove)
- HTTP: Vapor 4
- Inference: mlx-swift-lm (MLXLLM, MLXVLM, MLXLMCommon, MLXEmbedders)

## Key Files

- Package.swift — dependencies and target
- Sources/Paw/Paw.swift — CLI entry point (@main)
- Sources/Paw/Commands/ — CLI subcommands (serve, pull, models, remove)
- Sources/Paw/Engine/ — model management, generation, embedding manager
- Sources/Paw/Routes/ — API endpoint handlers
- Sources/Paw/Types/ — request/response Codable types
- Sources/Paw/PromptCache/ — KV cache management for prompt reuse
- Sources/Paw/Utilities.swift — constants, error types, SSE helpers

## Dependencies

- swift-argument-parser 1.3+ — CLI subcommands
- vapor 4+ — HTTP server, CORS, routing
- mlx-swift-lm (main branch) — MLXLLM, MLXLMCommon, MLXVLM, MLXEmbedders

Note: ArgumentParser property wrappers (@Option, @Flag, @Argument) must be
fully qualified as `ArgumentParser.Option` etc. to avoid ambiguity with Vapor's
ConsoleKit which exports identically-named types.

## API Endpoints

- POST /v1/chat/completions (streaming + non-streaming)
- POST /v1/completions
- POST /v1/embeddings
- GET /v1/models
- GET /health

## Coding Standards

- Swift 5.12+ (Swift 6.x), macOS 14+
- Use actors for thread-safe model state
- Codable types with explicit CodingKeys (snake_case for JSON)
- Follow OpenAI API response format exactly

## References

- references/swift-mlx-server/ — base project
- references/swama/ — advanced features
- references/maclocal-api/ — dual backend

@~/.claude/CLAUDE.md
