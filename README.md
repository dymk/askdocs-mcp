# AskDocs MCP Server

A Model Context Protocol (MCP) server that provides RAG-powered semantic search over technical documentation PDFs using Ollama.

## Features

- Semantic search with natural language queries
- Multiple PDF documents with page citations
- Docker support with persistent caching
- TOML-based configuration

## Quick Start

**1. Create `askdocs-mcp.toml` in your docs directory:**

```toml
[[doc]]
name = "my_manual"
description = "My Product Manual"
path = "pdf/manual.pdf"
```

**2. Run with Docker:**

```bash
docker run -it --rm --network=host -v ./docs:/docs askdocs-mcp:latest
```

`askdocs-mcp` expects an Ollama server to be running on `http://localhost:11434`.

**3. Directory structure:**

```
docs/
├── askdocs-mcp.toml    # Configuration
├── .askdocs-cache/     # Vector store (auto-created)
└── pdf/
    └── manual.pdf
```

Add `**/.askdocs-cache/**` to your `.gitignore` file.

## Configuration

```toml
# Optional: Configure models
embedding_model = "snowflake-arctic-embed:latest"
llm_model = "qwen3:14b"

[[doc]]
name = "unique_identifier"
description = "Human description"
path = "pdf/document.pdf"
```

**Environment variable:**
- `ASKDOCS_OLLAMA_URL`: Ollama server URL (default: `http://localhost:11434`)

## Available Tools

### `list_docs()`
List all documentation sources.

### `ask_docs(source_name: str, query: str)`
Search documentation with natural language.

### `get_doc_page(source_name: str, page_start: int, page_end: int = None)`
Retrieve full text from specific pages.

## Requirements

Ollama must be running with the required models:

```bash
ollama pull snowflake-arctic-embed:latest
ollama pull qwen3:14b
```

## Building

```bash
# Docker
docker build -t askdocs-mcp:latest .

# Local
uv sync
uv run askdocs-mcp --docs-dir /path/to/docs
```

## License

MIT