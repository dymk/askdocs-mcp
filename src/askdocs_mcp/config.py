"""Configuration loading and data structures."""

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


def log(msg):
    """Print to stderr instead of stdout for MCP compatibility"""
    print(msg, file=sys.stderr)


# Default models and server
DEFAULT_EMBEDDING_MODEL = "snowflake-arctic-embed:latest"
DEFAULT_LLM_MODEL = "qwen3:14b"
DEFAULT_SERVER_URL = "http://localhost:11434"

# Retrieval settings
RETRIEVAL_K = 12  # Number of chunks to retrieve (higher = more context)
USE_MMR = True  # Use Maximum Marginal Relevance for diverse results

# Chunking settings (larger chunks = more context, smaller = more precise)
CHUNK_SIZE = 3000  # Characters per chunk
CHUNK_OVERLAP = 600  # Overlap between chunks


@dataclass
class DocSource:
    """Configuration for a PDF documentation source"""

    name: str  # Tool name (will be prefixed with 'search_')
    description: str  # Tool description
    path: Path  # Path to PDF file


@dataclass
class InitializedDoc:
    """Runtime data for an initialized documentation source"""

    retriever: any  # LangChain retriever
    chain: any  # RAG chain
    description: str  # Human-readable description


def load_config(config_path: Path) -> tuple[list[DocSource], dict]:
    """Load documentation sources from a TOML configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Tuple of (list of DocSource objects, config dict with settings)

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    log(f"Loading configuration from: {config_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    if "doc" not in config:
        raise ValueError("Configuration file must contain [[doc]] entries")

    doc_sources = []
    config_dir = config_path.parent

    for doc_config in config["doc"]:
        # Validate required fields
        if "name" not in doc_config:
            raise ValueError("Each [[doc]] entry must have a 'name' field")
        if "description" not in doc_config:
            raise ValueError(
                f"[[doc]] entry '{doc_config['name']}' missing 'description' field"
            )
        if "path" not in doc_config:
            raise ValueError(
                f"[[doc]] entry '{doc_config['name']}' missing 'path' field"
            )

        # Resolve path relative to config file directory
        doc_path = Path(doc_config["path"])
        if not doc_path.is_absolute():
            doc_path = (config_dir / doc_path).resolve()

        doc_sources.append(
            DocSource(
                name=doc_config["name"],
                description=doc_config["description"],
                path=doc_path,
            )
        )

    # Extract optional model and server configuration
    # Priority: Environment variable > hardcoded default
    server_url = os.environ.get("ASKDOCS_OLLAMA_URL", DEFAULT_SERVER_URL)

    settings = {
        "embedding_model": config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "llm_model": config.get("llm_model", DEFAULT_LLM_MODEL),
        "server_url": server_url,
    }

    log(f"✓ Loaded {len(doc_sources)} documentation source(s) from config")
    log(f"✓ Using Ollama server: {settings['server_url']}")
    log(f"✓ Using embedding model: {settings['embedding_model']}")
    log(f"✓ Using LLM model: {settings['llm_model']}")

    return doc_sources, settings
