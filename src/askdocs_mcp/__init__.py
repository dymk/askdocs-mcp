"""General-purpose AskDocs MCP Server."""

import argparse
import logging
import os
import sys
from pathlib import Path

import requests
from mcp.server.fastmcp import FastMCP

from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    load_config,
    log,
)
from .document_processor import get_pdf_page_content
from .server import AskDocsServer

# Configure logging to go to stderr for MCP compatibility
logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stderr,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Suppress noisy library loggers
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Suppress ChromaDB telemetry output and UV progress
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["UV_NO_PROGRESS"] = "1"

# Initialize MCP server
mcp = FastMCP("AskDocs")

# Global server instance (initialized in main())
server = None


# Register MCP tools
@mcp.tool()
def list_docs() -> str:
    """List all available documentation sources.

    Returns a formatted list of all documentation sources that can be queried,
    including their names and descriptions.

    Returns:
        Formatted string listing all available documentation sources

    Example:
        list_docs()
    """
    if server is None:
        return "Server not initialized."
    return server.list_docs()


@mcp.tool()
def ask_docs(source_name: str, query: str) -> str:
    """Search documentation sources with semantic search.

    Query technical documentation using natural language questions.
    Returns answers with citations to specific pages.

    Args:
        source_name: Name of the documentation source to search. Use list_docs() to see all available sources.
        query: Your question about the documentation

    Returns:
        Answer based on the documentation with source page references

    Example:
        ask_docs("my_manual", "What is the base address of the peripheral?")
        ask_docs("product_spec", "How do I configure the interface?")
    """
    if server is None:
        return "Server not initialized."
    return server.ask_docs(source_name, query)


@mcp.tool()
def get_doc_page(source_name: str, page_start: int, page_end: int = None) -> str:
    """Retrieve full text content of PDF page(s).

    Use this tool when you need to see the complete text content of specific pages
    from search results, including tables and detailed sections.

    Args:
        source_name: Name of the PDF source (e.g., "my_manual", "product_spec")
        page_start: Starting page number (use the page number from search results)
        page_end: Optional ending page number for ranges (default: same as page_start)

    Returns:
        Complete text content extracted from the specified pages

    Example:
        get_doc_page("my_manual", 23, 25)  # Get pages 23-25
        get_doc_page("product_spec", 108)  # Get single page 108
    """
    if server is None:
        return "Server not initialized."
    return get_pdf_page_content(server.doc_sources, source_name, page_start, page_end)


def check_server_connection(server_url: str, timeout: int = 5) -> bool:
    """Check if the Ollama server is reachable.

    Args:
        server_url: The URL of the Ollama server
        timeout: Connection timeout in seconds

    Returns:
        True if server is reachable, False otherwise
    """
    try:
        log(f"Checking connection to Ollama server at {server_url}...")
        response = requests.get(f"{server_url}/api/tags", timeout=timeout)
        response.raise_for_status()
        log("✓ Successfully connected to Ollama server")
        return True
    except requests.exceptions.ConnectionError:
        log(f"✗ Error: Cannot connect to Ollama server at {server_url}")
        log("  Please ensure Ollama is running and accessible at the specified URL")
        return False
    except requests.exceptions.Timeout:
        log(f"✗ Error: Connection to {server_url} timed out after {timeout} seconds")
        return False
    except requests.exceptions.RequestException as e:
        log(f"✗ Error: Failed to connect to Ollama server: {e}")
        return False


def main():
    """Main entry point for the askdocs-mcp CLI"""
    global server

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AskDocs MCP Server - RAG-powered documentation search"
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("/docs"),
        help="Path to documentation directory containing askdocs-mcp.toml (default: /docs)",
    )

    args = parser.parse_args()

    # Load configuration from docs directory
    docs_dir = args.docs_dir
    config_file = docs_dir / "askdocs-mcp.toml"

    try:
        doc_sources, settings = load_config(config_file)
    except FileNotFoundError as e:
        log(f"Error: {e}")
        log(f"\nExpected config file at: {config_file}")
        log("Please create askdocs-mcp.toml in your docs directory")
        log("Example config file format:")
        log(
            f"""
# Optional: Configure Ollama models
embedding_model = "{DEFAULT_EMBEDDING_MODEL}"
llm_model = "{DEFAULT_LLM_MODEL}"

[[doc]]
name = "my_manual"
description = "My Product Manual"
path = "pdf/manual.pdf"

[[doc]]
name = "another_doc"
description = "Another Documentation"
path = "pdf/another.pdf"
"""
        )
        sys.exit(1)
    except (ValueError, Exception) as e:
        log(f"Error loading configuration: {e}")
        sys.exit(1)

    # Check server connectivity before proceeding
    if not check_server_connection(settings["server_url"]):
        log("\nCannot proceed without a working Ollama server connection.")
        log("Please start Ollama or check your server_url configuration.")
        sys.exit(1)

    # Initialize server with configured settings
    log("Initializing AskDocs server...")
    server = AskDocsServer(
        doc_sources=doc_sources,
        embedding_model=settings["embedding_model"],
        llm_model=settings["llm_model"],
        server_url=settings["server_url"],
        cache_dir=docs_dir,
    )

    # Initialize documentation sources
    server.initialize_docs(doc_sources)

    # Start MCP server
    mcp.run()


if __name__ == "__main__":
    main()
