"""AskDocs MCP Server implementation."""

import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm

from .config import DocSource, InitializedDoc, log
from .document_processor import (
    create_retriever_from_vectorstore,
    create_text_splitter,
    create_vectorstore_with_progress,
    format_docs,
    load_document_source,
    perform_search,
)


def _source_is_newer_than_vectorstore(source: DocSource, complete_marker: Path) -> bool:
    """Check if source PDF has been modified since vector store was created"""
    if not source.path.exists():
        return False

    vectorstore_time = complete_marker.stat().st_mtime
    source_time = source.path.stat().st_mtime
    return source_time > vectorstore_time


class AskDocsServer:
    """Encapsulates the AskDocs MCP server state and operations."""

    def __init__(
        self,
        doc_sources: list[DocSource],
        embedding_model: str,
        llm_model: str,
        server_url: str,
        cache_dir: Path,
    ):
        """Initialize the server with specified models.

        Args:
            doc_sources: List of documentation sources to make available
            embedding_model: Ollama embedding model name
            llm_model: Ollama LLM model name
            server_url: Ollama server URL
            cache_dir: Directory to store vector store cache
        """
        self.doc_sources = doc_sources
        self.embeddings = OllamaEmbeddings(model=embedding_model, base_url=server_url)
        self.llm = OllamaLLM(model=llm_model, base_url=server_url)
        self.text_splitter = create_text_splitter()
        self.cache_dir = cache_dir

        # Storage for initialized documentation sources
        self.initialized_docs: dict[str, InitializedDoc] = {}
        self.failed_sources: list[tuple[str, str]] = []

    def create_retriever_for_source(self, source: DocSource):
        """Create a vector store and retriever for a documentation source"""
        persist_dir = self.cache_dir / ".askdocs-cache" / source.name
        complete_marker = persist_dir / ".complete"

        # Check if vector store already exists AND is complete
        if (
            persist_dir.exists()
            and (persist_dir / "chroma.sqlite3").exists()
            and complete_marker.exists()
        ):
            # Check if source documents have been updated since vector store was created
            if _source_is_newer_than_vectorstore(source, complete_marker):
                log(
                    f"⚠ Source documents for {source.name} have been updated, rebuilding vector store..."
                )
                import shutil

                shutil.rmtree(persist_dir)
                log("✓ Cleaned up outdated vector store")
            else:
                log(
                    f"Loading existing vector store for {source.name} from {persist_dir}"
                )
                try:
                    vector_store = Chroma(
                        persist_directory=str(persist_dir),
                        embedding_function=self.embeddings,
                    )
                    log(
                        f"✓ Successfully loaded existing vector store for {source.name}"
                    )
                    retriever = create_retriever_from_vectorstore(vector_store)
                    return retriever
                except Exception as e:
                    log(f"Error loading existing vector store for {source.name}: {e}")
                    log("Will recreate from documents...")

        # Check for incomplete vector store (interrupted previous run)
        if persist_dir.exists() and not complete_marker.exists():
            log(f"⚠ Found incomplete vector store for {source.name}, cleaning up...")
            import shutil

            shutil.rmtree(persist_dir)
            log("✓ Cleaned up incomplete vector store")

        # Vector store doesn't exist or failed to load - create from documents
        log(f"Creating new vector store for {source.name}")
        documents = load_document_source(source)
        if not documents:
            log(f"No documents loaded for {source.name}, skipping...")
            return None

        # Split into chunks with progress bar
        log("Splitting documents into chunks...")
        texts = []
        with tqdm(
            total=len(documents), desc="Chunking", file=sys.stderr, leave=False
        ) as pbar:
            for doc in documents:
                chunks = self.text_splitter.split_documents([doc])
                texts.extend(chunks)
                pbar.update(1)
        log(f"✓ Split into {len(texts)} chunks")

        # Create vector store with progress tracking
        log(f"Embedding {len(texts)} chunks (this will take a while)...")
        vector_store = create_vectorstore_with_progress(
            texts, self.embeddings, persist_dir, source.name
        )

        # Mark as complete only after successful creation
        complete_marker = persist_dir / ".complete"
        complete_marker.touch()
        log(f"✓ Vector store created and marked complete at {persist_dir}")

        # Create retriever
        retriever = create_retriever_from_vectorstore(vector_store)
        return retriever

    def create_rag_chain(self, retriever, source_description: str):
        """Create a RAG chain for a specific retriever"""
        rag_prompt = ChatPromptTemplate.from_template(
            f"""You are a technical documentation expert specializing in embedded systems, microcontrollers, and debug interfaces.
You are answering questions about: {source_description}

CRITICAL INSTRUCTIONS:
1. Use ONLY the information provided in the context below - do not make up or infer technical details
2. When providing register addresses, bit values, or protocol sequences, quote them EXACTLY as shown in the context
3. If the context contains conflicting information, acknowledge it
4. If the answer requires information not in the context, clearly state what's missing
5. Prioritize accuracy over completeness - it's better to say "not found in context" than to guess
6. Include specific technical details like:
   - Exact register names and addresses (e.g., 0x1234_5678)
   - Bit field definitions and values (e.g., bits [7:4])
   - Protocol sequences and timing requirements
   - Memory map ranges
7. When relevant, structure your answer with clear sections for better readability

Context: {{{{context}}}}

Question: {{{{question}}}}

Technical Answer:"""
        )

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def initialize_docs(self, doc_sources: list[DocSource]) -> None:
        """Initialize all documentation sources.

        Args:
            doc_sources: List of DocSource objects to initialize
        """
        log("=" * 80)
        log("Initializing documentation sources...")
        log("=" * 80)

        self.initialized_docs = {}
        self.failed_sources = []

        for source in doc_sources:
            try:
                log(
                    f"\n[{len(self.initialized_docs) + 1}/{len(doc_sources)}] Processing: {source.name}"
                )
                retriever = self.create_retriever_for_source(source)
                if retriever is None:
                    self.failed_sources.append((source.name, "No documents loaded"))
                    continue

                rag_chain = self.create_rag_chain(retriever, source.description)

                # Store for use by ask_docs tool
                self.initialized_docs[source.name] = InitializedDoc(
                    retriever=retriever, chain=rag_chain, description=source.description
                )
                log(f"✓ Initialized: {source.name}")

            except Exception as e:
                self.failed_sources.append((source.name, str(e)))
                log(f"✗ Error initializing {source.name}: {e}")
                import traceback

                log(traceback.format_exc())

        # Print summary
        log("\n" + "=" * 80)
        log("INITIALIZATION COMPLETE")
        log("=" * 80)
        log(
            f"✓ Successfully initialized {len(self.initialized_docs)} documentation source(s):"
        )
        for name, doc in self.initialized_docs.items():
            log(f"  - {name}: {doc.description}")

        log("\n✓ Registered tools:")
        log("  - list_docs (list all available documentation sources)")
        log("  - ask_docs (search any documentation source)")
        log("  - get_doc_page (retrieve full PDF page text content)")

        if self.failed_sources:
            log(f"\n✗ Failed to initialize {len(self.failed_sources)} source(s):")
            for name, error in self.failed_sources:
                log(f"  - {name}: {error}")

        log("\n" + "=" * 80)
        log("AskDocs MCP server ready")
        log("=" * 80)

    def list_docs(self) -> str:
        """List all available documentation sources."""
        if not self.initialized_docs:
            return "No documentation sources are currently available."

        result = "Available Documentation Sources:\n\n"
        for name in sorted(self.initialized_docs.keys()):
            result += f"- {name}\n"
            result += f"  {self.initialized_docs[name].description}\n\n"

        return result.strip()

    def ask_docs(self, source_name: str, query: str) -> str:
        """Search documentation sources with semantic search."""
        if source_name not in self.initialized_docs:
            available = ", ".join(sorted(self.initialized_docs.keys()))
            return f"Error: Unknown documentation source '{source_name}'. Available sources: {available}"

        doc = self.initialized_docs[source_name]
        return perform_search(doc.chain, doc.retriever, query)
