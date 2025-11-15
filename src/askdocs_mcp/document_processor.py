"""Document loading and vector store management."""

import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from tqdm import tqdm

from .config import CHUNK_OVERLAP, CHUNK_SIZE, RETRIEVAL_K, USE_MMR, DocSource, log


def load_document_source(source: DocSource) -> list:
    """Load documents from a PDF source"""
    log(f"Loading {source.name} from {source.path}...")

    if not source.path.exists():
        log(f"Warning: PDF does not exist: {source.path}")
        return []

    log("Loading PDF (this may take a moment)...")
    loader = PyPDFLoader(str(source.path))
    documents = loader.load()
    log(f"✓ Loaded {len(documents)} pages from PDF")
    return documents


def _source_is_newer_than_vectorstore(source: DocSource, complete_marker: Path) -> bool:
    """Check if source PDF has been modified since vector store was created"""
    if not source.path.exists():
        return False

    vectorstore_time = complete_marker.stat().st_mtime
    source_time = source.path.stat().st_mtime
    return source_time > vectorstore_time


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def perform_search(rag_chain, retriever, query: str) -> str:
    """Perform a search using a RAG chain and retriever"""
    # Get the answer from the RAG chain
    answer = rag_chain.invoke(query)

    # Get source documents for citation
    source_docs = retriever.invoke(query)

    if source_docs:
        answer += "\n\n\nSources:\n"
        seen_sources = set()
        # Show up to 8 unique sources (more context = more sources)
        for _i, doc in enumerate(source_docs, 1):
            source_file = Path(doc.metadata.get("source", "unknown")).name
            if source_file not in seen_sources:
                seen_sources.add(source_file)
                # Show page number if available (for PDFs)
                page = doc.metadata.get("page")
                if page is not None:
                    answer += f"{len(seen_sources)}. {source_file} (page {page + 1})\n"
                else:
                    answer += f"{len(seen_sources)}. {source_file}\n"
                if len(seen_sources) >= 8:
                    break

    return answer


def get_pdf_page_content(
    doc_sources: list[DocSource], pdf_name: str, page_start: int, page_end: int = None
) -> str:
    """
    Extract text content from PDF pages.

    Args:
        doc_sources: List of available documentation sources
        pdf_name: Name of the PDF source (e.g., "my_manual")
        page_start: Starting page number (1-indexed, as shown in citations)
        page_end: Optional ending page number for page ranges (1-indexed)

    Returns:
        Formatted string containing raw text content from the specified pages
    """
    # Find the PDF source
    pdf_source = None
    for source in doc_sources:
        if source.name == pdf_name:
            pdf_source = source
            break

    if pdf_source is None:
        return f"Error: PDF source '{pdf_name}' not found. Available sources: {', '.join(s.name for s in doc_sources)}"

    if not pdf_source.path.exists():
        return f"Error: PDF file not found at {pdf_source.path}"

    # Handle single page or range
    if page_end is None:
        page_end = page_start

    # Validate page numbers
    if page_start < 1 or page_end < page_start:
        return "Error: Invalid page range. page_start must be >= 1 and page_end must be >= page_start"

    try:
        # Convert to 0-indexed for pypdf
        page_start_idx = page_start - 1
        page_end_idx = page_end - 1

        # Extract text content using pypdf
        reader = PdfReader(str(pdf_source.path))
        total_pages = len(reader.pages)

        if page_end_idx >= total_pages:
            return f"Error: Page {page_end} exceeds total pages ({total_pages}) in PDF"

        # Build response
        response = f"# PDF Content: {pdf_source.name}\n"
        response += f"## Pages {page_start}-{page_end} (of {total_pages} total)\n\n"

        # Extract each page
        for page_num in range(page_start_idx, page_end_idx + 1):
            display_page_num = page_num + 1
            response += f"---\n\n### Page {display_page_num}\n\n"

            # Extract text content
            page = reader.pages[page_num]
            text_content = page.extract_text()

            response += f"{text_content}\n\n"

        return response

    except Exception as e:
        log(f"Error extracting PDF content: {e}")
        import traceback

        log(traceback.format_exc())
        return f"Error extracting PDF content: {str(e)}"


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create a text splitter for chunking PDF documents."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences
            " ",  # Words
            "",  # Characters (fallback for long unbreakable strings)
        ],
    )


def create_vectorstore_with_progress(texts, embeddings, persist_dir, source_name):
    """Create a vector store with progress tracking during embedding"""
    # Process in batches to show real progress
    batch_size = 50  # Smaller batches for more frequent updates

    # Create empty vector store first
    vector_store = Chroma(
        persist_directory=str(persist_dir), embedding_function=embeddings
    )

    # Add documents in batches with progress bar
    with tqdm(
        total=len(texts),
        desc=f"Embedding {source_name}",
        file=sys.stderr,
        unit="chunk",
        ncols=80,
    ) as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Extract text content and metadata
            batch_texts = [doc.page_content for doc in batch]
            batch_metadatas = [doc.metadata for doc in batch]

            # Add batch to vector store (this triggers embedding)
            vector_store.add_texts(texts=batch_texts, metadatas=batch_metadatas)

            # Update progress
            pbar.update(len(batch))

    return vector_store


def create_retriever_from_vectorstore(vector_store):
    """Create a retriever with configured search parameters"""
    if USE_MMR:
        # MMR provides diverse results (reduces redundancy)
        # lambda_mult: 0 = max diversity, 1 = max relevance
        # For technical docs, bias towards relevance (0.7) while still getting some diversity
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVAL_K,
                "fetch_k": RETRIEVAL_K
                * 4,  # Fetch more candidates for better MMR selection
                "lambda_mult": 0.7,  # Favor relevance over diversity for technical accuracy
            },
        )
        log(f"Using MMR retrieval with k={RETRIEVAL_K}, lambda={0.7}")
    else:
        # Standard similarity search
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": RETRIEVAL_K}
        )
        log(f"Using similarity search with k={RETRIEVAL_K}")

    return retriever
