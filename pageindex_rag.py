"""
Page-Index RAG Pipeline
-----------------------
Strategy: Build a two-level hierarchical index.

  Level 1 – Page Summaries
      Each PDF page is summarised into a short description using llama3.2.
      These summaries are embedded and stored in a "page-index" FAISS store.

  Level 2 – Chunk Store
      Each page is further split into smaller chunks, each tagged with its
      source page number, stored in a second FAISS store.

  Retrieval:
      1. Query the page-summary store → find the most relevant pages.
      2. Filter the chunk store to only those pages.
      3. Return the top-k filtered chunks as context.

This replicates the "page-index" concept from the original design without
depending on the non-existent PageIndexClient library.
"""

import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

PDF_PATH = "./policies/insurances.pdf"
PAGE_INDEX_DIR = "./faiss_indexes/pageindex_summaries"
CHUNK_INDEX_DIR = "./faiss_indexes/pageindex_chunks"
SUMMARIES_CACHE = "./faiss_indexes/pageindex_summaries_cache.json"

SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
You are a document analyst. Write a single concise sentence (max 60 words) describing
what topics and insurance policy details are covered on this page.
Focus on key terms, coverage types, and policy clauses mentioned.

Page content:
{page_content}

One-sentence summary:
""")

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are an expert AI assistant for Indian insurance policies.
Answer the following question based ONLY on the provided context.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")


def _summarise_page(llm, parser, page_text: str) -> str:
    """Generate a one-sentence summary of a single page."""
    try:
        return parser.invoke(
            llm.invoke(SUMMARY_PROMPT.invoke({"page_content": page_text[:3000]}))
        ).strip()
    except Exception:
        # Fallback: use first 200 chars of the page as the summary
        return page_text[:200].replace("\n", " ")


def _build_indexes(embeddings: OllamaEmbeddings):
    """
    Load PDF, generate per-page summaries, embed them into the page-index store,
    and embed all chunks into the chunk store. Cache everything to disk.
    """
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()          # one Document per page

    llm = ChatOllama(model="llama3.2", temperature=0)
    parser = StrOutputParser()

    # ── Page-summary index ──────────────────────────────────────────────────
    print("  [PageIndex] Generating page summaries (this runs once)…")
    summary_docs = []
    summaries_cache = {}

    for i, page in enumerate(pages):
        page_num = page.metadata.get("page", i)
        print(f"    Summarising page {page_num + 1}/{len(pages)}…", end="\r")
        summary_text = _summarise_page(llm, parser, page.page_content)
        summaries_cache[str(page_num)] = summary_text
        summary_docs.append(
            Document(
                page_content=summary_text,
                metadata={"page": page_num},
            )
        )

    print()
    Path(PAGE_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    summary_vs = FAISS.from_documents(documents=summary_docs, embedding=embeddings)
    summary_vs.save_local(PAGE_INDEX_DIR)

    # Save cache for inspection
    with open(SUMMARIES_CACHE, "w") as f:
        json.dump(summaries_cache, f, indent=2)

    # ── Chunk index ─────────────────────────────────────────────────────────
    print("  [PageIndex] Building chunk index…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(pages)   # metadata carries page number

    Path(CHUNK_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    chunk_vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    chunk_vs.save_local(CHUNK_INDEX_DIR)

    return summary_vs, chunk_vs


def _load_indexes(embeddings: OllamaEmbeddings):
    summary_vs = FAISS.load_local(
        PAGE_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    chunk_vs = FAISS.load_local(
        CHUNK_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    return summary_vs, chunk_vs


def build_pageindex_rag():
    """
    Returns a callable: question -> {"answer": str, "contexts": list[str]}

    Uses a two-level page-index:
      1. Find the most relevant pages via page-summary embeddings.
      2. Retrieve fine-grained chunks only from those pages.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Build or load both indexes
    if Path(PAGE_INDEX_DIR).exists() and Path(CHUNK_INDEX_DIR).exists():
        summary_vs, chunk_vs = _load_indexes(embeddings)
    else:
        summary_vs, chunk_vs = _build_indexes(embeddings)

    llm = ChatOllama(model="llama3.2", temperature=0)
    parser = StrOutputParser()

    # Retrieve all chunk documents once for page-filtered lookup
    # (FAISS docstore gives us access to all stored docs)
    def _get_chunks_for_pages(target_pages: set, top_k: int = 6) -> list:
        """
        Scan the chunk docstore and return chunks whose page metadata is in target_pages.
        Falls back to a regular similarity search if no chunks match.
        """
        filtered = []
        for doc_id in chunk_vs.index_to_docstore_id.values():
            doc = chunk_vs.docstore.search(doc_id)
            if doc and doc.metadata.get("page") in target_pages:
                filtered.append(doc)
            if len(filtered) >= top_k * 3:  # gather extras for quality selection
                break
        return filtered[:top_k]

    def invoke(question: str) -> dict:
        # Level 1: find relevant pages via summary index
        top_summary_docs = summary_vs.similarity_search(question, k=3)
        target_pages = {d.metadata["page"] for d in top_summary_docs}

        # Level 2: retrieve chunks from those pages
        chunks = _get_chunks_for_pages(target_pages, top_k=6)

        # Fallback if page-filter yields nothing
        if not chunks:
            chunks = chunk_vs.similarity_search(question, k=5)

        context_str = "\n\n".join(d.page_content for d in chunks)

        answer = parser.invoke(
            llm.invoke(ANSWER_PROMPT.invoke({"context": context_str, "question": question}))
        )
        return {"answer": answer, "contexts": [d.page_content for d in chunks]}

    return invoke


if __name__ == "__main__":
    print("Building PageIndex RAG pipeline…")
    chain = build_pageindex_rag()
    print("Pipeline built successfully.\n")
    q = "What are the exclusions for the Social Media Cover?"
    print(f"Question: {q}\n")
    result = chain(q)
    print("Answer:", result["answer"])
    print(f"\nRetrieved {len(result['contexts'])} context chunks.")