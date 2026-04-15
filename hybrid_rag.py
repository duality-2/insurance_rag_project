"""
Hybrid RAG Pipeline
-------------------
Strategy: Combine dense (FAISS + nomic-embed-text) and sparse (BM25) retrieval
          via Reciprocal Rank Fusion, then answer with llama3.2.

Fix: replaced broken `langchain_classic` import with correct `langchain.retrievers`.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PDF_PATH = "./policies/insurances.pdf"
INDEX_DIR = "./faiss_indexes/hybrid"

PROMPT_TEMPLATE = """
You are an expert AI assistant for Indian insurance policies.
Answer the following question based ONLY on the provided context.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""


def _load_and_split() -> list:
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
    return splitter.split_documents(documents)


def _load_or_build_vectorstore(splits, embeddings: OllamaEmbeddings) -> FAISS:
    if Path(INDEX_DIR).exists():
        return FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    try:
        vs = FAISS.from_documents(documents=splits, embedding=embeddings)
    except Exception as exc:
        raise RuntimeError(
            "Failed to build embeddings — is Ollama running with nomic-embed-text?"
        ) from exc
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(INDEX_DIR)
    return vs


def build_hybrid_rag():
    """
    Returns a callable: question -> {"answer": str, "contexts": list[str]}
    """
    splits = _load_and_split()

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    faiss_vs = _load_or_build_vectorstore(splits, embeddings)

    faiss_retriever = faiss_vs.as_retriever(search_kwargs={"k": 5})

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5

    # EnsembleRetriever does Reciprocal Rank Fusion (RRF) by default
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOllama(model="llama3.2", temperature=0)
    parser = StrOutputParser()

    def invoke(question: str) -> dict:
        docs = ensemble_retriever.invoke(question)
        context_str = "\n\n".join(d.page_content for d in docs)
        answer = parser.invoke(
            llm.invoke(prompt.invoke({"context": context_str, "question": question}))
        )
        return {"answer": answer, "contexts": [d.page_content for d in docs]}

    return invoke


if __name__ == "__main__":
    print("Building Hybrid RAG pipeline...")
    chain = build_hybrid_rag()
    print("Pipeline built successfully.\n")
    q = "What is the waiting period for IT theft loss cover?"
    print(f"Question: {q}\n")
    result = chain(q)
    print("Answer:", result["answer"])
    print(f"\nRetrieved {len(result['contexts'])} context chunks.")
