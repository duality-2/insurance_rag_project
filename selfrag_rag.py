"""
Self-RAG Pipeline
-----------------
Strategy (proper implementation, not just a fancy prompt):

  1. Retrieve top-k candidates from FAISS.
  2. Grade each chunk for RELEVANCE to the question (llama3.2 as judge).
     Keep only relevant chunks; if none pass, fall back to top-3 regardless.
  3. Generate an answer from the filtered context.
  4. Grade the answer for HALLUCINATION / GROUNDEDNESS.
     If the answer is not grounded, retry generation with a stricter prompt.
  5. Return the final grounded answer and the relevant context chunks.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PDF_PATH = "./policies/insurances.pdf"
INDEX_DIR = "./faiss_indexes/selfrag"

# ── Prompts ────────────────────────────────────────────────────────────────────

RELEVANCE_PROMPT = ChatPromptTemplate.from_template("""
You are a grader assessing whether a retrieved document chunk is relevant to a question.
Respond with a single word: "yes" if it is relevant, "no" if it is not.

Document:
{document}

Question:
{question}

Relevant (yes/no):
""")

GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert AI assistant for Indian insurance policies.
Answer the following question based ONLY on the provided context.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")

STRICT_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert AI assistant. Your previous answer was flagged as potentially
hallucinated. Re-read the context carefully and provide a strictly grounded answer.
Every claim in your answer must be directly supported by the context below.
If you cannot find the answer, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Strictly grounded answer:
""")

HALLUCINATION_PROMPT = ChatPromptTemplate.from_template("""
You are a grader assessing whether an AI-generated answer is grounded in the provided context.
Respond with a single word: "yes" if the answer is grounded, "no" if it contains hallucinations.

Context:
{context}

Answer:
{answer}

Grounded (yes/no):
""")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_or_build_vectorstore(embeddings: OllamaEmbeddings) -> FAISS:
    if Path(INDEX_DIR).exists():
        return FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
    splits = splitter.split_documents(documents)

    try:
        vs = FAISS.from_documents(documents=splits, embedding=embeddings)
    except Exception as exc:
        raise RuntimeError(
            "Failed to build embeddings — is Ollama running with nomic-embed-text?"
        ) from exc

    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(INDEX_DIR)
    return vs


def _grade_relevance(llm, parser, chunk_text: str, question: str) -> bool:
    response = parser.invoke(
        llm.invoke(RELEVANCE_PROMPT.invoke({"document": chunk_text, "question": question}))
    )
    return response.strip().lower().startswith("yes")


def _grade_groundedness(llm, parser, context_str: str, answer: str) -> bool:
    response = parser.invoke(
        llm.invoke(HALLUCINATION_PROMPT.invoke({"context": context_str, "answer": answer}))
    )
    return response.strip().lower().startswith("yes")


# ── Pipeline builder ───────────────────────────────────────────────────────────

def build_selfrag_rag():
    """
    Returns a callable: question -> {"answer": str, "contexts": list[str]}

    The pipeline implements a real Self-RAG loop:
      retrieve → filter-by-relevance → generate → check-groundedness → (retry once)
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = _load_or_build_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    llm = ChatOllama(model="llama3.2", temperature=0)
    parser = StrOutputParser()

    def invoke(question: str) -> dict:
        # Step 1: Retrieve candidates
        candidate_docs = retriever.invoke(question)

        # Step 2: Grade each chunk for relevance
        relevant_docs = [
            doc for doc in candidate_docs
            if _grade_relevance(llm, parser, doc.page_content, question)
        ]

        # Fallback: if no chunk passed, use top-3 anyway
        if not relevant_docs:
            print("  [Self-RAG] No chunks graded relevant — using top-3 fallback.")
            relevant_docs = candidate_docs[:3]

        context_str = "\n\n".join(d.page_content for d in relevant_docs)

        # Step 3: Generate answer
        answer = parser.invoke(
            llm.invoke(GENERATION_PROMPT.invoke({"context": context_str, "question": question}))
        )

        # Step 4: Grade for hallucination
        grounded = _grade_groundedness(llm, parser, context_str, answer)
        if not grounded:
            print("  [Self-RAG] Answer flagged as hallucinated — retrying with strict prompt.")
            answer = parser.invoke(
                llm.invoke(
                    STRICT_GENERATION_PROMPT.invoke({"context": context_str, "question": question})
                )
            )

        return {"answer": answer, "contexts": [d.page_content for d in relevant_docs]}

    return invoke


if __name__ == "__main__":
    print("Building Self-RAG pipeline...")
    chain = build_selfrag_rag()
    print("Pipeline built successfully.\n")
    q = "Explain the process for filing a claim under IT Theft Loss Cover."
    print(f"Question: {q}\n")
    result = chain(q)
    print("Answer:", result["answer"])
    print(f"\nUsed {len(result['contexts'])} graded-relevant context chunks.")