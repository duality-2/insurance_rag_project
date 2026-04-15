"""
Naive RAG Pipeline
------------------
Strategy: Chunk PDF → embed with nomic-embed-text → store in FAISS →
          retrieve top-k chunks → answer with llama3.2.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PDF_PATH = "./policies/insurances.pdf"
INDEX_DIR = "./faiss_indexes/naive"

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


def build_naive_rag():
    """
    Returns a callable that accepts a question string and returns:
        {"answer": str, "contexts": list[str]}
    so evaluate.py can capture retrieved context for RAGAS.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = _load_or_build_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOllama(model="llama3.2", temperature=0)
    parser = StrOutputParser()

    def invoke(question: str) -> dict:
        docs = retriever.invoke(question)
        context_str = "\n\n".join(d.page_content for d in docs)
        answer = parser.invoke(llm.invoke(prompt.invoke({"context": context_str, "question": question})))
        return {"answer": answer, "contexts": [d.page_content for d in docs]}

    return invoke


if __name__ == "__main__":
    print("Building Naive RAG pipeline...")
    chain = build_naive_rag()
    print("Pipeline built successfully.\n")
    q = "What is the policy on cyber stalking?"
    print(f"Question: {q}\n")
    result = chain(q)
    print("Answer:", result["answer"])
    print(f"\nRetrieved {len(result['contexts'])} context chunks.")