from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PDF_PATH = "./policies/insurances.pdf"
INDEX_DIR = "./faiss_indexes/hybrid"

def format_docs(docs):
    """Helper function to format documents for the context."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_hybrid_rag():
    """Builds a hybrid RAG pipeline using Ollama (llama3.2), FAISS, and BM25."""
    # 1. Load and Chunk Document (used by BM25, and by FAISS if cache is missing)
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # 2. Initialize Retrievers
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if Path(INDEX_DIR).exists():
        faiss_vectorstore = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        try:
            faiss_vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        except Exception as exc:
            raise RuntimeError(
                "Failed to build embeddings. Embedding service may be unavailable. "
                "Ensure Ollama is running with nomic-embed-text model."
            ) from exc
        Path(INDEX_DIR).parent.mkdir(parents=True, exist_ok=True)
        faiss_vectorstore.save_local(INDEX_DIR)
    
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5

    # 3. Initialize Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    # 4. Define Prompt and LLM
    template = """
    You are an expert AI assistant for Indian insurance policies.
    Answer the following question based only on the provided context.
    If the answer is not in the context, state that you don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model="llama3.2", temperature=0)

    # 5. Build RAG Chain
    rag_chain = (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__ == '__main__':
    print("Building Hybrid RAG pipeline...")
    hybrid_chain = build_hybrid_rag()
    print("Pipeline built successfully.")
    question = "What is the waiting period for IT theft loss cover?"
    print(f"Querying with: '{question}'")
    response = hybrid_chain.invoke(question)
    print("Response:")
    print(response)