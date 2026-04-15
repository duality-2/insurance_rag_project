from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PDF_PATH = "./policies/insurances.pdf"
INDEX_DIR = "./faiss_indexes/selfrag"

def format_docs(docs):
    """Helper function to format documents for the context."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_selfrag_rag():
    """
    Builds a placeholder for a Self-RAG pipeline using Ollama.
    Note: A true Self-RAG implementation is more complex and typically involves
    a graph-based or agentic approach to reflect on retrieved documents and
    iteratively improve the context. This is a simplified structure.
    """
    # 1. Load embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 2. Reuse on-disk FAISS index when available to avoid repeated embedding calls
    if Path(INDEX_DIR).exists():
        vectorstore = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        try:
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        except Exception as exc:
            raise RuntimeError(
                "Failed to build embeddings. Embedding service may be unavailable. "
                "Ensure Ollama is running with nomic-embed-text model."
            ) from exc
        Path(INDEX_DIR).parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(INDEX_DIR)

    retriever = vectorstore.as_retriever()

    # 3. Define a more complex prompt for Self-Correction/Reflection
    template = """
    You are an advanced AI assistant. Your task is to answer the question by first critically evaluating the provided context.
    1. Reflect on whether the context is relevant to the question.
    2. If relevant, extract the precise information to construct the answer.
    3. If the context is not sufficient, state that the answer cannot be found in the provided documents.

    Based on this process, provide a direct and concise answer to the user's question.

    Context:
    {context}

    Question:
    {question}

    Reflective Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Initialize LLM
    llm = ChatOllama(model="llama3.2", temperature=0)

    # 5. Build RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__ == '__main__':
    print("Building Self-RAG (placeholder) pipeline...")
    selfrag_chain = build_selfrag_rag()
    print("Pipeline built successfully.")
    question = "Explain the process for filing a claim under IT Theft Loss Cover."
    print(f"Querying with: '{question}'")
    response = selfrag_chain.invoke(question)
    print("Response:")
    print(response)