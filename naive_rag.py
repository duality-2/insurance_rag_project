from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PDF_PATH = "./policies/insurances.pdf"
INDEX_DIR = "./faiss_indexes/naive"

def format_docs(docs):
    """Helper function to format documents for the context."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_naive_rag():
    """Builds a naive RAG pipeline using Ollama (llama3.2) and FAISS."""
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

    # 3. Define Prompt Template
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
    # Example usage:
    print("Building Naive RAG pipeline...")
    naive_chain = build_naive_rag()
    print("Pipeline built successfully.")
    question = "What is the policy on cyber stalking?"
    print(f"Querying with: '{question}'")
    response = naive_chain.invoke(question)
    print("Response:")
    print(response)
