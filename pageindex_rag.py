import os
import json
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from PageIndex.pageindex import PageIndexClient

def build_pageindex_rag():
    """Builds a RAG pipeline using the PageIndex hierarchical index with Ollama."""
    pdf_path = "./policies/insurances.pdf"
    workspace = "./pageindex_workspace"

    # Initialize PageIndex client with Ollama
    client = PageIndexClient(
        model="llama3.2",
        retrieve_model="llama3.2",
        workspace=workspace
    )

    # Index the document or load existing index
    doc_id = None
    if os.path.exists(workspace):
        # Try to find existing indexed document
        if hasattr(client, 'documents') and client.documents:
            doc_id = list(client.documents.keys())[0]
    
    if not doc_id:
        # Index the PDF
        print("Indexing PDF with PageIndex...")
        doc_id = client.index(pdf_path, mode="pdf")
        print(f"Document indexed with ID: {doc_id}")

    # Define the PageIndex retriever function
    def pageindex_retriever(query: str):
        """
        Retrieve relevant context from PageIndex using document structure and content.
        This is a simplified approach that uses page structure to find relevant sections.
        """
        try:
            # Get document structure to understand layout
            structure_json = client.get_document_structure(doc_id)
            structure = json.loads(structure_json)
            
            # For simplicity, retrieve a broad range of pages
            # In a production system, you'd parse the structure more intelligently
            document_json = client.get_document(doc_id)
            document_info = json.loads(document_json)
            
            # Get content from key pages (this is simplified)
            if document_info.get('page_count'):
                # Retrieve from pages distributed across the document
                page_count = document_info['page_count']
                sample_pages = [
                    str(max(1, page_count // 4)),
                    str(max(1, page_count // 2)),
                    str(min(page_count, 3 * page_count // 4))
                ]
                pages_str = ",".join(sample_pages)
            else:
                pages_str = "1-3"
            
            content_json = client.get_page_content(doc_id, pages_str)
            content_list = json.loads(content_json)
            
            # Combine content from all retrieved pages
            context = "\n\n".join([item.get('content', '') for item in content_list])
            return context
            
        except Exception as e:
            print(f"Error retrieving from PageIndex: {e}")
            return f"Unable to retrieve content. Error: {str(e)}"

    # Define Prompt and LLM
    template = """
    You are an expert AI assistant for Indian insurance policies.
    Answer the following question based only on the provided context, which has been intelligently retrieved from a document's hierarchical index.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model="llama3.2", temperature=0)

    # Build RAG Chain
    rag_chain = (
        {"context": pageindex_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__ == '__main__':
    print("Building PageIndex RAG pipeline...")
    pageindex_chain = build_pageindex_rag()
    print("Pipeline built successfully.")
    question = "What are the exclusions for the Social Media Cover?"
    print(f"Querying with: '{question}'")
    response = pageindex_chain.invoke(question)
    print("Response:")
    print(response)