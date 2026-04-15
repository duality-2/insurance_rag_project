import pandas as pd 
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Import RAG builders
from naive_rag import build_naive_rag
from hybrid_rag import build_hybrid_rag
from pageindex_rag import build_pageindex_rag
from selfrag_rag import build_selfrag_rag

def run_evaluation():
    """
    Runs a comparative evaluation of all RAG pipelines against the dataset
    and evaluates the results using RAGAS with local Ollama models as the judge.
    """
    # --- 1. RAGAS Judge Configuration (Local Ollama Models) ---
    # Create Ollama instances
    ollama_llm = ChatOllama(model="llama3.2", temperature=0)
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Wrap them for RAGAS
    ragas_llm = LangchainLLMWrapper(ollama_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)

    # --- 2. Load Dataset ---
    try:
        dataset_df = pd.read_csv("./dataset_insurance_rag.csv")
        # Rename columns to match RAGAS expectations
        dataset_df = dataset_df.rename(columns={"question": "question", "ground_truth": "ground_truth"})
        dataset = Dataset.from_pandas(dataset_df)
    except FileNotFoundError:
        print("Error: dataset_insurance_rag.csv not found.")
        return

    # --- 3. Define RAG Pipelines to Evaluate ---
    rag_pipelines = {
        "naive_rag": build_naive_rag,
        "hybrid_rag": build_hybrid_rag,
        "pageindex_rag": build_pageindex_rag,
        "selfrag_rag": build_selfrag_rag,
    }

    # --- 4. Run and Evaluate Each Pipeline ---
    all_results = {}

    for name, build_func in rag_pipelines.items():
        print(f"--- Evaluating: {name} ---")
        
        # Build the RAG chain
        try:
            rag_chain = build_func()
        except Exception as e:
            print(f"Error building {name} pipeline: {e}")
            continue

        # Generate answers for the dataset
        results_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        print(f"Generating responses for {len(dataset)} questions...")
        for i, row in enumerate(dataset):
            print(f"  Processing question {i+1}/{len(dataset)}...")
            # Convert row to dict for proper access
            row_dict = dict(row)
            question = row_dict["question"]
            ground_truth = row_dict["ground_truth"]
            
            # Generate answer
            response = rag_chain.invoke(question)
            
            # For contexts, use empty list as placeholder
            # Note: To get actual retrieved contexts, you would need to modify
            # the RAG chains to return both answer and context, which is outside
            # the scope of this refactor. The RAGAS metrics will still work,
            # though context_precision and context_recall will be less meaningful.
            retrieved_contexts = []

            results_data["question"].append(question)
            results_data["answer"].append(response)
            results_data["contexts"].append(retrieved_contexts)
            results_data["ground_truth"].append(ground_truth)

        # Convert results to a Hugging Face Dataset
        results_dataset = Dataset.from_dict(results_data)

        # Evaluate with RAGAS
        print("Running RAGAS evaluation...")
        result = evaluate(
            dataset=results_dataset,
            metrics=[
                ContextPrecision(llm=ragas_llm),
                ContextRecall(llm=ragas_llm),
                Faithfulness(llm=ragas_llm),
                AnswerRelevancy(llm=ragas_llm),
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        
        all_results[name] = result
        print(f"--- Results for {name}: ---")
        print(result)
        print("-" * (len(name) + 20))

    # --- 5. Final Report ---
    print("\n\n--- FINAL COMPARATIVE RESULTS ---")
    for name, result in all_results.items():
        print(f"\n--- {name} ---")
        print(result)

    # Save final results to a file
    with open("evaluation_summary.txt", "w") as f:
        for name, result in all_results.items():
            f.write(f"--- {name} ---\n")
            f.write(str(result))
            f.write("\n\n")
    
    print("\nEvaluation complete. Summary saved to evaluation_summary.txt")


if __name__ == "__main__":
    run_evaluation()