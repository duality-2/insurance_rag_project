"""
evaluate.py
-----------
Runs all four RAG pipelines against the dataset and evaluates each using
RAGAS with local Ollama models (llama3.2 as LLM judge, nomic-embed-text
for embedding-based metrics).

Metrics:
  • Faithfulness          – is the answer grounded in the retrieved context?
  • Answer Relevancy      – does the answer address the question?
  • Context Recall        – does the context cover the ground truth?
  • Context Precision     – are the retrieved chunks actually useful?

Output:
  • Prints a per-strategy table to stdout.
  • Saves full results to  evaluation_results.csv
  • Saves a human-readable summary to  evaluation_summary.txt
"""

import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from openai import OpenAI as OpenAIClient

# RAG pipeline builders (each returns a callable: str -> {"answer": str, "contexts": list[str]})
from naive_rag import build_naive_rag
from hybrid_rag import build_hybrid_rag
from pageindex_rag import build_pageindex_rag
from selfrag_rag import build_selfrag_rag


# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_PATH = "./dataset_insurance_rag.csv"
RESULTS_CSV = "./evaluation_results.csv"
SUMMARY_TXT = "./evaluation_summary.txt"

# Subset of questions to run (set to None to run all)
MAX_QUESTIONS: int | None = None  # e.g. 10 for a quick smoke-test


# ── Main evaluation loop ───────────────────────────────────────────────────────


def run_evaluation():
    llm_client = OpenAIClient(base_url="http://localhost:11434/v1", api_key="ollama")
    embeddings_client = OpenAIClient(
        base_url="http://localhost:11434/v1", api_key="ollama"
    )

    ragas_llm = llm_factory("llama3.2", provider="openai", client=llm_client)
    ragas_embeddings = embedding_factory(
        "openai", model="nomic-embed-text", client=embeddings_client
    )

    # 2. Load Q&A dataset
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        return

    column_mapping = {
        "Question": "question",
        "Ground Truth Answer": "ground_truth",
    }
    df = df.rename(columns=column_mapping)

    required_cols = {"question", "ground_truth"}
    if not required_cols.issubset(df.columns):
        print(
            f"ERROR: Dataset must contain columns: {required_cols}. Found: {set(df.columns)}"
        )
        return

    if MAX_QUESTIONS:
        df = df.head(MAX_QUESTIONS)

    questions = df["question"].tolist()
    ground_truths = df["ground_truth"].tolist()
    print(f"Loaded {len(questions)} questions from dataset.\n")

    # 3. Pipeline registry
    pipelines = {
        "naive_rag": build_naive_rag,
        "hybrid_rag": build_hybrid_rag,
        "pageindex_rag": build_pageindex_rag,
        "selfrag_rag": build_selfrag_rag,
    }

    all_results: dict[str, dict] = {}
    all_rows: list[dict] = []

    # 4. Evaluate each pipeline
    for name, build_fn in pipelines.items():
        print("=" * 60)
        print(f"  Pipeline: {name}")
        print("=" * 60)

        try:
            chain = build_fn()
        except Exception as exc:
            print(f"  [ERROR] Could not build pipeline: {exc}\n")
            continue

        answers, contexts = [], []

        for i, (question, _) in enumerate(zip(questions, ground_truths)):
            print(f"  Q{i + 1}/{len(questions)}: {question[:70]}…")
            try:
                result = chain(question)
                answers.append(result["answer"])
                contexts.append(result["contexts"])
            except Exception as exc:
                print(f"    [ERROR] {exc}")
                answers.append("ERROR")
                contexts.append([])

        # Build RAGAS-compatible dataset
        # RAGAS expects:
        #   question     : str
        #   answer       : str
        #   contexts     : list[str]   ← retrieved chunks (NOT ground truth)
        #   ground_truth : str
        ragas_data = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        print(f"\n  Running RAGAS evaluation for {name}…")
        try:
            result = evaluate(
                dataset=ragas_data,
                metrics=[
                    Faithfulness(llm=ragas_llm),
                    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
                    ContextRecall(llm=ragas_llm),
                    ContextPrecision(llm=ragas_llm),
                ],
            )
            scores = result.to_pandas()
            all_results[name] = result
            print(
                f"  Done. Scores:\n{scores[['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']].mean().to_string()}\n"
            )

            # Collect per-question rows for the CSV
            for _, row in scores.iterrows():
                all_rows.append({"pipeline": name, **row.to_dict()})

        except Exception as exc:
            print(f"  [ERROR] RAGAS evaluation failed: {exc}\n")

    # 5. Save outputs
    print("\n" + "=" * 60)
    print("  FINAL COMPARATIVE RESULTS")
    print("=" * 60)

    summary_lines = []
    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]

    for name, result in all_results.items():
        df_res = result.to_pandas()
        mean_scores = df_res[metric_cols].mean()
        line = f"\n{'─' * 40}\n{name}\n{'─' * 40}\n{mean_scores.to_string()}"
        print(line)
        summary_lines.append(line)

    # Save CSV
    if all_rows:
        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(RESULTS_CSV, index=False)
        print(f"\nFull results saved → {RESULTS_CSV}")

    # Save text summary
    with open(SUMMARY_TXT, "w") as f:
        f.write("Insurance RAG Evaluation Summary\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(summary_lines))

    print(f"Summary saved → {SUMMARY_TXT}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    run_evaluation()
