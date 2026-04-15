# Insurance RAG Project — Fixed & Completed

Compares four RAG strategies on Indian insurance policy documents, evaluated
with RAGAS using fully local Ollama models (no external API keys required).

---

## Prerequisites

1. **Ollama** installed and running:  
   ```
   ollama serve
   ```

2. Pull the required models:
   ```
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Place your insurance PDF at:
   ```
   ./policies/insurances.pdf
   ```

---

## Project Structure

```
.
├── policies/
│   └── insurances.pdf          ← your source document
├── dataset_insurance_rag.csv   ← Q&A pairs with "question" and "ground_truth" columns
├── naive_rag.py
├── hybrid_rag.py
├── pageindex_rag.py
├── selfrag_rag.py
├── evaluate.py                 ← run this to benchmark all strategies
└── requirements.txt
```

---

## Running Individual Pipelines

Each pipeline can be tested standalone:

```bash
python naive_rag.py
python hybrid_rag.py
python pageindex_rag.py
python selfrag_rag.py
```

---

## Running the Full Evaluation

```bash
python evaluate.py
```

Outputs:
- **`evaluation_results.csv`** — per-question scores for all pipelines  
- **`evaluation_summary.txt`** — mean scores per pipeline

---

## RAG Strategies

| Strategy | Description |
|---|---|
| **Naive RAG** | Chunk → FAISS dense retrieval → generate |
| **Hybrid RAG** | BM25 (sparse) + FAISS (dense) fused via Reciprocal Rank Fusion |
| **Page-Index RAG** | Two-level index: page summaries → page-scoped chunk retrieval |
| **Self-RAG** | Retrieve → grade relevance → generate → grade hallucination → retry |

---

## What Was Fixed

| File | Problem | Fix |
|---|---|---|
| `hybrid_rag.py` | `from langchain_classic.retrievers import EnsembleRetriever` — package doesn't exist | Changed to `from langchain.retrievers import EnsembleRetriever` |
| `pageindex_rag.py` | Depended on non-existent `PageIndexClient` library | Re-implemented as a real two-level page-summary + chunk FAISS index |
| `selfrag_rag.py` | Was just a naive RAG with a reflection prompt — no actual self-grading | Implemented proper retrieve → relevance-grade → generate → hallucination-check → retry loop |
| `evaluate.py` | Contexts were empty lists (breaking RAGAS metrics); wrong RAGAS metric import paths | All pipelines now return `{"answer", "contexts"}`; updated to RAGAS v0.2+ API |

---

## RAGAS Metrics

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevancy** | Does the answer actually address the question? |
| **Context Recall** | Does the retrieved context cover the ground truth? |
| **Context Precision** | Are the retrieved chunks relevant (low noise)? |