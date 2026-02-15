# Personal Research Portal (Local RAG)

Iteration 1:
A local Retrieval-Augmented Generation (RAG) system that uses Llama 3 to answer questions based on uploaded PDFs. It runs entirely offline using Ollama.


## Prerequisites

To run, you must have the following installed:

1.  **Ollama**: This application runs the Large Language Model locally.
    * Download from: https://ollama.com
    * Verify installation by running `ollama --version` in your terminal.
2.  **Python**: Version 3.10 or higher.
3.  **Git**: To clone this repository.

## Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/yourusername/personal-research-portal.git](https://github.com/yourusername/personal-research-portal.git)
cd personal-research-portal

```

### 2. Install Dependencies
We HIGHLY HIGHLY recommend doing it in a venv so you have a completely fresh place (very painful if not) due to req mismatches.

```bash
conda create -n research_portal python=3.10 -y
conda activate research_portal
pip install -r requirements.txt
```

### 3. Pull the AI Models

Open your terminal and run these commands to download Deepseek and the embedding model required for the vector database:

```bash
ollama pull deepseek-r1
ollama pull llama3.2
ollama pull nomic-embed-text

```

## Configuration and Data Setup

### 1. Add Your Files (We ald have 20 uploaded)

Place your PDF (`.pdf`) or Text (`.txt`) files into the `data/raw/` folder.

### 2. Update the Manifest

Open `data/data_manifest.csv`. Every file must have a corresponding entry. This manifest is the "Source of Truth" for the Structured Citation system.

**CSV Format:**

```csv
source_id, title, authors, year, type, link/ DOI, raw_path, relevance, in_text_citation
Zhang2025, "Emoti-Attack", "Yangshijie Zhang", 2025, Paper, https://arxiv.org/..., data/raw/Zhang2025.pdf, "Note...", "(Zhang, 2025)"

```

*Note: Ensure titles are enclosed in double quotes if they contain commas. (Will mess up csv parsing if not)*

### 3. Build the Hybrid Database

This script performs Semantic Chunking and builds both the FAISS (Vector) and BM25 (Keyword) indices.

```bash
python ingest.py

```

## Usage

Run the main chat interface:

```bash
python rag_pipeline.py

```

1. Wait for the prompt `research portal phase 2...` to appear.
2. Type your question (e.g., "What does Zhang say about emoji attacks?").
3. The AI will answer and provide citations in the format `(Paper, Chunk, In_text_citation)`.
4. Toggle Reasoning: Type `toggle think` to show/hide DeepSeek's internal thought process.
5. Type `quit` or `exit` to close the program.

Characteristics:
1. Hybrid Search: Balances keyword accuracy (BM25) with semantic meaning (FAISS).

2. Cross-Encoder Reranking: Re-scores top results to ensure only the most relevant context is sent to the LLM.

3. Automatic Bibliography: Every answer resolves internal tags into a readable (Author, Year) format and appends a References section.

4. Production Logging: Every query, response, and latency metric is saved to logs/rag_logs.csv.


## Advanced Evaluation (2-Step Checkpointing)

To handle the high latency of reasoning models, the evaluation is split into two phases. This allows us to generate answers once and grade them multiple times without re-running the LLM.

Phase 1: Generation
Generates answers for 22 benchmark queries and caches them to logs/generation_cache.json.

```bash
python generate_answers.py
```

Phase 2: Grading
Uses Llama 3.2 to grade the cached answers against RAGAs metrics (Faithfulness and Relevancy).

```bash
python evaluation.py
```

## Stretch Goals Implemented

1. Hybrid Retrieval (BM25 + FAISS): Merges traditional search with vector embeddings to catch both technical terms and general concepts.

2. Cross-Encoder Reranking: Utilizes ms-marco-MiniLM to re-rank the top 20 retrieved chunks down to the best 5.

3. Semantic Chunking: Breaks documents at logical semantic shifts using AI embeddings instead of arbitrary character counts.

4. Structured Citations: A custom post-processing loop that maps internal IDs to your manifest's in_text_citation column.

## Project Structure

```text
.
├── data/
│   ├── raw/                  # Source PDFs
│   ├── vectorstore_llama/    # FAISS Dense Index
│   ├── bm25_retriever.pkl    # BM25 Sparse Index
│   └── data_manifest.csv     # Citation Metadata
├── logs/
│   ├── rag_logs.csv          # Interaction history
│   ├── generation_cache.json # Cached answers for Eval
│   └── evaluation_results.csv# Final RAGAs scores
├── ingest.py                 # Semantic + BM25 Ingestion
├── rag_pipeline.py           # Engine (Hybrid + Rerank + Logging)
├── generate_answers.py       # Eval Phase 1 (Generating)
├── evaluation.py             # Eval Phase 2 (Grading)
└── requirements.txt          # Pinned Dependency list
```
