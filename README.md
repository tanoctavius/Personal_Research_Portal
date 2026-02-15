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

```bash
pip install -r requirements.txt

```

### 3. Pull the AI Models

Open your terminal and run these commands to download Deepseek and the embedding model required for the vector database:

```bash
ollama pull deepseek-r1
ollama pull nomic-embed-text

```

## Configuration and Data Setup

### 1. Add Your Files (We ald have 20 uploaded)

Place your PDF (`.pdf`) or Text (`.txt`) files into the `data/raw/` folder.

### 2. Update the Manifest

Open `data/data_manifest.csv`. You must add a row for each new file you add to the raw folder. This file provides the metadata (Title, Author, Year) used for citations.

**CSV Format:**

```csv
source_id, title, authors, year, type, link/ DOI, raw_path, relevance, in_text_citation
Zhang2025, "Emoti-Attack", "Zhang, Y.", 2025, Paper, [https://arxiv.org/](https://arxiv.org/)..., data/raw/Zhang2025.pdf, "Relevance note...", "(Zhang, 2025)"

```

*Note: Ensure titles are enclosed in double quotes if they contain commas. (Will mess up csv parsing if not)*

### 3. Build the Database

This processes your documents using Semantic Chunking and builds two indices: a FAISS vector store and a BM25 keyword index.

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
Hybrid Search: Automatically balances keyword matching and semantic meaning.

Automatic Bibliography: Every answer extracts citations from the manifest and appends a Reference list.

Toggle Reasoning: Type toggle think to see the model's internal logic.


## Evaluation

To automatically test the system against 20 pre-defined queries (Direct, Synthesis, and Edge Cases):

```bash
python evaluate.py

```

This script evaluates the system using RAGAs metrics:

Faithfulness: Measures if the answer is derived solely from the retrieved context (prevents hallucination).

Answer Relevance: Measures how well the response addresses the specific query.

Results: Detailed logs and average scores per query type (Direct, Synthesis, Edge Case) are saved to logs/evaluation_results.csv.


## Stretch Goals Implemented

1. Hybrid Retrieval (BM25 + FAISS): Combines traditional keyword search with modern vector embeddings to ensure technical terms (like specific paper names) are never missed.

2. Cross-Encoder Reranking: Once candidates are retrieved, a secondary model re-scores them to ensure only the most relevant 5 chunks reach the LLM.

3. Semantic Chunking: Uses a "safety-first" split followed by an embedding-based analyzer to break text at logical semantic shifts rather than arbitrary character counts.

4. Structured Citations: Implements a post-processing loop that resolves inline tags into a formatted bibliography using your data manifest.


## Project Structure

```text
.
├── data/
│   ├── raw/                  # Source PDFs and Text files
│   ├── vectorstore_llama/    # FAISS Dense Index
│   ├── bm25_retriever.pkl    # BM25 Sparse Index
│   └── data_manifest.csv     # Metadata (Title, Author, DOI)
├── logs/
│   ├── rag_logs.csv          # Chat interaction history
│   └── evaluation_results.csv # RAGAs scores and latency logs
├── ingest.py                 # Multi-stage ingestion (Semantic + BM25)
├── rag_pipeline.py           # The RAG engine (Hybrid + Rerank + Citations)
├── evaluate.py               # RAGAs evaluation suite (20 queries)
└── requirements.txt          # Full dependency list

```
