Here is the clean, standard `README.md` file without emojis.

```markdown
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

Open your terminal and run these commands to download Llama 3 and the embedding model required for the vector database:

```bash
ollama pull llama3
ollama pull nomic-embed-text

```

## Configuration and Data Setup

### 1. Add Your Files (We ald have 20 uploaded)

Place your PDF (`.pdf`) or Text (`.txt`) files into the `data/raw/` folder.

### 2. Update the Manifest

Open `data/data_manifest.csv`. You must add a row for each new file you add to the raw folder. This file provides the metadata (Title, Author, Year) used for citations.

**CSV Format:**

```csv
source_id, title, authors, year, type, link/ DOI, raw_path
Zhang2025, "Paper Title", "Zhang, Y.", 2025, Paper, [https://arxiv.org/](https://arxiv.org/)..., data/raw/Zhang2025.pdf

```

*Note: Ensure titles are enclosed in double quotes if they contain commas. (Will mess up csv parsing if not)*

### 3. Build the Database

Run the ingestion script to process your files and build the vector index. You must run this every time you add new files or update the manifest. Currently, we have ald run it for the 20 files.

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
3. The AI will answer and provide citations in the format `(Author, Year)`.
4. Type `quit` or `exit` to close the program.

## Project Structure

```text
.
├── data/
│   ├── raw/               # Directory for input PDFs and TXT files
│   ├── vectorstore_llama/ # Vector database created by ingest.py
│   └── data_manifest.csv  # Metadata mapping for your files
├── logs/                  # Stores a CSV log of chat history
├── ingest.py              # Script to build the vector database
├── rag_pipeline.py        # Main script for the chat interface
├── requirements.txt       # List of Python dependencies
└── README.md

```

```

```
