import os
import csv
import pickle
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

MANIFEST_PATH = "data/data_manifest.csv"
INDEX_PATH = "data/vectorstore_llama"
BM25_PATH = "data/bm25_retriever.pkl"

def load_clean_manifest():
    if not os.path.exists(MANIFEST_PATH):
        print(f"error: manifest not found at {MANIFEST_PATH}")
        return []
    clean_data = []
    with open(MANIFEST_PATH, 'r', encoding='utf-8-sig') as f: 
        reader = csv.DictReader(f, skipinitialspace=True)
        reader.fieldnames = [k.strip() for k in reader.fieldnames]
        for row in reader:
            clean_row = {k: v.strip() for k, v in row.items() if k is not None}
            clean_data.append(clean_row)
    return clean_data

def ingest():
    manifest = load_clean_manifest()
    if not manifest: return

    print("initializing embedding model...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("initializing splitters...")
    semantic_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    safety_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

    all_docs = []

    print(f"processing {len(manifest)} files...")
    for entry in manifest:
        source_id = entry.get('source_id')
        file_path = entry.get('raw_path')
        
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            
            raw_docs = loader.load()
            safe_docs = safety_splitter.split_documents(raw_docs)

            for doc in safe_docs:
                doc.metadata['source_id'] = source_id
                doc.metadata['title'] = entry.get('title', 'Unknown')
                doc.metadata['authors'] = entry.get('authors', 'Unknown')
                doc.metadata['year'] = entry.get('year', 'n.d.')
                doc.metadata['url'] = entry.get('link/ DOI', '')

            try:
                print(f"semantic splitting {source_id}...")
                splits = semantic_splitter.split_documents(safe_docs)
            except Exception as e:
                print(f"semantic failed for {source_id}, falling back. Error: {e}")
                splits = safe_docs

            all_docs.extend(splits)
            print(f"loaded {source_id}: {len(splits)} chunks")
            
        except Exception as e:
            print(f"failed to load {source_id}: {e}")

    if not all_docs:
        print("no documents loaded.")
        return

    print("building BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    print("building FAISS index...")
    vectorstore = FAISS.from_documents(documents=all_docs, embedding=embeddings)
    vectorstore.save_local(INDEX_PATH)
    
    print("ingestion complete.")

# COMMENT THIS OUT IF YOURE NOT RUNNING IT OR ITLL REDO IT WHICH IS NO BUENO
if __name__ == "__main__":
    ingest()
