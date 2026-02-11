import os
import csv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

MANIFEST_PATH = "data/data_manifest.csv"
INDEX_PATH = "data/vectorstore_llama"

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

def create_vector_db():
    manifest = load_clean_manifest()
    if not manifest:
        return

    print(f"manifest loaded: {len(manifest)} entries found.")
    
    all_docs = []

    for entry in manifest:
        source_id = entry.get('source_id')
        file_path = entry.get('raw_path')
        
        if not source_id:
            continue
            
        if not file_path:
            print(f"skipping {source_id}: no file path in csv")
            continue
            
        file_path = os.path.normpath(file_path)
        
        if not os.path.exists(file_path):
            print(f"skipping {source_id}: file not found at '{file_path}'")
            continue

        try:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
            
            loaded_docs = loader.load()
            
            for doc in loaded_docs:
                doc.metadata['source_id'] = source_id
                doc.metadata['title'] = entry.get('title', 'Unknown Title')
                doc.metadata['authors'] = entry.get('authors', 'Unknown Authors')
                doc.metadata['year'] = entry.get('year', 'n.d.')
                doc.metadata['url'] = entry.get('link/ DOI', '')
                
                doc.metadata['in_text_citation'] = entry.get('in_text_citation', f"({source_id})")
                
            all_docs.extend(loaded_docs)
            print(f"loaded: {source_id}")
            
        except Exception as e:
            print(f"failed to load {source_id}: {e}")

    if not all_docs:
        print("no documents were successfully loaded rip.")
        return

    print(f"\nprocessing {len(all_docs)} total pages...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    
    for i, split in enumerate(splits):
        split.metadata['chunk_id'] = i
    
    print(f"created {len(splits)} vector chunks.")
    print("initializing embeddings (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("building vector store...")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    print(f"saving index to {INDEX_PATH}...")
    vectorstore.save_local(INDEX_PATH)
    print("ingestion complete!")

# COMMENT THIS OUT IF YOURE NOT RUNNING IT OR ITLL REDO IT WHICH IS NO BUENO
if __name__ == "__main__":
    create_vector_db()