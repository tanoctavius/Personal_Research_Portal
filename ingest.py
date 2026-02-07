import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

MANIFEST_PATH = "data/data_manifest.csv"
INDEX_PATH = "data/vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def ingest():
    print("starting ingestion")
    
    df = pd.read_csv(MANIFEST_PATH)
    all_docs = []

    for _, row in df.iterrows():
        path = row['raw_path']
        if not os.path.exists(path):
            print(f"file not found: {path}")
            continue
            
        print(f"processing: {row['source_id']}")
        
        loader = PyPDFLoader(path)
        raw_docs = loader.load()
        
        for doc in raw_docs:
            doc.metadata['source_id'] = row['source_id']
            doc.metadata['title'] = row['title']
    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        split_docs = text_splitter.split_documents(raw_docs)
        
        for i, doc in enumerate(split_docs):
            doc.metadata['chunk_id'] = i
            all_docs.append(doc)

    print(f"total chunks created: {len(all_docs)}")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    
    vectorstore.save_local(INDEX_PATH)
    print("index saved to disk!")

if __name__ == "__main__":
    ingest()