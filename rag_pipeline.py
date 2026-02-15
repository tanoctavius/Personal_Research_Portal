import os
import sys
import csv
import time
import pickle
import re
from datetime import datetime
import logging
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TQD_DISABLE"] = "True"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

try:
    from langchain.retrievers import EnsembleRetriever
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    from langchain_community.retrievers import EnsembleRetriever
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.document_compressors import CrossEncoderReranker

from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

INDEX_PATH = "data/vectorstore_llama"
BM25_PATH = "data/bm25_retriever.pkl"
MANIFEST_PATH = "data/data_manifest.csv"
LOG_FILE = "logs/rag_logs.csv"
LLM_MODEL = "deepseek-r1"
EMBEDDING_MODEL = "nomic-embed-text"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def load_citation_map():
    mapping = {}
    if not os.path.exists(MANIFEST_PATH):
        return mapping 
    try:
        with open(MANIFEST_PATH, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            for row in reader:
                s_id = row.get('source_id')
                citation = row.get('in_text_citation')
                if s_id and citation:
                    mapping[s_id.strip()] = citation.strip()
    except Exception:
        pass
    return mapping

def get_enhanced_retriever():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    if not os.path.exists(INDEX_PATH) or not os.path.exists(BM25_PATH):
        raise FileNotFoundError("Indices not found. Run ingest.py first.")
        
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) 
    
    with open(BM25_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
        bm25_retriever.k = 10
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    
    model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=model, top_n=5) 
    
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)

def format_citations(response_text, retrieved_docs):
    cited_ids = set(re.findall(r'[\(\[]([A-Za-z0-9]+)(?:,.*?)?[\)\]]', response_text))
    citation_map = load_citation_map()
    
    def replace_match(match):
        return citation_map.get(match.group(1), f"({match.group(1)})")

    formatted_text = re.sub(r'[\(\[]([A-Za-z0-9]+)(?:,.*?)?[\)\]]', replace_match, response_text)

    if not cited_ids: return formatted_text
    
    bibliography = ["\n\n### References"]
    found_sources = set()
    doc_map = {d.metadata.get('source_id'): d.metadata for d in retrieved_docs}
    
    for source_id in cited_ids:
        if source_id in doc_map and source_id not in found_sources:
            meta = doc_map[source_id]
            entry = f"- **{source_id}**: {meta.get('authors', 'Unk')} ({meta.get('year', 'n.d.')}). *{meta.get('title', 'Untitled')}*."
            if meta.get('url'): entry += f" [Link]({meta.get('url')})"
            bibliography.append(entry)
            found_sources.add(source_id)
            
    return formatted_text + "\n".join(bibliography)

def log_interaction(query, response, sources, latency):
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'query', 'response', 'sources', 'latency_sec', 'model'])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query,
            response,
            "|".join(sources),
            f"{latency:.2f}",
            LLM_MODEL
        ])

def run_pipeline():
    print("Loading resources...")
    try:
        retriever = get_enhanced_retriever()
    except Exception as e:
        print(f"Error: {e}")
        return

    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    
    system_prompt = """
    You are a rigorous research assistant. Answer based ONLY on the provided context.
    CITATION RULES:
    1. Every claim must be immediately followed by a citation in the format [SourceID].
    2. Do NOT use (Author, Year) format yourself. Use the ID.
    3. If the context suggests an answer but isn't explicit, state your uncertainty.
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])
    chain = prompt_template | llm
    
    print("\n" + "="*50)
    print("Research Portal Phase 2 Ready.")
    print("="*50)
    
    while True:
        try:
            query = input("\nAsk (or 'exit'): ").strip()
            if query.lower() in ['exit', 'quit']: break
            if not query: continue
            
            start_time = time.time()
            
            print("searching & reranking...")
            docs = retriever.invoke(query)
            if not docs:
                print("No relevant documents found.")
                continue
                
            context_text = "\n\n".join([f"[{d.metadata.get('source_id')}]: {d.page_content}" for d in docs])
            
            print("generating...")
            raw_response = chain.invoke({"context": context_text, "question": query}).content
            
            final_output = format_citations(raw_response, docs)
            clean_output = re.sub(r'<think>.*?</think>', '', final_output, flags=re.DOTALL).strip()
            
            end_time = time.time()
            
            source_ids = [d.metadata.get('source_id', 'Unknown') for d in docs]
            log_interaction(query, clean_output, source_ids, end_time - start_time)
            
            print("\n" + "="*50)
            print(clean_output)
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    run_pipeline()