import os
import pickle
import re
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate

INDEX_PATH = "data/vectorstore_llama"
BM25_PATH = "data/bm25_retriever.pkl"
LLM_MODEL = "deepseek-r1"
EMBEDDING_MODEL = "nomic-embed-text"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_enhanced_retriever():
    print("loading resources...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    with open(BM25_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
        bm25_retriever.k = 10
    
    print("initializing hybrid retrieval (BM25 + FAISS)...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    
    print("initializing reranker (Cross-Encoder)...")
    model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=model, top_n=5)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )
    
    return compression_retriever

def format_citations(response_text, retrieved_docs):
    cited_ids = set(re.findall(r'[\(\[]([A-Za-z0-9]+)(?:,.*?)?[\)\]]', response_text))
    
    if not cited_ids:
        return response_text
    
    bibliography = ["\n\n### References"]
    found_sources = set()
    
    doc_map = {d.metadata.get('source_id'): d.metadata for d in retrieved_docs}
    
    for source_id in cited_ids:
        if source_id in doc_map and source_id not in found_sources:
            meta = doc_map[source_id]
            entry = f"- **{source_id}**: {meta.get('authors')} ({meta.get('year')}). *{meta.get('title')}*."
            if meta.get('url'):
                entry += f" [Link]({meta.get('url')})"
            bibliography.append(entry)
            found_sources.add(source_id)
            
    return response_text + "\n".join(bibliography)

def run_pipeline():
    retriever = get_enhanced_retriever()
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    
    system_prompt = """
    You are a rigorous research assistant. Answer based ONLY on the provided context.
    
    CITATION RULES:
    1. Every claim must be immediately followed by a citation in the format [SourceID].
    2. Example: "Deep learning approaches have improved accuracy [Smith2023]."
    3. Do not create a "References" section yourself; just use the inline tags.
    4. If the context suggests an answer but isn't explicit, state your uncertainty.
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    chain = prompt_template | llm
    
    print("Research Portal V2 (Hybrid + Rerank) Ready.")
    
    while True:
        query = input("\nAsk (or 'exit'): ").strip()
        if query.lower() == 'exit': break
        
        print("searching & reranking...")
        docs = retriever.invoke(query)
        context_text = "\n\n".join([f"[{d.metadata.get('source_id')}]: {d.page_content}" for d in docs])
        
        print("generating...")
        response = chain.invoke({"context": context_text, "question": query})
        
        final_output = format_citations(response.content, docs)
        
        clean_output = re.sub(r'<think>.*?</think>', '', final_output, flags=re.DOTALL).strip()
        
        print("\n" + "="*50)
        print(clean_output)
        print("="*50 + "\n")

if __name__ == "__main__":
    run_pipeline()