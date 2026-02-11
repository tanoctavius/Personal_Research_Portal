import os
import datetime
import csv
import re
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate # <--- This is the fix

# --- Configuration ---
index_path = "data/vectorstore_llama"
log_file = "logs/rag_logs.csv"
llm_model = "deepseek-r1"
embedding_model = "nomic-embed-text"

config = {"show_thinking": True}

def get_resources():
    print("loading models...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    if not os.path.exists(index_path):
        print(f"error: index not found at {index_path}. run ingest first.")
        exit()

    try:
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = ChatOllama(model=llm_model, temperature=0)
        return retriever, llm
    except Exception as e:
        print(f"error loading resources: {e}")
        exit()

retriever, llm = get_resources()

def clean_deepseek_think(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def expand_query(original_query):
    template = """Generate 3 specific search queries to find evidence for this question. Output ONLY the queries, one per line.
    Question: {query}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    response = chain.invoke({"query": original_query})
    
    content = clean_deepseek_think(response.content)
    
    queries = content.split('\n')
    clean_queries = [q.strip() for q in queries if q.strip()]
    return clean_queries[:3] + [original_query]

def log_interaction(query, sources, answer):
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "query", "retrieved_sources", "answer"])
        writer.writerow([datetime.datetime.now(), query, str(sources), answer])

def run_rag(user_query):
    search_queries = expand_query(user_query)
    print(f"searching with: {search_queries}")
    
    unique_docs = {}
    for q in search_queries:
        docs = retriever.invoke(q)
        for doc in docs:
            source_id = doc.metadata.get('source_id', 'unknown')
            chunk_id = doc.metadata.get('chunk_id', hash(doc.page_content) % 10000)
            
            key = (source_id, chunk_id)
            unique_docs[key] = doc
    
    retrieved_docs = list(unique_docs.values())[:7]
    
    context_text = ""
    for doc in retrieved_docs:
        meta = doc.metadata
        s_id = meta.get('source_id', 'unknown')
        c_id = meta.get('chunk_id', hash(doc.page_content) % 10000)
        context_text += f"Source: {s_id} (Chunk {c_id})\nContent: {doc.page_content}\n---\n"

    system_prompt = """You are a rigorous research assistant. 
    Answer the user's question using ONLY the provided context.
    
    RULES:
    1. Every claim must be immediately followed by a citation in format (SourceID, ChunkID).
    2. If the context does not support the claim, DO NOT invent information. Say "I cannot find evidence for this."
    3. Keep answers concise and direct.
    """
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    chain = final_prompt | llm
    print("generating answer...")
    response = chain.invoke({"context": context_text, "question": user_query})
    
    final_answer = response.content
    if not config["show_thinking"]:
        final_answer = clean_deepseek_think(final_answer)
    
    source_list = [d.metadata.get('source_id') for d in retrieved_docs]
    log_interaction(user_query, source_list, final_answer)
    
    return final_answer

if __name__ == "__main__":
    print(f"research portal phase 2 initialized ({llm_model})")
    print("commands: 'quit' to exit, 'toggle think' to show/hide reasoning")
    
    while True:
        q = input("\nask a research question: ")
        if q.lower() in ['quit', 'exit']: 
            break
        
        if q.lower() == 'toggle think':
            config["show_thinking"] = not config["show_thinking"]
            print(f"show thinking set to: {config['show_thinking']}")
            continue
        
        answer = run_rag(q)
        print("\nanswer:\n")
        print(answer)
        print("\n" + "-"*50)