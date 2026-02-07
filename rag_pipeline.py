import os
import datetime
import csv
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

INDEX_PATH = "data/vectorstore_llama"
LOG_FILE = "logs/rag_logs.csv"
MODEL_NAME = "llama3"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOllama(model=MODEL_NAME, temperature=0)

def expand_query(original_query):
    expansion_prompt = ChatPromptTemplate.from_template(
        "You are a research assistant. Generate 3 variations of this search query "
        "to improve retrieval from a technical vector database. "
        "Output only the questions separated by newlines.\nQuery: {query}"
    )
    chain = expansion_prompt | llm
    response = chain.invoke({"query": original_query})
    queries = response.content.split('\n')
    return [q.strip() for q in queries if q.strip()] + [original_query]

def run_rag(user_query):
    search_queries = expand_query(user_query)
    print(f"searching with: {search_queries}")
    
    unique_docs = {}
    for q in search_queries:
        docs = retriever.invoke(q)
        for doc in docs:
            key = (doc.metadata.get('source_id'), doc.page_content[:20])
            unique_docs[key] = doc
    
    retrieved_docs = list(unique_docs.values())[:7]
    
    context_text = ""
    for doc in retrieved_docs:
        meta = doc.metadata
        context_text += (
            f"Source ID: {meta.get('source_id')}\n"
            f"Title: {meta.get('title', 'Unknown')}\n"
            f"Author: {meta.get('authors', 'Unknown')}\n"
            f"Year: {meta.get('year', 'n.d.')}\n"
            f"Content: {doc.page_content}\n"
            f"---\n"
        )

    system_prompt = """You are a rigorous research assistant. 
    Answer the user's question using ONLY the provided context.
    
    RULES:
    1. Cite your sources using the format: (Author, Year).
    2. If the context does not support the claim, DO NOT invent information. Say "I cannot find evidence for this."
    3. Do not use outside knowledge.
    """
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    chain = final_prompt | llm
    response = chain.invoke({"context": context_text, "question": user_query})
    
    log_interaction(user_query, [d.metadata.get('source_id') for d in retrieved_docs], response.content)
    
    return response.content

def log_interaction(query, sources, answer):
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "query", "retrieved_sources", "answer"])
        writer.writerow([datetime.datetime.now(), query, str(sources), answer])

if __name__ == "__main__":
    print("research portal phase 2 (local llama + query expansion)")
    while True:
        q = input("\nask a research question (or 'quit'): ")
        if q.lower() in ['quit', 'exit']: break
        
        try:
            answer = run_rag(q)
            print("\nanswer:\n")
            print(answer)
        except Exception as e:
            print(f"error: {e}")
        
        print("\n" + "-"*50)