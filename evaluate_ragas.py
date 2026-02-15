import pandas as pd
import re
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rag_pipeline_v2 import get_enhanced_retriever

evaluator_llm = ChatOllama(model="deepseek-r1", temperature=0)
evaluator_embeddings = OllamaEmbeddings(model="nomic-embed-text")

def run_ragas_eval():
    retriever = get_enhanced_retriever()
    llm = ChatOllama(model="deepseek-r1", temperature=0)
    
    eval_data = [
        {"question": "What is the specific methodology used in Paper A?", "ground_truth": "The authors used a transformer-based..."},
        {"question": "How does the proposed system handle encryption?", "ground_truth": "It uses AES-256 for data at rest..."}
    ]
    
    results_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print("Generating answers for evaluation...")
    for item in eval_data:
        q = item['question']
        
        docs = retriever.invoke(q)
        contexts = [d.page_content for d in docs]
        
        prompt = f"Context: {contexts}\n\nQuestion: {q}\nAnswer:"
        response = llm.invoke(prompt).content
        
        clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        results_data["question"].append(q)
        results_data["answer"].append(clean_response)
        results_data["contexts"].append(contexts)
        results_data["ground_truth"].append(item.get("ground_truth", ""))

    dataset = Dataset.from_dict(results_data)
    
    print("Running RAGAs metrics (this takes time)...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevance],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    print("\nEvaluation Results:")
    print(results)
    
    df = results.to_pandas()
    df.to_csv("logs/ragas_results.csv", index=False)
    print("Saved to logs/ragas_results.csv")

if __name__ == "__main__":
    run_ragas_eval()