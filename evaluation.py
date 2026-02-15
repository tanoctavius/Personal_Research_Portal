# NOTE: USED TO EVALUATE GEN_ANS - onyl run if there exists ans (run after generate_answers)

import os
import json
import logging
import warnings
import pandas as pd
from datasets import Dataset 
from ragas import evaluate
from ragas.run_config import RunConfig

try:
    from ragas.metrics import Faithfulness, ResponseRelevancy
except ImportError:
    from ragas.metrics import Faithfulness, AnswerRelevance as ResponseRelevancy

from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

CACHE_FILE = "logs/generation_cache.json"
OUTPUT_FILE = "logs/evaluation_results.csv"

# its fine to ignore htis
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("ragas").setLevel(logging.ERROR)

def grade_cache():
    if not os.path.exists(CACHE_FILE):
        print(f"Error: {CACHE_FILE} not found. Run 'generate_answers.py' first!")
        return

    print("Loading Judge Model (DeepSeek-R1)...")
    _eval_llm = ChatOllama(model="deepseek-r1", temperature=0, num_ctx=8192, timeout=600.0)
    evaluator_llm = LangchainLLMWrapper(_eval_llm)
    evaluator_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    faithfulness = Faithfulness()
    answer_relevance = ResponseRelevancy()

    print(f"Loading generated answers from {CACHE_FILE}...")
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data_dict = json.load(f)
    
    dataset = Dataset.from_dict(data_dict)

    print(f"Starting Grading on {len(dataset)} items...")
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevance],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(timeout=600, max_workers=1)
    )

    df = results.to_pandas()
    df['type'] = data_dict['type']
    
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\nEvaluation Complete!")
    print(f"Results saved to: {OUTPUT_FILE}")
    
    cols = [c for c in df.columns if c in ['faithfulness', 'answer_relevancy', 'answer_relevance']]
    if cols:
        print("\nAverage Scores by Type:")
        print(df.groupby('type')[cols].mean())

if __name__ == "__main__":
    grade_cache()