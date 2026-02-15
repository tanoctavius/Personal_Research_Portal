import pandas as pd
import os
import re
import sys
import logging
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQD_DISABLE"] = "True"

from datasets import Dataset 
from ragas import evaluate

try:
    from ragas.metrics import Faithfulness, ResponseRelevancy
except ImportError:
    from ragas.metrics import Faithfulness, AnswerRelevance as ResponseRelevancy

from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from rag_pipeline import get_enhanced_retriever, format_citations

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("ragas").setLevel(logging.ERROR)

def run_evaluation():
    print("Initializing Evaluation Resources...")
    
    _eval_llm = ChatOllama(model="deepseek-r1", temperature=0)
    evaluator_llm = LangchainLLMWrapper(_eval_llm)
    evaluator_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    faithfulness = Faithfulness()
    answer_relevance = ResponseRelevancy()

    retriever = get_enhanced_retriever()
    generator_llm = ChatOllama(model="deepseek-r1", temperature=0)

    SYSTEM_PROMPT = """
    You are a rigorous research assistant. Answer based ONLY on the provided context.

    CITATION RULES:
    1. Every claim must be immediately followed by a citation in the format [SourceID].
    2. Do NOT use (Author, Year) format yourself. Use the ID. The system will format it later.
    3. If the context suggests an answer but isn't explicit, state your uncertainty.
    """

    eval_data = [
        {"type": "Direct", "query": "What is the 'Emoji Attack' method proposed by Wei et al. (2025) and how does it affect Judge LLMs?"},
        {"type": "Direct", "query": "According to Chen et al. (2024), how do gender and age influence emoji comprehension?"},
        {"type": "Direct", "query": "What is 'EmojiLM' and how was the Text2Emoji corpus created?"},
        {"type": "Direct", "query": "Describe the 'Emojinize' system. How does it translate text to emojis?"},
        {"type": "Direct", "query": "What success rate did Gopinadh and Hussain (2026) report for emoji-based jailbreaking on the Qwen 2 7B model?"},
        {"type": "Direct", "query": "How does 'EmojiPrompt' obfuscate private data in cloud-based LLM interactions?"},
        {"type": "Direct", "query": "According to Zappavigna (2025), what are the two main ways LLMs use emojis as interpersonal resources?"},
        {"type": "Direct", "query": "What method does Zhang (2025) introduce in 'Emoti-Attack'?"},
        {"type": "Direct", "query": "How does ChatGPT perform when annotating emoji irony compared to humans, according to Zhou et al. (2025)?"},
        {"type": "Direct", "query": "What is the specific vulnerability identified in 'Small Symbols, Big Risks' regarding ASCII-based emoticons?"},
        {"type": "Synthesis", "query": "Compare the adversarial attack strategies in Wei2025 ('Emoji Attack') vs Zhang2025 ('Emoti-Attack'). How do they differ in their use of emojis?"},
        {"type": "Synthesis", "query": "Contrast the text-to-emoji translation approaches taken by 'EmojiLM' (Peng2023) and 'Emojinize' (Klein2024)."},
        {"type": "Synthesis", "query": "Discuss the safety implications of emojis in LLMs by synthesizing findings from Gopinadh2026 and Cui2025."},
        {"type": "Synthesis", "query": "How does human interpretation of emojis (Chen2024) compare to LLM interpretation of emojis (Zhou2025)?"},
        {"type": "Synthesis", "query": "What evidence exists in the corpus regarding emojis being used for privacy (Lin2025) versus emojis being used for attacks (Wei2025)?"},
        {"type": "Edge Case", "query": "Does the corpus contain evidence about the use of emojis in audio-to-text transcription models like Whisper?"},
        {"type": "Edge Case", "query": "What is the impact of emojis on stock market prediction algorithms according to these papers?"},
        {"type": "Edge Case", "query": "Does the corpus mention 'EmojiGAN' or image generation models for creating new emojis?"},
        {"type": "Edge Case", "query": "What specific hardware GPU was used to train the 'EmojiPrompt' system?"},
        {"type": "Edge Case", "query": "Are there any papers in the corpus published before 2018?"},
        {"type": "Edge Case", "query": "What is life?"},
        {"type": "Edge Case", "query": "How many chickens would fit in Carnegie Mellon?"}
    ]

    print(f"Starting RAGAs Evaluation on {len(eval_data)} queries...")
    
    data_for_ragas = {
        "question": [],
        "answer": [],
        "contexts": [],
        "type": []
    }
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])
    chain = prompt_template | generator_llm

    for i, item in enumerate(eval_data):
        q = item['query']
        print(f"[{i+1}/{len(eval_data)}] Processing: {q[:50]}...")
        
        try:
            docs = retriever.invoke(q)
            contexts = [d.page_content for d in docs]
            context_text = "\n\n".join([f"[{d.metadata.get('source_id')}]: {d.page_content}" for d in docs])
            
            response = chain.invoke({"context": context_text, "question": q}).content
            
            formatted_response = format_citations(response, docs)
            clean_response = re.sub(r'<think>.*?</think>', '', formatted_response, flags=re.DOTALL).strip()

            data_for_ragas["question"].append(q)
            data_for_ragas["answer"].append(clean_response)
            data_for_ragas["contexts"].append(contexts)
            data_for_ragas["type"].append(item['type'])
            
        except Exception as e:
            print(f"Error on query '{q}': {e}")

    print("\nCalculating Metrics...")
    
    dataset = Dataset.from_dict(data_for_ragas)
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevance],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    df = results.to_pandas()
    df['type'] = data_for_ragas['type']
    
    os.makedirs("logs", exist_ok=True)
    df.to_csv("logs/evaluation_results.csv", index=False)
    
    print("\nEvaluation Complete!")
    print("Results saved to: logs/evaluation_results.csv")
    print("\nAverage Scores by Query Type:")
    cols = [c for c in df.columns if c in ['faithfulness', 'answer_relevancy', 'answer_relevance']]
    print(df.groupby('type')[cols].mean())

if __name__ == "__main__":
    run_evaluation()