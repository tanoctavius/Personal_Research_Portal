import pandas as pd
import time
import csv
import os
from rag_pipeline import run_rag 

eval_data = [
    {
        "type": "Direct",
        "query": "What is the 'Emoji Attack' method proposed by Wei et al. (2025) and how does it affect Judge LLMs?",
        "target_source": "Wei2025"
    },
    {
        "type": "Direct",
        "query": "According to Chen et al. (2024), how do gender and age influence emoji comprehension?",
        "target_source": "Chen2024"
    },
    {
        "type": "Direct",
        "query": "What is 'EmojiLM' and how was the Text2Emoji corpus created?",
        "target_source": "Peng2023"
    },
    {
        "type": "Direct",
        "query": "Describe the 'Emojinize' system. How does it translate text to emojis?",
        "target_source": "Klein2024"
    },
    {
        "type": "Direct",
        "query": "What success rate did Gopinadh and Hussain (2026) report for emoji-based jailbreaking on the Qwen 2 7B model?",
        "target_source": "Gopinadh2026"
    },
    {
        "type": "Direct",
        "query": "How does 'EmojiPrompt' obfuscate private data in cloud-based LLM interactions?",
        "target_source": "Lin2025"
    },
    {
        "type": "Direct",
        "query": "According to Zappavigna (2025), what are the two main ways LLMs use emojis as interpersonal resources?",
        "target_source": "Zappavigna2025"
    },
    {
        "type": "Direct",
        "query": "What method does Zhang (2025) introduce in 'Emoti-Attack'?",
        "target_source": "Zhang2025"
    },
    {
        "type": "Direct",
        "query": "How does ChatGPT perform when annotating emoji irony compared to humans, according to Zhou et al. (2025)?",
        "target_source": "Zhou2025"
    },
    {
        "type": "Direct",
        "query": "What is the specific vulnerability identified in 'Small Symbols, Big Risks' regarding ASCII-based emoticons?",
        "target_source": "Jiang2026"
    },

    {
        "type": "Synthesis",
        "query": "Compare the adversarial attack strategies in Wei2025 ('Emoji Attack') vs Zhang2025 ('Emoti-Attack'). How do they differ in their use of emojis?",
        "target_source": "Wei2025, Zhang2025"
    },
    {
        "type": "Synthesis",
        "query": "Contrast the text-to-emoji translation approaches taken by 'EmojiLM' (Peng2023) and 'Emojinize' (Klein2024).",
        "target_source": "Peng2023, Klein2024"
    },
    {
        "type": "Synthesis",
        "query": "Discuss the safety implications of emojis in LLMs by synthesizing findings from Gopinadh2026 and Cui2025.",
        "target_source": "Gopinadh2026, Cui2025"
    },
    {
        "type": "Synthesis",
        "query": "How does human interpretation of emojis (Chen2024) compare to LLM interpretation of emojis (Zhou2025)?",
        "target_source": "Chen2024, Zhou2025"
    },
    {
        "type": "Synthesis",
        "query": "What evidence exists in the corpus regarding emojis being used for privacy (Lin2025) versus emojis being used for attacks (Wei2025)?",
        "target_source": "Lin2025, Wei2025"
    },

    {
        "type": "Edge Case",
        "query": "Does the corpus contain evidence about the use of emojis in audio-to-text transcription models like Whisper?",
        "target_source": "None (Should answer Negative)"
    },
    {
        "type": "Edge Case",
        "query": "What is the impact of emojis on stock market prediction algorithms according to these papers?",
        "target_source": "None (Should answer Negative)"
    },
    {
        "type": "Edge Case",
        "query": "Does the corpus mention 'EmojiGAN' or image generation models for creating new emojis?",
        "target_source": "None (Should answer Negative)"
    },
    {
        "type": "Edge Case",
        "query": "What specific hardware GPU was used to train the 'EmojiPrompt' system?",
        "target_source": "Unknown (Likely not in abstract/corpus)"
    },
    {
        "type": "Edge Case",
        "query": "Are there any papers in the corpus published before 2018?",
        "target_source": "None (Oldest is likely 2019 or later based on manifest)"
    }
]

def run_evaluation():
    results = []
    print(f"Starting Evaluation on {len(eval_data)} queries...")
    
    output_file = "logs/evaluation_results.csv"
    os.makedirs("logs", exist_ok=True)
    
    for i, item in enumerate(eval_data):
        print(f"\n[{i+1}/{len(eval_data)}] Running {item['type']} Query: {item['query'][:50]}...")
        
        start_time = time.time()
        
        try:
            response = run_rag(item['query'])
        except Exception as e:
            response = f"ERROR: {str(e)}"
            
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        results.append({
            "query_id": i+1,
            "type": item['type'],
            "query": item['query'],
            "target_source": item['target_source'],
            "response": response,
            "latency_sec": duration
        })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nEvaluation Complete! Results saved to {output_file}")
    print(df[['type', 'latency_sec']].groupby('type').mean())

if __name__ == "__main__":
    run_evaluation()