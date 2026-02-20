import sys
import os
import streamlit as st
import time
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_pipeline import get_enhanced_retriever, format_citations, log_interaction, LLM_MODEL

st.set_page_config(page_title="Personal Research Portal", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    with st.spinner("Loading Research Pipeline..."):
        st.session_state.retriever = get_enhanced_retriever()

llm = ChatOllama(model=LLM_MODEL, temperature=0)

system_prompt = """
You are a rigorous research assistant. Answer based ONLY on the provided context.
CITATION RULES:
1. Every claim must be immediately followed by a citation in the format [SourceID].
2. Do NOT use (Author, Year) format yourself. Use the ID.
3. If the context suggests an answer but isn't explicit, state your uncertainty and suggest a next retrieval step.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "Context:\n{context}\n\nQuestion: {question}")
])

chain = prompt_template | llm

with st.sidebar:
    st.title("Research Portal")
    st.markdown("### Tools & Export")
    
    if st.session_state.messages:
        chat_history = "\n\n".join([f"**{m['role'].capitalize()}**: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            label="Export Research Thread (MD)",
            data=chat_history,
            file_name="research_thread.md",
            mime="text/markdown"
        )
    
    st.markdown("---")
    st.markdown("### Artifact Generator")
    artifact_type = st.selectbox("Select Artifact", ["Synthesis Memo", "Evidence Table", "Annotated Bibliography"])
    if st.button("Generate Artifact"):
        st.info("Artifact generation logic will be triggered here based on current thread.")

st.title("Ask the Corpus")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your research question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence..."):
            start_time = time.time()
            
            docs = st.session_state.retriever.invoke(prompt)
            
            if not docs:
                st.warning("No relevant documents found. Please try broadening your search terms.")
                st.session_state.messages.append({"role": "assistant", "content": "No relevant documents found."})
            else:
                context_text = "\n\n".join([f"[{d.metadata.get('source_id')}]: {d.page_content}" for d in docs])
                
                with st.spinner("Synthesizing answer..."):
                    raw_response = chain.invoke({"context": context_text, "question": prompt}).content
                    
                    final_output = format_citations(raw_response, docs)
                    clean_output = re.sub(r'<think>.*?</think>', '', final_output, flags=re.DOTALL).strip()
                    
                    st.markdown(clean_output)
                    
                    with st.expander("View Retrieved Evidence"):
                        for d in docs:
                            st.markdown(f"**[{d.metadata.get('source_id')}]**: {d.page_content[:200]}...")
                    
                    end_time = time.time()
                    source_ids = [d.metadata.get('source_id', 'Unknown') for d in docs]
                    log_interaction(prompt, clean_output, source_ids, end_time - start_time)
                    
                    st.session_state.messages.append({"role": "assistant", "content": clean_output})