import sys
import os
import re
import csv
import time
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_pipeline import get_enhanced_retriever, format_citations, log_interaction, LLM_MODEL, MANIFEST_PATH

st.set_page_config(page_title="Advanced Research Portal", layout="wide", page_icon="🔬")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stChatInputContainer { padding-bottom: 20px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px; padding: 10px 16px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #e2e8f0; border-bottom: 2px solid #0f172a; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    with st.spinner("Initializing Hybrid Retriever & Reranker..."):
        st.session_state.retriever = get_enhanced_retriever()
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

llm = ChatOllama(model=LLM_MODEL, temperature=0)

base_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a rigorous research assistant. Answer ONLY based on the context. Every claim MUST end with a citation [SourceID]. State uncertainty if context is lacking."),
    ("user", "Context:\n{context}\n\nQuestion: {question}")
])
chain = base_prompt | llm

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research planner. Break the user's complex question into 3 distinct, search-optimized sub-queries. Return ONLY the sub-queries separated by a pipe (|)."),
    ("user", "Question: {question}")
])
planner_chain = planner_prompt | llm

gap_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a critical research reviewer. Analyze the provided context against the user's question. Identify exactly 3 research gaps. For each, state 'What is missing' and 'What evidence would resolve it'."),
    ("user", "Question: {question}\n\nCurrent Evidence Context:\n{context}")
])
gap_chain = gap_prompt | llm

artifact_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert academic writer. Generate a '{artifact_type}' based strictly on the provided context. Follow these specific rules: \n- Evidence Table: Markdown table with Claim | Evidence snippet | Citation (source_id) | Confidence | Notes.\n- Annotated Bibliography: List sources with 4 fields each (claim, method, limitations, why it matters).\n- Synthesis Memo: 800-1200 word memo with inline citations and a reference list.\nDo NOT hallucinate information."),
    ("user", "Context:\n{context}\n\nGenerate the artifact.")
])
artifact_chain = artifact_prompt | llm

def generate_bibtex():
    bibtex_str = ""
    if not os.path.exists(MANIFEST_PATH):
        return "No manifest found."
    with open(MANIFEST_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            s_id = row.get('source_id', 'unknown')
            authors = row.get('authors', 'Unknown')
            year = row.get('year', 'n.d.')
            title = row.get('title', 'Untitled')
            venue = row.get('venue', '')
            bibtex_str += f"@article{{{s_id},\n  author = {{{authors}}},\n  title = {{{title}}},\n  year = {{{year}}},\n  journal = {{{venue}}}\n}}\n\n"
    return bibtex_str

def create_knowledge_graph(docs, query):
    G = nx.Graph()
    G.add_node(query, size=20, color='#ef4444', type='query')
    
    for d in docs:
        s_id = d.metadata.get('source_id', 'Unknown')
        authors = d.metadata.get('authors', 'Unknown')
        G.add_node(s_id, size=15, color='#3b82f6', type='source')
        G.add_edge(query, s_id)
        
        for author in [a.strip() for a in authors.split(',')]:
            if author:
                G.add_node(author, size=10, color='#10b981', type='author')
                G.add_edge(s_id, author)
                
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    
    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(G.nodes[node]['color'])
        node_text.append(node)
        node_size.append(G.nodes[node]['size'])
        
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="bottom center",
        hoverinfo='text', marker=dict(color=node_color, size=node_size, line_width=2)
    )
    
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False, hovermode='closest', margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    ))
    return fig

with st.sidebar:
    st.title("🔬 Portal Tools")
    st.markdown("---")
    agentic_mode = st.toggle("Enable Agentic Deep Loop", value=False)
    
    st.markdown("### Export Capabilities")
    
    chat_md = "\n\n".join([f"**{m['role'].capitalize()}**: {m['content']}" for m in st.session_state.messages]) if st.session_state.messages else "No conversation history yet."
    
    st.download_button(
        label="Export Thread (Markdown)", 
        data=chat_md, 
        file_name="research_thread.md", 
        mime="text/markdown", 
        width="stretch",
        disabled=len(st.session_state.messages) == 0
    )
    
    bibtex_data = generate_bibtex()
    st.download_button("Export Corpus (BibTeX)", bibtex_data, "references.bib", "text/plain", width="stretch")
    
    st.markdown("---")
    st.markdown("### Artifact Generator")
    artifact_type = st.selectbox("Schema", ["Evidence Table", "Annotated Bibliography", "Synthesis Memo"])
    if st.button("Generate Artifact", width="stretch"):
        if st.session_state.last_docs:
            with st.spinner(f"Generating {artifact_type}..."):
                ctx = "\n\n".join([f"[{d.metadata.get('source_id')}]: {d.page_content}" for d in st.session_state.last_docs])
                raw_artifact = artifact_chain.invoke({"context": ctx, "artifact_type": artifact_type}).content
                clean_artifact = re.sub(r'<think>.*?</think>', '', raw_artifact, flags=re.DOTALL).strip()
                st.session_state.messages.append({"role": "assistant", "content": f"**Generated Artifact: {artifact_type}**\n\n{clean_artifact}"})
                st.rerun()
        else:
            st.warning("Please run a query first to retrieve evidence for the artifact.")

st.title("Personal Research Portal")

tab_chat, tab_graph, tab_gaps = st.tabs(["💬 Synthesis Chat", "🕸️ Knowledge Graph", "🔍 Gap Finder"])

with tab_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your main research question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.last_query = prompt
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            start_time = time.time()
            all_docs = []
            
            if agentic_mode:
                with st.status("Running Agentic Research Loop...", expanded=True) as status:
                    st.write("🧠 Planning sub-queries...")
                    plan_raw = planner_chain.invoke({"question": prompt}).content
                    sub_queries = [q.strip() for q in plan_raw.split('|') if q.strip()]
                    
                    for sq in sub_queries:
                        st.write(f"🔎 Searching: *{sq}*")
                        docs = st.session_state.retriever.invoke(sq)
                        all_docs.extend(docs)
                        
                    st.write("📚 Synthesizing cross-source evidence...")
                    status.update(label="Agentic Loop Complete!", state="complete", expanded=False)
            else:
                with st.spinner("Retrieving & Reranking..."):
                    all_docs = st.session_state.retriever.invoke(prompt)

            unique_docs = {d.page_content: d for d in all_docs}.values()
            st.session_state.last_docs = list(unique_docs)
            
            if not unique_docs:
                st.warning("No relevant documents found. Please adjust your query.")
                st.session_state.messages.append({"role": "assistant", "content": "No relevant documents found."})
            else:
                context_text = "\n\n".join([f"[{d.metadata.get('source_id')}]: {d.page_content}" for d in unique_docs])
                
                with st.spinner("Generating rigorous response..."):
                    raw_response = chain.invoke({"context": context_text, "question": prompt}).content
                    final_output = format_citations(raw_response, list(unique_docs))
                    clean_output = re.sub(r'<think>.*?</think>', '', final_output, flags=re.DOTALL).strip()
                    
                    st.markdown(clean_output)
                    
                    with st.expander(f"View Retrieved Evidence ({len(unique_docs)} chunks)"):
                        for d in unique_docs:
                            st.markdown(f"**[{d.metadata.get('source_id')}]**: {d.page_content[:250]}...")
                    
                    end_time = time.time()
                    source_ids = list(set([d.metadata.get('source_id', 'Unknown') for d in unique_docs]))
                    log_interaction(prompt, clean_output, source_ids, end_time - start_time)
                    st.session_state.messages.append({"role": "assistant", "content": clean_output})

with tab_graph:
    st.markdown("### Entity & Source Relationships")
    if st.session_state.last_docs and st.session_state.last_query:
        fig = create_knowledge_graph(st.session_state.last_docs, st.session_state.last_query)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Ask a question in the Synthesis Chat to generate a knowledge graph of the retrieved evidence.")

with tab_gaps:
    st.markdown("### Automated Disagreement & Gap Analysis")
    if st.session_state.last_docs and st.session_state.last_query:
        if st.button("Identify Gaps in Current Context", type="primary", width="stretch"):
            with st.spinner("Analyzing cross-source logic..."):
                context_text = "\n\n".join([f"[{d.metadata.get('source_id')}]: {d.page_content}" for d in st.session_state.last_docs])
                gap_analysis = gap_chain.invoke({"context": context_text, "question": st.session_state.last_query}).content
                clean_gaps = re.sub(r'<think>.*?</think>', '', gap_analysis, flags=re.DOTALL).strip()
                st.markdown(clean_gaps)
    else:
        st.info("Run a query first to analyze missing evidence.")