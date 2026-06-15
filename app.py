#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gradio as gr
import sys
from pathlib import Path
import networkx as nx
from typing import List, Dict
from collections import defaultdict
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import json
import plotly.graph_objects as go

from hybrid_rag import HybridRAG, Chunk, RAGConfig, build_all_chunks

# Paths (adjust to your actual paths)
CORPUS_DIR = "data/processed/corpus/"
LEXICON_PATH = "data/lexicons/no-llm/"
NER_DIR = "data/ner/gliner/"
CHUNK_DIR = "data/retrieval_results/RAG"

sys.modules['__main__'].Chunk = Chunk

_rag = None
_full_graph = None

# ============================================================
# =============== LOAD CORPUS, LEXICON, NER ==================
# ============================================================
def load_corpus(directory: str) -> Dict[str, Dict[str, str]]:
    corpus_docs = defaultdict(dict)
    csv_files = ['AgroEcology-Abstracts.csv']
    if not csv_files:
        corpus_docs["corpus_default"]["doc_default"] = "Default document."
    for csv_file in tqdm(csv_files, desc="Loading CSV files"):
        try:
            df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.read_csv(os.path.join(directory, csv_file))
            if not all(col in df.columns for col in ['id', 'source', 'text']):
                continue
            for _, row in df.iterrows():
                corpus_id = str(row['source'])
                doc_id = str(row['id'])
                text = str(row['text']) if pd.notna(row['text']) else ""
                corpus_docs[corpus_id][doc_id] = text
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    return corpus_docs

def load_lexicon(path: str):
    all_terms = set()
    lexicon_files = ['AgroEcology_Abstracts_biotex_terms.csv']
    for lex_file in tqdm(lexicon_files, desc="Loading lexicon"):
        try:
            df_lex = pd.read_csv(os.path.join(path, lex_file))
            col = 'terms' if 'terms' in df_lex.columns else 'keyword' if 'keyword' in df_lex.columns else None
            if col:
                all_terms.update(df_lex[col].astype(str).tolist())
        except Exception as e:
            print(f"Error reading {lex_file}: {e}")
    return sorted(list(all_terms))

def load_ner_terms(folder: str):
    all_ner = set()
    ner_files = ['AgroEcology-Abstracts_entities_aggregated.csv']
    for nf in tqdm(ner_files, desc="Loading NER terms"):
        try:
            df = pd.read_csv(os.path.join(folder, nf))
            if 'entity_text' in df.columns:
                all_ner.update(df['entity_text'].astype(str).tolist())
        except Exception as e:
            print(f"Error reading {nf}: {e}")
    return sorted(list(all_ner))

# ============================================================
# =============== GRAPH VISUALIZATION ========================
# ============================================================
def plotly_network_graph(full_graph: nx.Graph, query: str, ranked_chunks: List[Dict],
                         max_other_entities: int = 5, max_other_lexicons: int = 5,
                         fig_width: int = 1200, fig_height: int = 800) -> go.Figure:
    if full_graph is None or len(full_graph.nodes()) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No graph available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=f"Query: {query}", width=fig_width, height=fig_height)
        return fig

    chunk_ids = []
    all_entities = []
    all_lexicons = []
    node_info = {}
    entity_to_chunks = defaultdict(set)
    lexicon_to_chunks = defaultdict(set)

    for ch in ranked_chunks:
        cid = ch['chunk_id']
        chunk_ids.append(cid)
        for e in ch.get('matched_entities', []):
            entity_to_chunks[e].add(cid)
            if e not in node_info or node_info[e][1] > 1:
                node_info[e] = ("ENT", 1, "#e41a1c", "matched entity", 40)
        for e in ch.get('neighbour_entities', []):
            entity_to_chunks[e].add(cid)
            if e not in node_info or node_info[e][1] > 2:
                node_info[e] = ("ENT", 2, "#ff7f0e", "neighbour entity", 30)
        for l in ch.get('matched_lexicons', []):
            lexicon_to_chunks[l].add(cid)
            if l not in node_info or node_info[l][1] > 3:
                node_info[l] = ("LEX", 3, "#2ca02c", "matched lexicon", 30)
        all_entities.extend(ch.get('entities', []))
        all_lexicons.extend(ch.get('lexicon_hits', []))

    from collections import Counter
    entity_counter = Counter(all_entities)
    for label in node_info:
        if node_info[label][0] == "ENT" and label in entity_counter:
            del entity_counter[label]
    top_other_entities = [e for e, _ in entity_counter.most_common(max_other_entities)]
    for e in top_other_entities:
        if e not in node_info or node_info[e][1] > 4:
            node_info[e] = ("ENT", 4, "#17becf", "other entity (top)", 25)

    lexicon_counter = Counter(all_lexicons)
    for label in node_info:
        if node_info[label][0] == "LEX" and label in lexicon_counter:
            del lexicon_counter[label]
    top_other_lexicons = [l for l, _ in lexicon_counter.most_common(max_other_lexicons)]
    for l in top_other_lexicons:
        if l not in node_info or node_info[l][1] > 5:
            node_info[l] = ("LEX", 5, "#9467bd", "other lexicon (top)", 25)

    G_sub = nx.Graph()
    for cid in chunk_ids:
        G_sub.add_node(cid, kind="chunk", label=cid[:30] + "..." if len(cid) > 30 else cid)

    for label, (node_type, priority, color, kind, size) in node_info.items():
        if node_type == "ENT":
            node_name = f"ENT::{label}"
            G_sub.add_node(node_name, kind=kind, label=label, color=color, size=size)
            for cid in entity_to_chunks.get(label, []):
                if cid in G_sub:
                    G_sub.add_edge(node_name, cid, weight=1)
        else:
            node_name = f"LEX::{label}"
            G_sub.add_node(node_name, kind=kind, label=label, color=color, size=size)
            for cid in lexicon_to_chunks.get(label, []):
                if cid in G_sub:
                    G_sub.add_edge(node_name, cid, weight=1)

    if len(G_sub.nodes()) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No nodes to display", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=f"Query: {query}", width=fig_width, height=fig_height)
        return fig

    pos = nx.spring_layout(G_sub, k=3.0, iterations=200, seed=42)

    node_x, node_y, node_colors, node_sizes, hover_texts = [], [], [], [], []
    label_texts = []

    for node in G_sub.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)

        if node.startswith("ENT::") or node.startswith("LEX::"):
            label = G_sub.nodes[node].get('label', node.split("::")[-1])
            color = G_sub.nodes[node].get('color', "#d3d3d3")
            kind = G_sub.nodes[node].get('kind', "entity")
            size = G_sub.nodes[node].get('size', 20)
        else:
            label = G_sub.nodes[node].get('label', node)
            color = "#1f77b4"
            kind = "chunk"
            size = 50

        node_colors.append(color)
        node_sizes.append(size)
        hover_texts.append(f"<b>{label}</b><br>Kind: {kind}")

        if len(label) < 40:
            label_texts.append(label)

    edge_traces = []
    for u, v in G_sub.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines', line=dict(width=1.5, color='lightgray'),
            hoverinfo='none', showlegend=False
        ))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=label_texts, textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=node_sizes, color=node_colors,
                    line=dict(width=1, color='black')),
        hoverinfo='text', hovertext=hover_texts, showlegend=False
    )

    legend_items = [
        ("Chunk", "#1f77b4"),
        ("Matched entity", "#e41a1c"),
        ("Neighbour entity", "#ff7f0e"),
        ("Matched lexicon", "#2ca02c"),
        ("Other entity (top)", "#17becf"),
        ("Other lexicon (top)", "#9467bd")
    ]
    legend_traces = []
    for name, color in legend_items:
        legend_traces.append(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color), name=name, showlegend=True
        ))

    fig = go.Figure(data=edge_traces + [node_trace] + legend_traces,
                    layout=go.Layout(
                        title=dict(text=f"<b>Knowledge Graph</b><br>"
                                        f"<span style='font-size:12px'>Based on retrieved chunks: matched entities (red), neighbour entities (orange), matched lexicons (green).</span><br>"
                                        f"Query: {query[:100]}{'...' if len(query)>100 else ''}",
                                   font=dict(size=14)),
                        showlegend=True,
                        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        plot_bgcolor='white',
                        margin=dict(l=30, r=100, b=30, t=100),
                        width=fig_width, height=fig_height
                    ))
    return fig

# ============================================================
# =============== RAG INITIALIZATION =========================
# ============================================================
def initialize_rag():
    global _full_graph
    config = RAGConfig(
        corpus_dir=CORPUS_DIR,
        lexicon_path=LEXICON_PATH,
        ner_dir=NER_DIR,
        chunk_cache_dir=CHUNK_DIR,
    )
    print("Loading corpus...")
    corpus = load_corpus(CORPUS_DIR)
    print("Loading lexicon...")
    lexicon_terms = load_lexicon(LEXICON_PATH)
    print("Loading NER terms...")
    ner_terms = load_ner_terms(NER_DIR)
    print("Building chunks...")
    chunks = build_all_chunks(corpus, ner_terms, lexicon_terms, config)
    print("Initializing HybridRAG...")
    rag = HybridRAG(chunks, entity_lexicon=ner_terms, lexicon=lexicon_terms, config=config)
    _full_graph = rag.graph.copy() if hasattr(rag, 'graph') else None
    return rag

def get_rag():
    global _rag
    if _rag is None:
        _rag = initialize_rag()
    return _rag

# =============== Formatting for other retrieval methods (unchanged) ===============
def highlight_ground_truth(chunk_text: str, relevant_passages: List[str]) -> str:
    if not relevant_passages:
        return chunk_text
    highlighted = chunk_text
    for passage in relevant_passages:
        highlighted = highlighted.replace(passage, f"<span style='color: blue; font-weight: bold;'>{passage}</span>")
    return highlighted

def format_results_generic(results, method_name, relevant_passages=None):
    if not results:
        return f"**{method_name}** returned no results."
    lines = [f"**Method: {method_name}**", ""]
    for i, res in enumerate(results, 1):
        chunk_text = res['text']
        if relevant_passages:
            chunk_text = highlight_ground_truth(chunk_text, relevant_passages)
        lines.append(f"### Result {i}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>Chunk ID:</span> {res['chunk_id']}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>Corpus:</span> {res['corpus_id']} | <span style='color: blue; font-weight: bold;'>Doc:</span> {res['doc_id']}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>Score:</span> {res['score']:.4f}")
        if 'bm25_score' in res:
            lines.append(f"<span style='color: blue; font-weight: bold;'>BM25 Score:</span> {res['bm25_score']:.4f}")
        if 'dense_score' in res:
            lines.append(f"<span style='color: blue; font-weight: bold;'>Dense Score:</span> {res['dense_score']:.4f}")
        lines.append(f"<span style='color: green; font-weight: bold;'>✅ Matched entities:</span> {', '.join(res.get('matched_entities', []))}")
        if res.get('neighbour_entities'):
            lines.append(f"<span style='color: orange; font-weight: bold;'>🔗 Neighbour entities (from KG):</span> {', '.join(res['neighbour_entities'])}")
        if res.get('matched_lexicons'):
            lines.append(f"<span style='color: green; font-weight: bold;'>✅ Fuzzy Lexicon Hits:</span> {', '.join(res['matched_lexicons'])}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>chunk entities:</span> {', '.join(res.get('entities', []))}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>Chunk Lexicons:</span> {', '.join(res.get('lexicon_hits', []))}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>Text:</span> {chunk_text}")
        if res.get('relevant_segment'):
            lines.append(f"<span style='color: orange; font-weight: bold;'>✅ Relevant Segment:</span> {res['relevant_segment']}")
        lines.append("")
    return "\n".join(lines)

def format_entity_matching_results(results, method_name, relevant_passages=None):
    if not results:
        return f"**{method_name}** returned no matching chunks."
    lines = [f"**Method: {method_name}**", ""]
    for i, res in enumerate(results, 1):
        chunk_text = res['text']
        if relevant_passages:
            chunk_text = highlight_ground_truth(chunk_text, relevant_passages)
        lines.append(f"### Result {i} (Matches: {res['score']})")
        lines.append(f"<span style='color: blue; font-weight: bold;'>Chunk ID:</span> {res['chunk_id']}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>Corpus:</span> {res['corpus_id']} | <span style='color: blue; font-weight: bold;'>Doc:</span> {res['doc_id']}")
        lines.append(f"<span style='color: green; font-weight: bold;'>✅ Matched entities:</span> {', '.join(res['matched_entities'])}")
        if res.get('neighbour_entities'):
            lines.append(f"<span style='color: orange; font-weight: bold;'>🔗 Neighbour entities (from KG):</span> {', '.join(res['neighbour_entities'])}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>All chunk entities:</span> {', '.join(res['entities'])}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>Lexicon hits:</span> {', '.join(res['lexicon_hits'])}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>Text:</span> {chunk_text}")
        if res.get('relevant_segment'):
            lines.append(f"<span style='color: orange; font-weight: bold;'>✅ Relevant Segment:</span> {res['relevant_segment']}")
        lines.append("")
    return "\n".join(lines)

def format_lexicon_matching_results(results, method_name, relevant_passages=None):
    if not results:
        return f"**{method_name}** returned no matching chunks."
    lines = [f"**Method: {method_name}**", ""]
    for i, res in enumerate(results, 1):
        chunk_text = res['text']
        if relevant_passages:
            chunk_text = highlight_ground_truth(chunk_text, relevant_passages)
        lines.append(f"### Result {i} (Score: {res['score']:.2f})")
        lines.append(f"<span style='color: blue; font-weight: bold;'>Chunk ID:</span> {res['chunk_id']}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>Corpus:</span> {res['corpus_id']} | <span style='color: blue; font-weight: bold;'>Doc:</span> {res['doc_id']}")
        if res.get('matched_lexicons'):
            lines.append(f"<span style='color: green; font-weight: bold;'>✅ Fuzzy Lexicon Hits:</span> {', '.join(res['matched_lexicons'])}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>All chunk lexicon hits:</span> {', '.join(res['lexicon_hits'])}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>Entities:</span> {', '.join(res['entities'])}")
        lines.append(f"<span style='color: blue; font-weight: bold;'>Text:</span> {chunk_text}")
        if res.get('relevant_segment'):
            lines.append(f"<span style='color: orange; font-weight: bold;'>✅ Relevant Segment:</span> {res['relevant_segment']}")
        lines.append("")
    return "\n".join(lines)

# =============== Main retrieval dispatcher ===============
def retrieve_and_display(query, method, segment_method, llm_model):
    if not query or not query.strip():
        empty_fig = plotly_network_graph(None, "", [])
        return "Please enter a query.", "", empty_fig, None, gr.update(visible=False)
    try:
        rag = get_rag()
        top_k = 5

        if method == "Reranked Hybrid":
            result = rag.retrieve(query, use_llm_augmentation=True, use_graph_expansion=True)
            augmented = result.get("augmented_query", "")
            researcher = result.get("researcher_summary", "Not available.")
            agronomist = result.get("agronomist_summary", "Not available.")
            farmer = result.get("farmer_summary", "Not available.")
            chunks = result.get("chunks", [])

            # Build the output markdown
            lines = []
            lines.append(f"**🔍 Augmented Query:** {augmented}\n")
            lines.append("## 📚 Researcher Summary")
            lines.append(researcher)
            lines.append("\n## 🌾 Agronomist Summary")
            lines.append(agronomist)
            lines.append("\n## 👨‍🌾 Farmer Summary")
            lines.append(farmer)
            lines.append("\n---\n")
            lines.append("### 📂 Routed Corpora")
            try:
                routed = rag.route_corpora(query, top_k=3)
                for cid, score in routed:
                    lines.append(f"- {cid}: {score:.4f}")
            except:
                lines.append("Not available")
            lines.append("\n### 📄 Retrieved Chunks (Details)\n")
            for i, ch in enumerate(chunks, 1):
                lines.append(f"#### Chunk {i}: `{ch['chunk_id']}`")
                lines.append(f"- **Doc ID:** {ch.get('doc_id', 'N/A')}")
                lines.append(f"- **Entities:** {', '.join(ch.get('entities', []))}")
                lines.append(f"- **Lexicons:** {', '.join(ch.get('lexicons', []))}")
                lines.append(f"- **Matched Entities:** {', '.join(ch.get('matched_entities', []))}")
                lines.append(f"- **Matched Lexicons:** {', '.join(ch.get('matched_lexicons', []))}")
                lines.append(f"- **Neighbour Entities:** {', '.join(ch.get('neighbour_entities', []))}")
                lines.append(f"- **Relevant Segment:**\n  {ch.get('relevant_segment', '')}")
                lines.append(f"- **Full Text (first 800 chars):**\n  {ch.get('text', '')[:800]}...\n")
            output_md = "\n".join(lines)

            citations = "<br>".join(f"<span style='color: blue;'>[{i}]</span> {ch['chunk_id']}"
                                    for i, ch in enumerate(chunks, 1))
            fig = plotly_network_graph(_full_graph, query, chunks)
            return output_md, citations, fig, fig, gr.update(visible=True)

        else:
            # Other methods (BM25, Dense, etc.) – unchanged
            seg_method_map = {
                "Sentence Embedding": "sentence_embedding",
                "Cross Encoder": "cross_encoder",
                "Text Tiling": "texttiling",
                "Topic Tiling": "topictiling",
                "Graph Segmentation": "graphseg",
                "LLM": "phi-3.1-mini-4k-instruct"
            }
            seg_method = seg_method_map.get(segment_method, "sentence_embedding")
            if method == "BM25 only":
                results = rag.retrieve_bm25_only(query, top_k=top_k, seg_method=seg_method, llm_model=llm_model)
                formatted = format_results_generic(results, "BM25 only")
            elif method == "Dense only":
                results = rag.retrieve_dense_only(query, top_k=top_k, seg_method=seg_method, llm_model=llm_model)
                formatted = format_results_generic(results, "Dense only")
            elif method == "Hybrid (alpha=0.5)":
                results = rag.retrieve_hybrid_bm25_dense(query, top_k=top_k, alpha=0.5, seg_method=seg_method, llm_model=llm_model)
                formatted = format_results_generic(results, "Hybrid (alpha=0.5)")
            elif method == "Entity matching":
                results = rag.retrieve_entity_matching(query, top_k=top_k, seg_method=seg_method, llm_model=llm_model)
                formatted = format_entity_matching_results(results, "Entity matching")
            elif method == "Lexicon matching":
                results = rag.retrieve_lexicon_matching(query, top_k=top_k, seg_method=seg_method, llm_model=llm_model)
                formatted = format_lexicon_matching_results(results, "Lexicon matching")
            elif method == "Weighted hybrid":
                results = rag.retrieve_weighted_hybrid(query, top_k=top_k, seg_method=seg_method, llm_model=llm_model)
                formatted = format_results_generic(results, "Weighted hybrid")
            elif method == "Graph expansion":
                results = rag.retrieve_with_graph_expansion(query, top_k=top_k)
                formatted = format_results_generic(results, "Graph expansion")
            else:
                results = rag.retrieve_bm25_only(query, top_k=top_k, seg_method=seg_method, llm_model=llm_model)
                formatted = format_results_generic(results, "BM25 only")
            citations = "<br>".join(f"<span style='color: blue;'>[{i}]</span> {r['chunk_id']} (score: {r['score']:.4f})"
                                    for i, r in enumerate(results, 1))
            fig = plotly_network_graph(_full_graph, query, results)
            return formatted, citations, fig, fig, gr.update(visible=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        empty_fig = plotly_network_graph(None, "", [])
        return f"Error: {str(e)}", "", empty_fig, None, gr.update(visible=False)

def save_graph_as_png(fig):
    if fig is None:
        return None
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.write_image(tmp.name, width=1600, height=1200, scale=2)
            return tmp.name
    except Exception as e:
        print(f"Error saving PNG: {e}")
        return None
# =============== Gradio Interface ===============
with gr.Blocks(title="AgroEcology RAG", css=".scrollable-plot { overflow: auto; }") as demo:
    gr.Markdown("# 🌾 AgroEcology RAG: From Documents to Structured Answers")
    with gr.Tabs():
        with gr.TabItem("🔍 Query"):
            with gr.Row():
                with gr.Column(scale=1):
                    query_input = gr.Textbox(label="Query", lines=3, placeholder="Enter your question...")
                    method_dropdown = gr.Dropdown(
                        choices=["BM25 only", "Dense only", "Hybrid (alpha=0.5)",
                                 "Entity matching", "Lexicon matching",
                                 "Weighted hybrid", "Graph expansion", "Reranked Hybrid"],
                        value="Reranked Hybrid",
                        label="Retrieval Method"
                    )
                    segment_dropdown = gr.Dropdown(
                        choices=["Sentence Embedding", "Cross Encoder", "LLM"],
                        value="Sentence Embedding",
                        label="Segmentation Method"
                    )
                    llm_model_dropdown = gr.Dropdown(
                        choices=["phi-3.1-mini-4k-instruct", "Llama-3.2-3B-Instruct"],
                        value="phi-3.1-mini-4k-instruct",
                        label="LLM Model"
                    )
                    submit_btn = gr.Button("Retrieve")
                    download_btn = gr.Button("📸 Save Graph", visible=False)
                    download_output = gr.File(visible=False)
                with gr.Column(scale=2):
                    output_md = gr.Markdown(label="Results")
                    output_citations = gr.Markdown(label="Chunk References")
            with gr.Row(visible=False) as plot_container:
                output_plot = gr.Plot(label="Knowledge Graph")
            fig_state = gr.State()
            submit_btn.click(
                fn=retrieve_and_display,
                inputs=[query_input, method_dropdown, segment_dropdown, llm_model_dropdown],
                outputs=[output_md, output_citations, output_plot, fig_state, plot_container]
            )
            download_btn.click(fn=save_graph_as_png, inputs=[fig_state], outputs=download_output)
            gr.Examples(
                examples=[
                    ["What are the principles and practices of agroecology?"],
                    ["How do vineyard water stress and yield vary spatially across Occitanie?"],
                    ["What are the spatio‑temporal dynamics of food security in West Africa?"]
                ],
                inputs=query_input,
                label="Example Queries"
            )

if __name__ == "__main__":
    demo.launch(share=True)