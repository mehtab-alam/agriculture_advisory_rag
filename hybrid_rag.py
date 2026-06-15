#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:03:31 2026

@author: syed
"""

import os
import re
import json
import pickle
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple
#from codecarbon import EmissionsTracker
import numpy as np
import pandas as pd
import spacy
import torch
import nltk
from pathlib import Path
from collections import Counter, deque, defaultdict
from tqdm import tqdm
from multiprocessing import Pool
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi
from gliner import GLiNER
import glob
import ollama
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import time  # for response timing
from functools import lru_cache
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000 

# ============================================================
# Configuration
# ============================================================
@dataclass
class RAGConfig:
    corpus_dir: str = "data/processed/corpus"
    lexicon_path: str = "data/lexicons/no-llm/"
    ner_dir: str = "data/ner/gliner/"
    lexicon_dir: str = "data/lexicon"
    rag_dir: str = "data/retrieval_results/RAG"
    chunk_cache_dir: str = "data/retrieval_results/RAG"
    chunk_size: int = 600
    chunk_overlap: int = 150
    embed_model: str =  "sentence-transformers/all-MiniLM-L6-v2" #"BAAI/bge-small-en-v1.5" #"sentence-transformers/all-MiniLM-L6-v2"
    ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gliner_model: str = "urchade/gliner_multi"
    llm_base_url: str = "http://localhost:1234/v1/chat/completions"
    llm_model_name: str = "phi-3.1-mini-4k-instruct"
    default_top_k: int = 10
    weights: Dict[str, float] = field(default_factory=lambda: {
        'dense': 0.3, 'bm25': 0.2, 'entity': 0.3, 'lexicon': 0.2
    })
    chunks_files = ['Dataverse-Organic Waste Management.pkl', 'AgroEcology-Abstracts.pkl']
    default_seg_method: str = "cross_encoder"
    max_llm_tokens: int = 800
    use_cuda: bool = True
    num_workers: int = 1
    confidence_threshold: float = 0.5

    def __post_init__(self):
        os.makedirs(self.rag_dir, exist_ok=True)
        os.makedirs(self.chunk_cache_dir, exist_ok=True)



config = RAGConfig()
_embedding_model = None
_current_model_name = None


def lexicon_based_router(
    query: str,
    corpus_ids: List[str],
    config: RAGConfig,
    stopwords: set = None
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Route a query to the most relevant corpus based on lexicon term overlap.
    
    Args:
        query: User query string.
        corpus_ids: List of corpus IDs (e.g., ["Dataverse-Organic Waste Management", "AgroEcology-Abstracts"]).
        config: RAGConfig object (to access lexicon_dir).
        stopwords: Optional set of stopwords to ignore.
    
    Returns:
        Tuple of (best_corpus_id, dict of scores for all corpora).
    """
    if stopwords is None:
        stopwords = set(["a", "an", "the", "is", "are", "was", "were", "to", "of", "and", "in", "for", "on", "with", "at", "by"])
    
    # Preprocess query: lowercased tokens, remove stopwords
    query_tokens = set(word.lower() for word in query.split() if word.lower() not in stopwords)
    
    scores = {}
    for cid in corpus_ids:
        # Load lexicon terms for this corpus
        term_set, keyword_set = load_lexicon_terms_all_sheets(cid)  # term_set is set of lowercased terms
        # Also include keywords (sheet names) if they are meaningful
        all_terms = term_set.union(keyword_set)
        
        # Calculate overlap: number of lexicon terms that appear in the query (as substrings or exact words)
        # Simple exact token match:
        overlap = len(query_tokens.intersection(all_terms))
        
        # Also count multi-word terms that appear as substrings in the query
        # (because queries may contain phrases like "organic waste management")
        for term in all_terms:
            if ' ' in term and term.lower() in query.lower():
                overlap += 1
        
        # Normalise by the size of the lexicon (optional, to avoid bias towards larger lexicons)
        # Here we just use raw overlap, which works well if lexicons are similar size.
        scores[cid] = overlap
    
    # If all scores are zero, return None (no clear routing)
    if max(scores.values()) == 0:
        print("[Lexicon Router] No term overlap found. Returning None.")
        return None, scores
    
    best_corpus = max(scores, key=scores.get)
    print(f"[Lexicon Router] Selected '{best_corpus}' with score {scores[best_corpus]}")
    return best_corpus, scores

class ConversationMemory:
    def __init__(self, max_length=3):
        self.history = deque(maxlen=max_length)
    
    def add(self, user_query, assistant_answer):
        self.history.append({"user": user_query, "assistant": assistant_answer})
    
    def clear(self):
        self.history.clear()
    
    def get_messages(self):
        """Return list of message dicts for Ollama chat, excluding system prompts."""
        messages = []
        for entry in self.history:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
        return messages


@lru_cache(maxsize=None)
def load_lexicon_terms_all_sheets(corpus_id: str) -> set:
    """
    Load all sheets from {corpus_id}.xlsx and return a set of all terms (lowercased).
    Ignores the 'rank' column; all terms are treated equally.
    """
    file_path = f"{config.lexicon_dir}/{corpus_id}.xlsx"
    # Get all sheet names
    print(f"opening file : {file_path}")
    xl = pd.ExcelFile(file_path)
    print(f"Lexicon file Opened: {xl.sheet_names}")
    all_terms = set()
    all_keywords = set()
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Sheet Name:{sheet_name}")
        all_keywords.add(sheet_name)
        
        if 'term' in df.columns:
            terms = df['term'].dropna().astype(str).str.strip().str.lower().tolist()
            all_terms.update(terms)
    return all_terms, all_keywords

def clean_units(text: str) -> str:
    """
    Normalize units in agricultural/technical text.
    Converts corrupted or inconsistent notation to standard form.
    
    Examples:
        "t day?1"        → "t day⁻¹"
        "CO2 e t?1 m"    → "CO₂e t⁻¹"
        "kg ha-1"        → "kg ha⁻¹"
        "degC"           → "°C"
        "N 20 kg/ha"     → "N 20 kg ha⁻¹" (optional)
    """
    # Ordered list of (regex_pattern, replacement)
    #print("before cleaning", text )
    patterns = [
        # ---------- Superscript/subscript corruptions ----------
        # "?1" (question mark + digit) -> superscript minus/digit
        (r'\?1\b', '⁻¹'),          # standalone ?1
        (r'\?2\b', '⁻²'),
        (r'\?3\b', '⁻³'),
        (r'\?(\d+)', r'⁻\1'),      # generic ?n -> ⁻ⁿ (for n>3)
        # "^1", "^2", "_1" (caret/underscore followed by digit)
        (r'\^1\b', '¹'),           # positive superscript (e.g., m^2 -> m²)
        (r'\^2\b', '²'),
        (r'\^3\b', '³'),
        (r'_1\b', '₁'),            # subscript (e.g., CO_2 -> CO₂)
        (r'_2\b', '₂'),
        (r'_3\b', '₃'),
        (r'(_\d+)', lambda m: m.group(1).replace('_', '').translate(str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉'))),
        # ---------- Specific unit corrections ----------
        # "CO2" -> "CO₂" (subscript 2)
        (r'CO2', 'CO₂'),
        # "degC", "deg C" -> "°C"
        (r'deg\s*C', '°C'),
        (r'°\s*C', '°C'),          # already has ° but space
        # "per" or "/" to superscript -1 or explicit per
        (r'\bper\s+(ha|kg|t|g|mg|L|mL)\b', r' \1⁻¹'),   # "per ha" -> "ha⁻¹"
        (r'/(?=\S)', '⁻¹'),        # slash before unit -> superscript -1, but careful
        # Better: handle "kg/ha" -> "kg ha⁻¹"
        (r'(\w+)/(\w+)', r'\1 \2⁻¹'),
        # "ha-1", "kg-1" (hyphen + digit) -> superscript minus
        (r'(\b\w+)-(\d+)\b', r'\1⁻\2'),
        # ---------- Stray characters & cleanup ----------
        # Remove stray trailing "m" after t⁻¹ (as in "t?1 m" -> "t⁻¹")
        (r'(t⁻¹)\s+m\b', r'\1'),
        # Remove orphaned "?" left after superscript correction
        
        (r'meq/100g', 'meq 100 g⁻¹'),
        (r'mg/kg', 'mg kg⁻¹'),
        
        
        
        # Area: "m2", "m^2" -> "m²"
        (r'm\^?2\b', 'm²'),
        
        # Volume: "l", "L" (keep uppercase L for liter)
        (r'\bl\b', 'L'),
    ]
    
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    
    # Optional: normalise spacing around units (e.g., "20kg" -> "20 kg")
    text = re.sub(r'(\d)([A-Za-z°])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # "CO2" already handled
    
    #print("After cleaning", text )
    # Ensure minus sign is superscript for per‑time/area (optional)
    # Example: "t day⁻¹" is fine; "t day-1" already fixed by earlier patterns
    return text.strip()

@dataclass
class Chunk:
    corpus_id: str
    doc_id: str
    chunk_id: str
    text: str
    lexicon: []
    keuwords: []
    segments: str = None
    metadata: Optional[dict] = None





def add_metadata(chunks: List[Chunk], filename: str, config) -> None:
    """
    Read metadata from CSV and attach to each chunk where chunk.doc_id matches CSV 'id'.
    The CSV is expected to have columns: 'id', 'metadata' (JSON string or dict literal).
    """
    metadata_file = f"{config.corpus_dir}/{filename}.csv"
    df = pd.read_csv(metadata_file)
    
    # Build fast lookup: doc_id -> metadata dict
    doc_to_metadata = {}
    for _, row in df.iterrows():
        doc_id = row['id']
        metadata_str = row['metadata']
        try:
            # Try parsing as JSON first (common in modern CSV)
            metadata_dict = json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            # Fallback to ast.literal_eval for Python dict literals
            import ast
            metadata_dict = ast.literal_eval(metadata_str)
        doc_to_metadata[doc_id] = metadata_dict
    
    # Attach metadata to chunks
    for chunk in chunks:
        if chunk.doc_id in doc_to_metadata:
            chunk.metadata = doc_to_metadata[chunk.doc_id]
    return chunks
    

def load_chunks(config):
    chunk_files= glob.glob(f"{config.chunk_cache_dir}/*.pkl")
    all_chunks = []
    for f_chunks in chunk_files[3:4]:#[0:1]: #[1:2] [2:3] [3:4]CEA-first
        file_name = os.path.splitext(os.path.basename(f_chunks))[0]
        is_metadata_updated = False
        with open(f_chunks, 'rb') as f:
            chunks = pickle.load(f)
            [setattr(chunk, 'text', clean_units(chunk.text)) for chunk in chunks]
            print(f"Loaded of {file_name}: {len(chunks)} chunks")
            if not chunks or chunks[0].metadata is None:
                is_metadata_updated = True
                chunks = add_metadata(chunks, file_name, config)    
        all_chunks.extend(chunks)  
        if is_metadata_updated:
            with open(f_chunks, 'wb') as f:
                pickle.dump(chunks, f)
    return all_chunks



# --------------------------------------------
# 2. Retrieve top-k chunks using sentence embeddings
# --------------------------------------------

def get_embedding_model(model_name):
    global _embedding_model, _current_model_name
    if _embedding_model is None or _current_model_name != model_name:
        print(f"Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
        _embedding_model.to(DEVICE)
        _current_model_name = model_name
    return _embedding_model

def build_corpus_embeddings(
    chunks: List[Chunk],
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    cache_dir: str = ".embedding_cache"
) -> Dict[str, torch.Tensor]:
    """
    Build a single embedding for each corpus by averaging all chunk embeddings within that corpus.
    Returns dictionary: corpus_id -> embedding tensor (1 x D).
    """
    # Group chunks by corpus_id
    corpus_chunks = defaultdict(list)
    for chunk in chunks:
        if chunk.corpus_id:
            corpus_chunks[chunk.corpus_id].append(chunk)
    
    model = get_embedding_model(model_name)
    corpus_embeddings = {}
    
    for cid, chunk_list in corpus_chunks.items():
        # Get embeddings for all chunks in this corpus (using cached embeddings)
        chunk_embeddings = get_chunk_embeddings_cached(chunk_list, model_name, cache_dir)
        # Average along the chunk dimension (axis=0)
        corpus_embedding = torch.mean(chunk_embeddings, dim=0, keepdim=True)
        corpus_embeddings[cid] = corpus_embedding
        print(f"Built embedding for corpus '{cid}' from {len(chunk_list)} chunks.")
    
    return corpus_embeddings

def get_chunk_embeddings_cached(chunks, model_name, cache_dir=".embedding_cache"):
    """
    Returns (chunk_embeddings_tensor, chunk_list) where chunk_list is the same order as passed.
    Uses disk cache to avoid recomputing.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a signature: hash of all chunk texts concatenated + model name
    texts_concat = "".join(chunk.text for chunk in chunks)
    signature = hashlib.md5((texts_concat + model_name).encode('utf-8')).hexdigest()
    cache_file = Path(cache_dir) / f"embeddings_{signature}.pkl"
    
    if cache_file.exists():
        print(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        # data should contain: embeddings (numpy array or tensor), and the chunks list
        # We also need to ensure that the chunks list matches the current `chunks` order.
        # If the corpus changed, signature will be different, so we are safe.
        chunk_embeddings = data['embeddings']
        # If you stored as numpy, convert to torch tensor if needed
        if not isinstance(chunk_embeddings, torch.Tensor):
            chunk_embeddings = torch.tensor(chunk_embeddings).to(DEVICE)
        return chunk_embeddings
    else:
        print(f"Computing embeddings for {len(chunks)} chunks...")
        model = get_embedding_model(model_name)
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True)
        # Save to cache (convert to numpy to save space, or keep as tensor)
        data = {
            'embeddings': chunk_embeddings.cpu().numpy(),  # save as numpy for portability
            'chunks': chunks  # optional, but may be large; we don't really need it if we trust order
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved embeddings to {cache_file}")
        return chunk_embeddings

def route_query_with_embeddings(
    query: str,
    corpus_embeddings: Dict[str, torch.Tensor],
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    threshold: float = 0.5
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Route a query to the most relevant corpus using cosine similarity with corpus embeddings.

    Args:
        query: User query string.
        corpus_embeddings: Dictionary of corpus_id -> embedding tensor (1 x D).
        model_name: Name of the SentenceTransformer model.
        threshold: Minimum similarity required to route. If best similarity < threshold, return None.

    Returns:
        Tuple of (best_corpus_id or None, dict of similarity scores).
    """
    # Encode query
    model = get_embedding_model(model_name)
    query_embedding = model.encode(query, convert_to_tensor=True).unsqueeze(0)  # 1 x D

    scores = {}
    for cid, corpus_emb in corpus_embeddings.items():
        # Cosine similarity
        sim = util.cos_sim(query_embedding, corpus_emb).item()
        scores[cid] = sim

    if not scores:
        return None, {}

    best_cid = max(scores, key=scores.get)
    best_score = scores[best_cid]

    if best_score < threshold:
        print(f"[Embedding Router] Best similarity {best_score:.3f} below threshold {threshold} of Corpus {best_cid}")
        return best_cid, scores

    print(f"[Embedding Router] Selected '{best_cid}' with similarity {best_score:.3f}")
    return best_cid, scores
'''
def retrieve_chunks_cached(query, chunks, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=15, cache_dir=".embedding_cache"):
    # Get cached embeddings (or compute once)
    chunk_embeddings = get_chunk_embeddings_cached(chunks, model_name, cache_dir)
    
    # Encode query
    model = get_embedding_model(model_name)
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute similarities
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = torch.argsort(similarities, descending=True)[:top_k].cpu().numpy()
    retrieved = [chunks[i] for i in top_indices]
    
    # Print retrieved details (same as before)
    print("\n" + "="*80)
    print(f"RETRIEVED TOP {len(retrieved)} CHUNKS (ranked by similarity):")
    for rank, chunk in enumerate(retrieved, 1):
        preview = chunk.text.replace('\n', ' ')[:200] + "..."
        print(f"\n[Rank {rank}] Chunk: {chunk.chunk_id} (doc: {chunk.doc_id})")
        print(f"  Preview: {preview}")
    print("="*80 + "\n")
    
    return retrieved
'''



def retrieve_chunks_cached(query, chunks, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=15, similarity_threshold=0.0, cache_dir=".embedding_cache"):
    """
    Retrieve chunks relevant to the query using cached embeddings.

    Args:
        query (str): The query string.
        chunks (list): List of chunk objects with .text, .chunk_id, .doc_id attributes.
        model_name (str): SentenceTransformer model name.
        top_k (int): Maximum number of chunks to return (ignored if None, returns all above threshold).
        similarity_threshold (float): Minimum cosine similarity to keep a chunk (default 0.0 = keep all).
        cache_dir (str): Directory for caching embeddings.

    Returns:
        list: Filtered list of chunk objects (ordered by decreasing similarity).
    """
    # Get cached embeddings (or compute once)
    chunk_embeddings = get_chunk_embeddings_cached(chunks, model_name, cache_dir)
    
    # Encode query
    model = get_embedding_model(model_name)
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute similarities
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]  # shape (len(chunks),)
    
    # Convert to numpy for easier indexing
    sims_np = similarities.cpu().numpy()
    
    # Apply threshold: keep indices where similarity >= threshold
    valid_indices = [i for i, s in enumerate(sims_np) if s >= similarity_threshold]
    
    if not valid_indices:
        print("\n" + "="*80)
        print(f"WARNING: No chunks passed the similarity threshold {similarity_threshold}.")
        print("="*80 + "\n")
        return []
    
    # Sort valid indices by similarity descending
    valid_indices_sorted = sorted(valid_indices, key=lambda i: sims_np[i], reverse=True)
    
    # Apply top_k (if specified)
    if top_k is not None:
        valid_indices_sorted = valid_indices_sorted[:top_k]
    
    retrieved = [chunks[i] for i in valid_indices_sorted]
    
    # Print retrieved details
    print("\n" + "="*80)
    print(f"RETRIEVED {len(retrieved)} CHUNKS (similarity >= {similarity_threshold}, top_k={top_k if top_k else 'all'}):")
    for rank, idx in enumerate(valid_indices_sorted, 1):
        chunk = chunks[idx]
        sim_score = sims_np[idx]
        preview = chunk.text.replace('\n', ' ')[:200] + "..."
        print(f"\n[Rank {rank}] Score: {sim_score:.4f} | Chunk: {chunk.chunk_id} (doc: {chunk.doc_id})")
        print(f"  Preview: {preview}")
    print("="*80 + "\n")
    
    return retrieved

# --------------------------------------------
# 2. Retrieve top-k chunks using BM-25
# --------------------------------------------

class WeightedBM25(BM25Okapi):
    def __init__(self, corpus: List[List[str]], doc_lexicon_sets: List[Set[str]], 
                 boost: float = 2.0, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        """
        corpus: list of tokenized documents (list of tokens per document)
        doc_lexicon_sets: list of sets, each containing terms to boost for that document
        boost: multiplier for lexicon terms (e.g., 2.0)
        """
        self.corpus = corpus
        self.doc_lexicon_sets = doc_lexicon_sets
        self.boost = boost
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.docs = corpus
        self.doc_len = [len(doc) for doc in self.docs]
        self.avgdl = np.mean(self.doc_len)

        self.idf = {}
        self.term_doc_freq = {}
        self.term_doc_pairs = {}   # term -> list of (doc_index, weighted_tf)

        for i, doc in enumerate(self.docs):
            term_counts = {}
            for term in doc:
                term_counts[term] = term_counts.get(term, 0) + 1
            lexicon_set = self.doc_lexicon_sets[i]
            for term, raw_tf in term_counts.items():
                weight = self.boost if term in lexicon_set else 1.0
                weighted_tf = raw_tf * weight
                self.term_doc_pairs.setdefault(term, []).append((i, weighted_tf))
                self.term_doc_freq[term] = self.term_doc_freq.get(term, 0) + 1

        # Compute IDF
        for term, freq in self.term_doc_freq.items():
            self.idf[term] = np.log((len(self.corpus) - freq + 0.5) / (freq + 0.5) + 1)

        self._prepare_scores()

    def _prepare_scores(self):
        self.doc_factors = []
        for doc_len in self.doc_len:
            factor = 1 / (1 - self.b + self.b * (doc_len / self.avgdl))
            self.doc_factors.append(factor)

    def get_scores(self, query: str):
        query_tokens = set(query.lower().split())
        scores = np.zeros(len(self.docs))
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            for doc_idx, weighted_tf in self.term_doc_pairs.get(token, []):
                k = self.k1 * self.doc_factors[doc_idx]
                score = idf * (weighted_tf * (self.k1 + 1)) / (weighted_tf + k)
                scores[doc_idx] += score
        return scores



def build_bm25_index(chunks: List[Any], use_lexicon_boost: bool = False, boost: float = 2.0) -> BM25Okapi:
    """
    Build a BM25 index from a list of chunk objects.
    If use_lexicon_boost is True, each chunk's corpus_id determines which lexicon to use.
    """
    tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
    
    if not use_lexicon_boost:
        return BM25Okapi(tokenized_corpus), {},{}
    
    # Collect unique corpus_ids from chunks
    corpus_ids = set()
    for chunk in chunks:
        cid = getattr(chunk, 'corpus_id', None)
        if cid is not None:
            corpus_ids.add(cid)
    
    # Load lexicon sets for each corpus
    lexicon_map = {}
    keywords_map ={}
    for cid in corpus_ids:
        lexicon_map[cid], keywords_map[cid] = load_lexicon_terms_all_sheets(cid)
    
    # Build per‑document lexicon sets (empty set for chunks without corpus_id)
    doc_lexicon_sets = []
    for chunk in chunks:
        cid = getattr(chunk, 'corpus_id', None)
        if cid is not None and cid in lexicon_map:
            doc_lexicon_sets.append(lexicon_map[cid])
        else:
            doc_lexicon_sets.append(set())   # no boost for this chunk
    
    
    return WeightedBM25(tokenized_corpus, doc_lexicon_sets, boost=boost), lexicon_map, keywords_map

def show_boosted_terms_in_chunk(chunk, lexicon_set):
    """Print which lexicon terms appear in the given chunk."""
    chunk_tokens = set(chunk.text.lower().split())
    boosted_terms = chunk_tokens.intersection(lexicon_set)
    return boosted_terms

def retrieve_chunks_bm25(
    query: str,
    chunks: List[Any],
    top_k: int = 15,
    corpus_id: Optional[str] = None
) -> List[Any]:
    """
    Retrieve top-k chunks using BM25 lexical matching.
    If corpus_id is given, only chunks belonging to that corpus are considered.
    """
    # Filter by corpus_id if provided
    if corpus_id is not None:
        filtered_chunks = [c for c in chunks if getattr(c, 'corpus_id', None) == corpus_id]
        if not filtered_chunks:
            print(f"Warning: No chunks found for corpus_id='{corpus_id}'")
            return []
    else:
        filtered_chunks = chunks

    # Build BM25 index
    bm25, lexicon_sets, keywords_map = build_bm25_index(filtered_chunks, use_lexicon_boost=True)
    #tokenized_query = query.lower().split()
    scores = bm25.get_scores(query)

    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    retrieved = [filtered_chunks[i] for i in top_indices]

    # Pretty print
    print("\n" + "="*80)
    print(f"BM25 RETRIEVED TOP {len(retrieved)} CHUNKS (corpus_id={corpus_id}):")
    for rank, chunk in enumerate(retrieved, 1):
        chunk.lexicon = show_boosted_terms_in_chunk(chunk, lexicon_sets[chunk.corpus_id])
        chunk.keywords = keywords_map[chunk.corpus_id]
        preview = chunk.text.replace('\n', ' ')[:500] + "..."
        print(f"\n[Rank {rank}] Chunk: {chunk.chunk_id} (doc: {chunk.doc_id})")
        print(f"  Preview: {preview}")
        print(f"  Boosted Lexicons: {chunk.lexicon}")
    print("="*80 + "\n")

    return retrieved


# --------------------------------------------
# 2. Retrieve top-k chunks using Hybrid with default(Equal weight) alpha = 0.5
# --------------------------------------------

def retrieve_chunks_with_routing(
    query: str,
    chunks: List[Chunk],
    top_k: int = 15,
    alpha: float = 0.5,
    corpus_embeddings: object = None,
    last_query: str = "",
    corpus_ids: List[str] = None,
    config: RAGConfig = None,
    **kwargs
) -> List[Chunk]:
    """
    First, route query to a corpus using lexicons (if corpus_ids provided).
    Then retrieve chunks only from that corpus.
    """
    '''
    corpus_ids = list({c.corpus_id for c in chunks if c.corpus_id is not None})
    print(f"Corpus Ids are: {corpus_ids}")
    best_corpus, scores = lexicon_based_router(query, corpus_ids, config)
    if best_corpus:
        print(f"Routing to corpus: {best_corpus}")
        corpus_id = best_corpus
    else:
        print("No clear routing, will retrieve from all corpora.")
        corpus_id = None
    '''
    threshold = 0.5
    best_corpus, scores = route_query_with_embeddings(query, corpus_embeddings, threshold=threshold)
    if scores[best_corpus] < threshold:
        query = last_query + query
        print("Rerouting Again...")
        best_corpus, scores = route_query_with_embeddings(query, corpus_embeddings, threshold=threshold) 
    
    return retrieve_chunks_hybrid(
        query=query,
        chunks=chunks,
        top_k=top_k,
        alpha=alpha,
        corpus_id=best_corpus,
        **kwargs
    )


def retrieve_chunks_hybrid(
    query: str,
    chunks: List[Any],
    top_k: int = 15,
    alpha: float = 0.5,                     # weight for BM25 (1-alpha for embedding)
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    cache_dir: str = ".embedding_cache",
    corpus_id: Optional[str] = None
) -> List[Any]:
    """
    Retrieve top-k chunks using a hybrid score:
        hybrid_score = alpha * norm_bm25 + (1-alpha) * cosine_similarity
    If corpus_id is given, only chunks belonging to that corpus are considered.
    """
    # Filter by corpus_id
    if corpus_id is not None:
        filtered_chunks = [c for c in chunks if getattr(c, 'corpus_id', None) == corpus_id]
        if not filtered_chunks:
            print(f"Warning: No chunks found for corpus_id='{corpus_id}'")
            return []
    else:
        filtered_chunks = chunks

    # 1. BM25 scores
    bm25, lexicon_sets, keywords_map = build_bm25_index(filtered_chunks, use_lexicon_boost=True)
   
    print(f"Keywords: {keywords_map}")
    #tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(query)
    #print("Type of lexicon list/set", type(lexicon_sets), lexicon_sets)
    # Normalize BM25 scores to [0,1] (min-max)
    if bm25_scores.max() > bm25_scores.min():
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
    else:
        bm25_norm = np.zeros_like(bm25_scores)

    # 2. Dense embedding similarities (cached)
    chunk_embeddings = get_chunk_embeddings_cached(filtered_chunks, model_name, cache_dir)
    model = get_embedding_model(model_name)
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_sims = util.cos_sim(query_embedding, chunk_embeddings)[0].cpu().numpy()
    # Cosine similarity is already in [-1,1], but for positive embeddings it's [0,1]
    # Clip to [0,1] just in case
    cos_sims = np.clip(cos_sims, 0.0, 1.0)

    # 3. Hybrid score
    hybrid_scores = alpha * bm25_norm + (1 - alpha) * cos_sims

    # 4. Get top_k
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    retrieved = [filtered_chunks[i] for i in top_indices]

    # Pretty print
    print("\n" + "="*80)
    print(f"HYBRID RETRIEVED TOP {len(retrieved)} CHUNKS (alpha={alpha}, corpus_id={corpus_id}):")
    for rank, chunk in enumerate(retrieved, 1):
        chunk.lexicon = show_boosted_terms_in_chunk(chunk, lexicon_sets[chunk.corpus_id])
        chunk.keywords = keywords_map[chunk.corpus_id]
        #preview = chunk.text.replace('\n', ' ')[:400] + "..."
        preview = chunk.text
        print(f"\n[Rank {rank}] Chunk: {chunk.chunk_id} (doc: {chunk.doc_id})")
        print(f"  Preview: {preview}")
        print(f"  Tokens: {len(nlp(preview))}")
        
        #print(f"  Lexicons Matched: {chunk.lexicon}")
    print("="*80 + "\n")

    return retrieved

def rerank(query, chunks):
    ce_model = CrossEncoder(config.ce_model, device=DEVICE)
    pairs = [[query, chunk.text] for chunk in chunks]
    scores = ce_model.predict(pairs)
    # Reorder retrieved by ce scores
    reranked = [chunk for score, chunk in sorted(zip(scores, chunks), reverse=True)]
    return reranked

# --------------------------------------------
# 2.1. Apply Segnementation techniques
# --------------------------------------------
def max_corpus_id_occurrence(chunks: List[Any]) -> Optional[Any]:
   
    # Count occurrences of each corpus_id
    counter = Counter(chunk.corpus_id for chunk in chunks)

    return counter.most_common(1)[0][0]

def split_sentences(text: str):
    doc = nlp(text)
    return [
        sent.text.strip()
        for sent in doc.sents
        if len(sent.text.strip()) > 10
    ]

def score_sentence(
    sentence: str,
    query_tokens,
    lexicon_set,
    *,
    # --- Weights (tunable) ---
    query_weight: float = 3.0,       # primary signal — query overlap
    lexicon_weight: float = 0.5,     # secondary signal — domain/lexicon bonus
    # --- BM25-style length normalisation ---
    use_length_norm: bool = True,
    avg_sent_len: float = 22.0,      # calibrate to your corpus
    b: float = 0.75,                 # BM25 length penalty strength (0 = off, 1 = full)
    # --- Optional penalties ---
    min_query_match: int = 0,        # set to 1 to hard-filter zero-query-match sentences
    # --- Optional exact-phrase bonus ---
    raw_sentence: Optional[str] = None,  # pass original case string for phrase detection
    query_phrase: Optional[str] = None,  # exact phrase to match (e.g. "organic waste")
    phrase_bonus: float = 2.0,
) -> float:
    """
    Score a sentence for RAG retrieval relevance.

    Design principles (grounded in BM25 + RAG literature):
      - Query overlap is the PRIMARY signal (high weight).
      - Lexicon/domain terms are a SECONDARY soft boost (low weight).
      - Length normalisation prevents long sentences from dominating
        purely by token count (BM25-b style).
      - Exact phrase match on the query gives an additional bonus.
      - Sentences with zero query match can be filtered out entirely.

    Returns:
        float — higher is more relevant. Returns 0.0 if below min_query_match.
    """
    sent_tokens = set(sentence.lower().split())
    sent_len    = max(len(sentence.split()), 1)

    # ── 1. Query overlap (primary, high weight) ───────────────────────────────
    query_match = len(sent_tokens.intersection(query_tokens))

    # Hard filter: skip sentences with no query signal at all
    if query_match < min_query_match:
        return 0.0

    # ── 2. BM25-style length normalisation ───────────────────────────────────
    #   Penalises very long sentences proportionally.
    #   Formula: score / (1 - b + b * (sent_len / avg_sent_len))
    #   At b=0.75: a sentence twice the average length gets ~0.67× the weight.
    if use_length_norm:
        length_norm = 1 - b + b * (sent_len / avg_sent_len)
    else:
        length_norm = 1.0

    # Normalised query score
    query_score = (query_match * query_weight) / length_norm

    # ── 3. Lexicon bonus (secondary, low weight) ──────────────────────────────
    #   Multi-word term support: check phrase membership, not just tokens.
    sentence_lower = sentence.lower()
    lexicon_match = sum(
        1 for term in lexicon_set
        if (
            term in sent_tokens          # single-word term
            if ' ' not in term
            else term in sentence_lower  # multi-word term (e.g. "sewage sludge")
        )
    )
    lexicon_score = lexicon_match * lexicon_weight  # kept deliberately low

    # ── 4. Exact query phrase bonus (optional) ────────────────────────────────
    phrase_score = 0.0
    if raw_sentence and query_phrase:
        if query_phrase.lower() in raw_sentence.lower():
            phrase_score = phrase_bonus

    return query_score + lexicon_score + phrase_score

def select_important_sentences(chunks, query):
    query_tokens = set(query.lower().split())
    
    #corpus_id = max_corpus_id_occurrence(chunks)
    #lexicon_set,_ = load_lexicon_terms_all_sheets(corpus_id)
    
    for chunk in chunks:
        sents = split_sentences(chunk.text)
        all_sentences = []
        lexicon_set,_ = load_lexicon_terms_all_sheets(chunk.corpus_id)
    
        for sent in sents:
            
            score = score_sentence(sent, query_tokens, lexicon_set, query_phrase=query)
            if score > 5:   # ignore sentences with no query/lexicon match
                all_sentences.append((score, sent))
    
        all_sentences.sort(key=lambda x: x[0], reverse=True)
        segments = [sent for score, sent in all_sentences]
        seg_str = " ".join(segments)
        chunk.segments = seg_str
    
    return chunks

# --------------------------------------------
# 3. Build prompt for context summary (fully dynamic)
# --------------------------------------------
def build_context_summary_prompt(retrieved_chunks, query):
    context_text = "\n\n".join([f"[Chunk {chunk.chunk_id}] {chunk.text[:500]}" 
                                for i, chunk in enumerate(retrieved_chunks)])
    prompt = f"""You are an agricultural research analyst. Based **only** on the research abstracts/text below, generate a concise **Context Window Summary** that answers the user's query.

User query: {query}

Your summary must include the following sections:
1. **Geographic coverage** – identify all countries and regions explicitly mentioned in the research abstracts/text (no external knowledge).
2. **Core theme** – what is the main focus of these studies?
3. **Relevant Summary of the Context** – extract key findings about efficiency (biogas yields, fertilizer replacement, soil improvements, etc.), and note any comparisons between Global South and global benchmarks if present.

Do **not** add outside knowledge. Use only the information in the research abstracts/text.

Abstracts/text:
{context_text}
"""
    return prompt



def build_generic_context_summary_prompt(
    retrieved_chunks,      # list of chunk objects (each has .text, .chunk_id, optional .boosted_lexicons)
    query: str,
    lexicon_sheet_names: list = None,
    top_n_boosted: int = 20
) -> str:
    """
    Build a prompt that is fully generic – no hardcoded terms.
    Uses query, retrieved text, and optional hints from boosted lexicons/sheet names.
    """
    # 1. Build context text (limit length if needed, but keep full context)
    context_text = "\n\n".join([
        f"[Chunk {chunk.chunk_id}] {chunk.segments}" 
        for chunk in retrieved_chunks
    ])
    keywords_set = set()
    all_boosted = set()
    for chunk in retrieved_chunks:
        keywords_set.update(chunk.keywords)   # add all keywords from this chunk (duplicates ignored)
        all_boosted.update(chunk.lexicon)
    keywords = list(keywords_set) 
    lexicons = list(all_boosted) 
    # 2. Extract all boosted lexicon terms from chunks (if available)
    boosted_hint = ""
    if lexicons:
        top_boosted = sorted(lexicons)[:top_n_boosted]  # alphabetical or by freq
        boosted_hint = f"\nFrequently occurring domain terms in the retrieved content: {', '.join(top_boosted)}\n"

    # 3. Add sheet names hint if provided
    
    sheet_hint = f"\nImportant domains keywords (if relevant to the query): {', '.join(keywords)}\n"

    # 4. Build the prompt – completely generic sections
    prompt = f"""You are an agricultural research analyst. Based **only** on the research abstracts/text below, generate a concise **Context Window Summary** that answers the user's query.

User query: {query}
{boosted_hint}{sheet_hint}
Your summary must include the following sections:

1. **Geographic coverage** – identify all countries, regions, or locations explicitly mentioned. If none, state "not specified".

2. **Core theme** – what is the main focus of the provided texts? Describe it in your own words, using any domain terms that appear.

3. **Relevant Summary of the Context** – extract key findings that directly address the user's query. Focus on any measures of **efficiency, performance, outcomes, or comparisons** that are reported.  
   - If the query asks about efficiency (e.g., cost, energy, time, environmental impact), report the specific numbers or qualitative assessments given.  
   - If the query asks about a comparison (e.g., between regions or technologies), extract that comparison if present.  
   - Otherwise, simply describe the main results.

Do **not** add outside knowledge. Use only the information in the abstracts/text.

Abstracts:
{context_text}
"""
    return prompt

# --------------------------------------------
# 4. Build persona-based answer prompts
# --------------------------------------------
'''
def build_generic_persona_prompt(persona: str, retrieved_chunks: list, query: str, boosted_keywords: list = None) -> str:
    context_text = "\n\n".join([
        f"[Chunk {chunk.chunk_id}] {chunk.segments}" 
        for chunk in retrieved_chunks
    ])
    
    # Dynamic hints from boosted keywords (if provided)
    keywords_set = set()
    all_boosted = set()
    for chunk in retrieved_chunks:
        keywords_set.update(chunk.keywords)   # add all keywords from this chunk (duplicates ignored)
        all_boosted.update(chunk.lexicon)
    keywords = list(keywords_set) 
    lexicons = list(all_boosted) 
    # 2. Extract all boosted lexicon terms from chunks (if available)
    
    if lexicons:
        top_boosted = sorted(lexicons)[:10]  # alphabetical or by freq
        boosted_hint = f"\nFrequently occurring domain terms in the retrieved content: {', '.join(top_boosted)}\n"

    # 3. Add sheet names hint if provided
    
    keyword_hint = f"\nImportant domains keywords (if relevant to the query): {', '.join(keywords)}\n"
    
    # Persona-specific generic instructions (no hardcoded metrics)
    instructions = {
        "agronomist": (
            "Focus on practical on‑farm implications: soil health, crop yield, nutrient cycling, and suitability for "
            "smallholder or resource‑limited systems. Use concrete examples from the text."
        ),
        "researcher": (
            "Focus on methodological comparisons, quantitative metrics (if reported), synergistic effects, and how findings "
            "compare to established knowledge. Cite study IDs or chunk references when available."
        ),
        "farmer": (
            "Use plain, simple language. Explain what works, what benefits to expect, and any cautions. Keep it very practical."
        )
    }
    
    prompt = f"""You are an expert advisor answering the same question from the perspective of a {persona}.

User query: {query}

{keyword_hint}{instructions[persona]}

Use **only** the information from the following research abstracts/texts. Do not add outside knowledge.

Abstracts/Texts:
{context_text}

Write a concise answer with conclusive remarks in the last sentence (5‑8 sentences).
"""
    return prompt
'''

def build_generic_persona_prompt(persona: str, retrieved_chunks: list, query: str) -> str:
    # Build context: prefer top 4 segments per chunk
    context_parts = []
    for chunk in retrieved_chunks:
        if len(chunk.segments) < len(chunk.text) and len(chunk.segments) > 0:
            text = chunk.segments
        else:
            text = chunk.text
        context_parts.append(f"[Chunk {chunk.chunk_id}] {text}")
    context_text = "\n\n".join(context_parts)
    
    base_instruction = f"""You are a {persona}. Answer the user query using **only** the provided texts. Do not add outside knowledge.

**Output format:** Write your answer as a **single coherent paragraph** (no bullet points, no numbered lists, no markdown). Use full sentences.

**Step 1 – Identify the query type:**
- YES/NO question (e.g., "Is X efficient?")
- COMPARISON question (contains "compared to", "versus", "more/less than", etc.)
- NUMERIC question (e.g., "how much", "what yield")
- EXPLANATION / SUMMARY question (e.g., "how does X work")
- **AMBIGUOUS** – the query lacks a clear subject (e.g., "What are the obstacles associated with a lack of efficiency?" – efficiency of what?).

**Step 2 – Answer according to the following rules:**

**If the query is AMBIGUOUS (no clear subject):**
- First, infer the most likely domain by looking at the **majority of retrieved chunks**. What topic appears most frequently?
- Start your paragraph by stating the domain you identified: "In the context of [inferred domain], the obstacles/limitations are..."
- Then answer the query specifically for that domain, using evidence from the chunks.
- Do **not** list obstacles from unrelated domains (e.g., if most chunks are about organic waste recycling, ignore a single chunk about biobutanol unless it clearly relates).
- If the chunks cover multiple unrelated topics equally, state: "The texts cover several topics, but the most relevant to the query is [X]."

**If the query is YES/NO (even with a comparison):**
- Your paragraph MUST start with "Yes", "No", or "Insufficient data", followed by a brief restatement.
- Then present evidence (cite chunk IDs).
- End with a one‑sentence conclusion restating your answer.

**If the query asks for a COMPARISON between two things (A vs B):**
- Paragraph structure: evidence for A, evidence for B, direct comparison, conclusion.
- If the query is also YES/NO, start with the verdict.
- Never use speculative language like "might", "seems", "appears", "could be", "possibly".

**If the query is NUMERIC:**
- Extract numbers with units. If multiple, give range or most representative. If none, say "No numerical data found."

**If the query is EXPLANATION/SUMMARY:**
- Write a concise paragraph summarizing the key findings.

**Step 3 – Always cite chunk IDs** (e.g., "according to chunk dataverse_xxx").

**Step 4 – Be concise:** 5–8 sentences total, unless more detail is essential.

**Step 5 – End with a one‑sentence conclusion** that directly answers the original query.

User query: {query}
Abstracts/Texts:
{context_text}
"""
    
    # Farmer adaptation: simpler language
    if persona == "farmer":
        base_instruction = base_instruction.replace("single coherent paragraph", "a few simple sentences")
        base_instruction = base_instruction.replace("Cite chunk IDs", "Mention which study")
    
    return base_instruction

def build_persona_prompt(persona, retrieved_chunks, query):
    context_text = "\n\n".join([f"[Chunk {chunk.chunk_id}] {chunk.text[:500]}"
                                for i, chunk in enumerate(retrieved_chunks)])
    
    instructions = {
        "agronomist": "Focus on practical on‑farm implications: soil health, crop yield improvements, nutrient cycling efficiency (N, P, K replacement), and suitability for smallholder or resource‑limited farming systems. Use specific examples.",
        "researcher": "Focus on methodological comparisons, quantitative efficiency metrics (methane yields in mL/g VS, COD removal %, T90 values), synergistic effects, and how these findings compare to established benchmarks. Cite study IDs if available.",
        "farmer": "Use plain, simple language. Explain what works (e.g., mixing wastes, using digestate as fertilizer), what benefits to expect (more biogas, better crops, less need for chemical fertilisers), and any cautions. Keep it very practical."
    }
    
    prompt = f"""You are an expert advisor answering the same question from the perspective of a {persona}.

User query: {query}

{instructions[persona]}

Use only the information from the following research abstracts/texts:

{context_text}

Write a concise answer with conclusive remarks in last sentence (5‑8 sentences).
"""
#Write a concise answer (3‑5 sentences for Farmer, 5‑8 sentences for others).
    return prompt

# --------------------------------------------
# 5. Call local LLM (Ollama) with debugging
# --------------------------------------------
def generate_llm_response(prompt, model='mistral:latest', debug=True, max_prompt_chars=12000, conversation_memory=None):
    """
    Generate LLM response with optional conversation memory.
    conversation_memory: a ConversationMemory instance (or any object with .get_messages() method)
    """
    if len(prompt) > max_prompt_chars:
        truncated_prompt = prompt[:max_prompt_chars//2] + "\n...[TRUNCATED]...\n" + prompt[-max_prompt_chars//2:]
        print(f"[WARNING] Prompt length {len(prompt)} exceeds {max_prompt_chars}. Truncating.")
        prompt = truncated_prompt
    
    if debug:
        print("\n" + "-"*50)
        print(f"[DEBUG] Sending prompt to model: {model}")
        print(f"[DEBUG] Prompt length: {len(prompt)} chars")
        if len(prompt) > 600:
            print(f"[DEBUG] Prompt preview (first 300 chars):\n{prompt[:300]}...\n[DEBUG] ... (last 200 chars):\n{prompt[-200:]}")
        else:
            print(f"[DEBUG] Full prompt:\n{prompt}")
        print("-"*50)
    
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if model not in result.stdout:
            print(f"[ERROR] Model '{model}' not found in Ollama. Available models:\n{result.stdout}")
            return f"Error: Model '{model}' not available. Please pull it with `ollama pull {model}`"
    except Exception as e:
        print(f"[WARNING] Could not check Ollama models: {e}")
    
    try:
        start_time = time.time()
        messages = []
        # Add conversation history if provided
        if conversation_memory is not None:
            messages.extend(conversation_memory.get_messages())
        # Add current user prompt
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(
            model=model,
            messages=messages,
            options={'temperature': 0.2}  # removed num_predict=-1 (invalid)
        )
        elapsed = time.time() - start_time
        answer = response['message']['content']
        
        if debug:
            print(f"[DEBUG] Response received in {elapsed:.2f} seconds")
            print(f"[DEBUG] Response length: {len(answer)} chars")
            if answer:
                print(f"[DEBUG] Response preview (first 400 chars):\n{answer[:400]}...")
            else:
                print("[DEBUG] Response is EMPTY!")
            print("-"*50)
        return answer if answer else "[No response from model]"
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return f"Error generating response: {str(e)}"


# --------------------------------------------
# 6. Query Reformulation
# --------------------------------------------

def reformulate_query_generic(original_query: str, use_llm: bool = True, llm_model: str = "qwen2.5:7b") -> str:
    """
    Reformulate a single query to improve retrieval recall and precision.
    Primary method: LLM‑based rephrasing (if use_llm=True).
    Fallback: rule‑based augmentation.
    Returns one enhanced query string.
    """
    # 1. LLM path (preferred)
    if use_llm:
        try:
            prompt = f"""You are a RAG query reformulator. Your task is to rewrite the user's query into a single, clear, retrieval‑friendly query that will help a search engine find relevant evidence.

Rules:
- Keep all original intent and key terms.
- If the query asks for a comparison, ask for separate evidence for each side.
- If it asks for a yes/no answer, ask for explicit supporting or contradicting evidence.
- If it asks for numbers, ask for exact values with units.
- If ambiguous, ask for key facts or definitions.
- Do not add extra examples or commentary.
- Output only the reformulated query, nothing else.

Original query: {original_query}
Reformulated query:"""
            
            response = ollama.chat(
                model=llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.2}
            )
            llm_query = response['message']['content'].strip()
            if llm_query:
                return llm_query
        except Exception as e:
            print(f"[WARN] LLM reformulation failed: {e}. Falling back to rule‑based.")
    
    # 2. Rule‑based fallback (if LLM disabled or failed)
    query_lower = original_query.lower()
    
    # Detect query type
    is_comparative = bool(re.search(r"\b(compared to|versus|vs|more than|less than|difference between|global|benchmark)\b", query_lower))
    is_yes_no = bool(re.search(r"^(is|are|does|do|can|could|will|would|should|has|have) ", query_lower)) or "?" in query_lower
    is_numeric = bool(re.search(r"\b(how many|how much|what is the (yield|rate|percentage|amount|cost|price)|what are the (yields|rates))\b", query_lower))
    
    # Build suffix
    if is_comparative:
        suffix = " Extract any direct comparison. If no direct comparison, give separate values for each side. Include metrics."
    elif is_yes_no:
        suffix = " Answer with 'Yes', 'No', or 'Insufficient data'. Extract supporting and contradicting evidence."
    elif is_numeric:
        suffix = " Extract all relevant numerical values with units. If multiple, give the range or most representative."
    else:
        suffix = " Provide specific evidence, numbers, comparisons, or explicit statements from the text."
    
    # Avoid double instruction
    if any(phrase in original_query for phrase in ["provide evidence", "extract numbers", "direct comparison"]):
        return original_query
    else:
        return original_query + suffix
            

if __name__ == "__main__":
    #tracker = EmissionsTracker()
    #tracker.start()
   
    
    print("Loading chunks...")
    chunks = load_chunks(config)
    print(f"Length of All chunks: {len(chunks)}")
    
    corpus_embeddings = build_corpus_embeddings(chunks, model_name=config.embed_model)
    
    # Initialize conversation memory (keeps last 3 Q&A pairs)
    memory = ConversationMemory(max_length=3)
    
    # queries = ["Is the recycling of organic waste efficient in the Global South (compared to globally)?",
    #     "What are the obstacles or limitations associated with a lack of efficiency?",
    #             ]
    
#     queries = ["What is the scientific discipline related to plant health that are used to study or address Corynespora based on this abstract?", 
# "Based on the provided corpus, identify the scientific disciplines related to plant health that are used to study or address Corynespora. Return a list of disciplines" 
#                ]
    # queries = ["Based on the provided documents, what are the main objectives of the project in terms of expected impacts and outcomes?", 
    #             "Based on the provided documents, what is the project's stage of innovation readiness (e.g., idea, hypothesis, application, etc.)" 
    #            ]
    
    queries = ["What specific research and innovation initiatives has Kenya implemented to address the issue of food security?",
"What are the partners involved in research projects dealing with food security in Kenya?",
"What are the main technical innovations proposed and what are the outcomes?",
"What types of experts are involved in the LEAP projects and what are their roles?",
"What countries deal with One Health projects?",
"What West Africa countries deal with food security?"
                ]
    last_query = ""
    for query in queries:
        
        print(f"\nQuery: {query}\n")
        
        print("Retrieving relevant chunks...")
        #retrieved = retrieve_chunks_cached(query, chunks, top_k=15)
        #retrieved = retrieve_chunks_with_routing(query, chunks, alpha = 0.5, top_k=5, corpus_embeddings = corpus_embeddings, last_query = last_query)
        
        #retrieved = retrieve_chunks_with_routing(query, chunks, alpha = 0.0, top_k=5, corpus_embeddings = corpus_embeddings, last_query = last_query)
        
        retrieved = retrieve_chunks_with_routing(query, chunks, alpha = 1.0, top_k=5, corpus_embeddings = corpus_embeddings, last_query = last_query)
        
        
        retrieved = select_important_sentences(retrieved, query)
        #retrieved = rerank(query, retrieved)  # I need to correct it (Important)
        
        print(f"Retrieved {len(retrieved)} Segments.")
        for i in range(len(retrieved)):
            
            print(f"Segment of Chunk {retrieved[i].chunk_id}: {retrieved[i].segments}")
            print(f"Tokens: {len(nlp(retrieved[i].segments))}")
        
        # 1. Generate the context window summary (dynamic country extraction)
       
        # 2. Generate three persona summaries
        #personas = ["agronomist", "researcher", "farmer"]
        personas = ["researcher"]
        reformulated_query = reformulate_query_generic(query)
        print("New Query:", reformulated_query)
        for persona in personas:
            print(f"\n--- Generating {persona.capitalize()} Summary ---")
            persona_prompt = build_generic_persona_prompt(persona, retrieved, reformulated_query)
            answer = generate_llm_response(persona_prompt, debug=True)
            print(f"\n{persona.upper()} SUMMARY:\n{answer}")
            print("-"*80)

        last_query = query 
    
    '''
        print("Context Summary Prompt...")  
        print("\n" + "="*80)
        context_prompt = build_generic_context_summary_prompt(retrieved, query)
        
        
        print(context_prompt)
        print("Generating context summary...")
        context_summary = generate_llm_response(context_prompt, debug=True)
        print("\n" + "="*80)
        print("CONTEXT WINDOW SUMMARY")
        print("="*80)
        print(context_summary)
    '''
    #emissions = tracker.stop()
    #print(f"Emissions: {emissions} kg CO₂")