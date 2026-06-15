# Hybrid RAG: Agricultural Research Retrieval & Synthesis Pipeline

A hybrid retrieval-augmented generation (RAG) system for querying agricultural research corpora. It combines dense embedding retrieval, BM25 lexical search with domain-lexicon boosting, embedding-based corpus routing, sentence-level relevance scoring, and persona-driven answer generation via a local LLM (Ollama).

## Features

- **Hybrid retrieval**: Weighted combination of BM25 (lexicon-boosted) and dense sentence-embedding similarity (`alpha`-tunable).
- **Corpus routing**: Automatically routes a query to the most relevant corpus using corpus-level embedding similarity, with a query-rewriting fallback when confidence is low.
- **Lexicon-aware boosting**: Domain-specific terminology (loaded from Excel lexicon sheets) boosts BM25 scores and informs prompt context.
- **Sentence-level segmentation**: Re-ranks and extracts the most query-relevant sentences within each retrieved chunk.
- **Cross-encoder reranking**: Optional reranking of retrieved chunks using a cross-encoder model.
- **Persona-based answer generation**: Generates answers tailored to different personas (researcher, agronomist, farmer) using a local Ollama LLM.
- **Query reformulation**: LLM-based or rule-based query rewriting to improve retrieval recall/precision.
- **Conversation memory**: Maintains short-term Q&A history for multi-turn context.
- **Embedding cache**: Disk-cached chunk embeddings keyed by content hash to avoid recomputation.
- **Unit normalization**: Cleans corrupted/inconsistent scientific unit notation (e.g., `kg/ha` → `kg ha⁻¹`, `CO2` → `CO₂`).

## Requirements

- Python 3.10–3.11 recommended (required for `spacy<3.8.0` compatibility)
- [Ollama](https://ollama.com) installed and running locally, with at least one chat model pulled (e.g., `mistral`, `qwen2.5:7b`)
- Optional: CUDA-capable GPU for faster embedding/cross-encoder/NER inference

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The requirements file pulls some packages directly from GitHub and external wheel URLs (e.g., `Biotex4Py`, `FastR`, the spaCy English model). Ensure you have internet access and `git` installed before running this step.

### 4. Download required NLTK and spaCy data

The script auto-downloads the NLTK `punkt` tokenizer on first run. The spaCy English model (`en_core_web_sm`) is installed via the wheel URL in `requirements.txt`. If it fails to install automatically, run:

```bash
python -m spacy download en_core_web_sm
```

### 5. Install and configure Ollama

```bash
# Install Ollama (see https://ollama.com for platform-specific instructions)
ollama pull mistral
ollama pull qwen2.5:7b   # used for query reformulation
```

Ensure the Ollama service is running (`ollama serve`) before executing the pipeline.

## Configuration

All runtime settings are controlled via the `RAGConfig` dataclass at the top of `hybrid_rag.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `corpus_dir` | `data/processed/corpus` | Directory containing source corpus metadata CSVs |
| `lexicon_dir` | `data/lexicon` | Directory containing per-corpus domain lexicon Excel files (`{corpus_id}.xlsx`) |
| `ner_dir` | `data/ner/gliner/` | Directory for GLiNER named-entity output |
| `rag_dir` / `chunk_cache_dir` | `data/retrieval_results/RAG` | Directory for chunk pickle files (`*.pkl`) and retrieval outputs |
| `chunk_size` / `chunk_overlap` | `600` / `150` | Chunking parameters (tokens) |
| `embed_model` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence embedding model for dense retrieval and routing |
| `ce_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for reranking |
| `gliner_model` | `urchade/gliner_multi` | GLiNER model for named-entity recognition |
| `llm_model_name` | `phi-3.1-mini-4k-instruct` | Default LLM identifier (legacy; Ollama models are set per-call) |
| `default_top_k` | `10` | Default number of chunks retrieved |
| `weights` | `dense: 0.3, bm25: 0.2, entity: 0.3, lexicon: 0.2` | Component weights for hybrid scoring (reference values) |
| `chunks_files` | list of `.pkl` filenames | Named chunk caches expected in `chunk_cache_dir` |
| `default_seg_method` | `cross_encoder` | Default segmentation/reranking strategy |
| `max_llm_tokens` | `800` | Maximum tokens for LLM responses |
| `confidence_threshold` | `0.5` | Minimum similarity for corpus routing confidence |

### Directory layout expected by the pipeline

```
data/
├── processed/corpus/        # {corpus_id}.csv files with 'id' and 'metadata' columns
├── lexicon/                  # {corpus_id}.xlsx domain lexicon files
├── ner/gliner/                # NER outputs
└── retrieval_results/RAG/    # chunk_*.pkl files and embedding cache
```

Each lexicon Excel file should contain one or more sheets, each with a `term` column listing domain-specific terminology used for BM25 boosting and prompt context.

## Usage

### Basic run

```bash
python hybrid_rag.py
```

By default, the script will:
1. Load pre-chunked documents from pickle files in `chunk_cache_dir`.
2. Build corpus-level embeddings for routing.
3. Iterate over a hardcoded list of example queries.
4. Retrieve, segment, and rerank chunks for each query.
5. Reformulate the query and generate a persona-based (`researcher`) summary via Ollama.

### Customizing queries

Edit the `queries` list in the `if __name__ == "__main__":` block, or import and call the pipeline functions programmatically:

```python
from hybrid_rag import (
    config, load_chunks, build_corpus_embeddings,
    retrieve_chunks_with_routing, select_important_sentences,
    build_generic_persona_prompt, generate_llm_response,
    reformulate_query_generic
)

chunks = load_chunks(config)
corpus_embeddings = build_corpus_embeddings(chunks, model_name=config.embed_model)

query = "What are the main biogas yield improvements reported?"
retrieved = retrieve_chunks_with_routing(
    query, chunks, alpha=0.5, top_k=5, corpus_embeddings=corpus_embeddings
)
retrieved = select_important_sentences(retrieved, query)

reformulated = reformulate_query_generic(query)
prompt = build_generic_persona_prompt("researcher", retrieved, reformulated)
answer = generate_llm_response(prompt)
print(answer)
```

### Retrieval modes

- `retrieve_chunks_cached` — pure dense embedding retrieval.
- `retrieve_chunks_bm25` — pure BM25 lexical retrieval with lexicon boosting.
- `retrieve_chunks_hybrid` — weighted hybrid of BM25 + dense (`alpha` controls BM25 weight).
- `retrieve_chunks_with_routing` — routes to the best-matching corpus via embeddings, then performs hybrid retrieval within it.

### Personas

`build_generic_persona_prompt(persona, retrieved_chunks, query)` supports:
- `"researcher"` — methodological/quantitative focus
- `"agronomist"` — practical on-farm implications
- `"farmer"` — plain-language explanation

## Notes on dependencies

- Several entries in `requirements.txt` are commented out or reference alternative sources (e.g., `pke`, `FastR` fallback via GitHub, French spaCy model, `codecarbon` emissions tracking). Review and uncomment as needed for your environment.
- `spacy<3.8.0` is pinned for compatibility with `Biotex4Py`/`dframcy`.
- GPU users should install `faiss-gpu` instead of `faiss-cpu`, and ensure the correct CUDA-enabled `torch` build is installed.
- `tensorflow` and `Ollama` (capitalized) entries near the end of `requirements.txt` appear environment-specific (server setup) — verify they're needed for your platform before installing.


