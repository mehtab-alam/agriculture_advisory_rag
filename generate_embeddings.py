#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 00:08:18 2026

@author: syed


Reads the unified metadata from data/processed/unified_metadata.parquet,
generates sentence embeddings using a specified model, and saves the
embeddings array and a copy of the metadata (with an added embedding column?)
to data/embeddings/.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def main(model_name: str):
    # Define paths relative to project root
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    embeddings_dir = project_root / "data" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Load unified metadata
    metadata_path = processed_dir / "unified_metadata.parquet"
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found. Run normalize_corpora.py first.")
        return

    df = pd.read_parquet(metadata_path)
    print(f"Loaded {len(df)} documents from {metadata_path}")

    # Load embedding model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Generate embeddings
    texts = df["text"].tolist()
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Save embeddings
    emb_path = embeddings_dir / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Embeddings saved to {emb_path}")

    # Also save metadata (without embeddings) to same folder for convenience
    meta_out = embeddings_dir / "metadata.parquet"
    df.to_parquet(meta_out, index=False)
    df.to_csv(embeddings_dir / "metadata.csv", index=False)
    print(f"Metadata copied to {meta_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from unified metadata.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (e.g., 'all-MiniLM-L6-v2', 'all-mpnet-base-v2').",
    )
    args = parser.parse_args()
    main(args.model_name)