#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 16:51:07 2026

@author: syed
"""

#!/usr/bin/env python3
"""

Reads all six corpora from data/raw/, normalizes them into a common format,
and saves a unified metadata file (parquet + csv) in data/processed/.
"""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import html

def clean(html_text,  unescape=True, collapse_whitespace=True):
    soup = BeautifulSoup(html_text, 'html.parser')
    text = soup.get_text()
    
    # Optionally unescape HTML entities
    if unescape:
        text = html.unescape(text)
    
    # Collapse all whitespace (newlines, tabs, multiple spaces) into a single space
    if collapse_whitespace:
        text = re.sub(r'\s+', ' ', text)
    
    # Trim leading/trailing spaces
    return text.strip()


# def remove_empty(df, cols=['title', 'text']):
#     """
#     Remove rows where any of the specified columns contain the substring 'nan' (case-insensitive).
#     """
#     # Create a mask: True for rows where any of the given columns contains 'nan'
#     mask = df[cols].apply(lambda row: row.astype(str).str.contains('nan', case=False).any(), axis=1)
#     df_clean = df[~mask]
#     return df_clean



def remove_empty(df, cols=['title', 'text'], empty_vals=None):
    if empty_vals is None:
        empty_vals = ['', 'nan', 'na', 'n/a', 'null', 'none', '-']
    
    def is_empty(val):
        if pd.isna(val):
            return True
        if isinstance(val, str):
            val_clean = val.strip()
            if val_clean == '':
                return True
            # Split into tokens
            tokens = val_clean.split()
            if len(tokens) == 0:
                return True
            # Check if all tokens (lowercased) are in empty_vals
            empty_set = set(empty_vals)
            if all(tok.lower() in empty_set for tok in tokens):
                return True
        return False

    # Create boolean masks for each column indicating emptiness
    masks = [df[col].apply(is_empty) for col in cols]
    # Combine: rows where ALL specified columns are empty
    both_empty_mask = pd.concat(masks, axis=1).all(axis=1)
    # Keep rows that are NOT both empty
    df_clean = df[~both_empty_mask]
    return df_clean
    

def main():
    # Define paths relative to project root
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    records = []
    colletotrichum =[]
    corynespora =[]
    keops =[]
    organic_waste_obs =[]
    cea_obs =[]
    # ----------------------------------------------------------------------
    # 1. AgroEcology-Abstracts
    agro_path = raw_dir / "AgroEcology-Abstracts"
    if agro_path.exists():
        txt_files = list(agro_path.rglob("*.txt"))
        for txt_file in tqdm(txt_files, desc="AgroEcology-Abstracts"):
            relative = txt_file.relative_to(agro_path)
            subfolder = relative.parts[0] if len(relative.parts) > 0 else ""
            filename = txt_file.name

            text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
            text = clean(text)
            if not text:
                continue

            title = filename[:-4]  # remove .txt
            title = clean(title)
            metadata = {"subfolder": subfolder, "filename": filename}
            records.append({
                "id": f"agro_{subfolder}_{filename}",
                "source": "AgroEcology-Abstracts",
                "title": title,
                "text": text,
                "metadata": json.dumps(metadata),
            })
    df_agro = pd.DataFrame(records)
    df_agro = remove_empty(df_agro)
    print(f"\nTotal documents AgroEcology-Abstracts collected: {len(df_agro)}")
    out_csv = processed_dir / "AgroEcology-Abstracts.csv"
    df_agro.to_csv(out_csv, index=False)
    
    # ----------------------------------------------------------------------
    # 2. CLAPAS-colletotrichum and CLAPAS-corynespora
    clapas_path = raw_dir / "CLAPAS BibEx4MaFHé"   # note the accent
    if clapas_path.exists():
        # colletotrichum
        coll_csv = clapas_path / "colletotrichum_formatted_corpus.csv"
        if coll_csv.exists():
            df_coll = pd.read_csv(coll_csv, keep_default_na=False)
            for idx, row in tqdm(df_coll.iterrows(), total=len(df_coll), desc="CLAPAS-colletotrichum"):
                title = row.get("title", "")
                abstract = row.get("abstract", "")
                text = f"{title} {abstract}".strip()
                text = clean(text)
                title = clean(title)
                if not text:
                    continue
                metadata = {
                    "author_keywords": row.get("author_keywords", ""),
                    "affiliations": row.get("affiliations", ""),
                }
                ob_colletotrichum ={
                    "id": f"clapas_colletotrichum_{idx}",
                    "source": "CLAPAS-colletotrichum",
                    "title": title,
                    "text": text,
                    "metadata": json.dumps(metadata),
                }
                records.append(ob_colletotrichum)
                colletotrichum.append(ob_colletotrichum)
        df_colletotrichum = pd.DataFrame(colletotrichum)
        df_colletotrichum = remove_empty(df_colletotrichum)
        print(f"\nTotal documents colletotrichum-corpus collected: {len(df_colletotrichum)}")
        colletotrichum_csv = processed_dir / "colletotrichum_corpus.csv"
        df_colletotrichum.to_csv(colletotrichum_csv, index=False)      
        
        
        # corynespora
        cory_csv = clapas_path / "corynespora_formatted_corpus.csv"
        if cory_csv.exists():
            df_cory = pd.read_csv(cory_csv, keep_default_na=False)
            for idx, row in tqdm(df_cory.iterrows(), total=len(df_cory), desc="CLAPAS-corynespora"):
                title = row.get("title", "")
                abstract = row.get("abstract", "")
                text = f"{title} {abstract}".strip()
                text = clean(text)
                title = clean(title)
                if not text:
                    continue
                metadata = {
                    "author_keywords": row.get("author_keywords", ""),
                    "affiliations": row.get("affiliations", ""),
                }
                ob_corynespora ={
                    "id": f"clapas_corynespora_{idx}",
                    "source": "CLAPAS-corynespora",
                    "title": title,
                    "text": text,
                    "metadata": json.dumps(metadata),
                }
                records.append(ob_corynespora)
                corynespora.append(ob_corynespora)
        df_corynespora = pd.DataFrame(corynespora)
        df_corynespora = remove_empty(df_corynespora)
        print(f"\nTotal documents corynespora-corpus collected: {len(df_corynespora)}")
        corynespora_csv = processed_dir / "corynespora_corpus.csv"
        df_corynespora.to_csv(corynespora_csv, index=False)  
    
    # ----------------------------------------------------------------------
    # 3. Dataverse-Organic Waste Management
    dataverse_path = raw_dir / "Dataverse-Organic Waste Management"
    if dataverse_path.exists():
        corpus_csv = dataverse_path / "corpus.csv"
        if corpus_csv.exists():
            df_dv = pd.read_csv(corpus_csv, encoding="latin1", sep =';', keep_default_na=False)
            for idx, row in tqdm(df_dv.iterrows(), total=len(df_dv), desc="Dataverse-Organic Waste"):
                title = row.get("Title", "")
                abstract = row.get("Abstract", "")
                text = f"{title} {abstract}".strip()
                text = clean(text)
                title = clean(title)
                if not text:
                    continue
                metadata = {
                    "Country": row.get("Country", ""),
                    "Region": row.get("Region", ""),
                }
                owm_ob = {
                    "id": f"dataverse_{idx}",
                    "source": "Dataverse-Organic Waste Management",
                    "title": title,
                    "text": text,
                    "metadata": json.dumps(metadata),
                }
                records.append(owm_ob)
                organic_waste_obs.append(owm_ob)
        df_organic_waste_obs = pd.DataFrame(organic_waste_obs)
        df_organic_waste_obs = remove_empty(df_organic_waste_obs)
        print(f"\nTotal documents organic-waste_management-corpus collected: {len(df_organic_waste_obs)}")
        organic_waste_csv = processed_dir / "owm_corpus.csv"
        df_organic_waste_obs.to_csv(organic_waste_csv, index=False)  
   
    # ----------------------------------------------------------------------
    # 4. KEOPS
    keops_path = raw_dir / "KEOPS"
    if keops_path.exists():
        keops_csv = keops_path / "KEOPS_export.csv"
        if keops_csv.exists():
            df_keops = pd.read_csv(keops_csv, keep_default_na=False)
            for idx, row in tqdm(df_keops.iterrows(), total=len(df_keops), desc="KEOPS"):
                title = row.get("title", "")
                text_content = row.get("text", "")
                text = f"{title} {text_content}".strip()
                text = clean(text)
                title = clean(str(title))
                if not text:
                    continue
                metadata = {
                    "country": row.get("country", ""),
                    "continent": row.get("continent", ""),
                    "CLS_Document Type": row.get("CLS_Document Type", ""),
                }
                keops_ob = {
                    "id": f"keops_{idx}",
                    "source": "KEOPS",
                    "title": title,
                    "text": text,
                    "metadata": json.dumps(metadata),
                }
                records.append(keops_ob)
                keops.append(keops_ob)
        df_keops = pd.DataFrame(keops)
        df_keops = remove_empty(df_keops)
        print(f"\nTotal documents KEOPS-corpus collected: {len(df_keops)}")
        keops_csv = processed_dir / "KEOPS_corpus.csv"
        df_keops.to_csv(keops_csv, index=False) 
   
    # ----------------------------------------------------------------------
    # 5. CEA-first
    cea_path = raw_dir / "CEA-first"
    if cea_path.exists():
        cea_csv = cea_path / "articles-agro.csv"
        if cea_csv.exists():
            df_cea = pd.read_csv(cea_csv, keep_default_na=False)
            for idx, row in tqdm(df_cea.iterrows(), total=len(df_cea), desc="CEA-first"):
                title = row.get("title", "")
                text_content = row.get("text", "")
                text = f"{title} {text_content}".strip()
                text = clean(text)
                title = clean(str(title))
                if not text:
                    continue
                metadata = {
                    "CLS_Document Type": row.get("CLS_Document Type", ""),
                    "country": row.get("country", ""),
                    "continent": row.get("continent", ""),
                    "CLS_Relevance": row.get("CLS_Relevance", ""),
                    "text_length": row.get("text_length", ""),
                }
                cea_ob = {
                    "id": f"cea_{idx}",
                    "source": "CEA-first",
                    "title": title,
                    "text": text,
                    "metadata": json.dumps(metadata),
                }
                records.append(cea_ob)
                cea_obs.append(cea_ob)
        df_cea= pd.DataFrame(cea_obs)
        df_cea = remove_empty(df_cea)
        print(f"\nTotal documents CEA-corpus collected: {len(df_cea)}")
        cea_csv = processed_dir / "CEA_corpus.csv"
        df_cea.to_csv(cea_csv, index=False) 

    # ----------------------------------------------------------------------
   
    # Create unified DataFrame
    df = pd.DataFrame(records)
    df = remove_empty(df)
    print(f"\nTotal documents collected: {len(df)}")

    if df.empty:
        print("No documents found. Check your raw data paths.")
        return

    # Save unified metadata
    out_parquet = processed_dir / "unified_metadata.parquet"
    out_csv = processed_dir / "unified_metadata.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"Unified metadata saved to:\n  {out_parquet}\n  {out_csv}")


if __name__ == "__main__":
    main()