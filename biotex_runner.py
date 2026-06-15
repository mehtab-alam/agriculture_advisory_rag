#!/usr/bin/env python3
"""
BioTex automatic term extraction.

This is a placeholder using TF‑IDF keyword extraction.
Replace with a real BioTex integration when available.
"""

from typing import List, Tuple, Optional
import pandas as pd
import spacy
import re
from biotex import Biotex

class BioTexRunner:
    def __init__(self, lang: str = "en", method: str = "f_tfidf_c"):
        """
        Parameters:
        lang : Language code for spaCy and Biotex ('en', 'fr', ...).
        method : Scoring method ('tf_idf', 'f_tfidf_c', 'c_value', ...).
        """
        self.lang = lang
        self.method = method

        # Load spaCy model
        try:
            self.nlp = spacy.load(f"{lang}_core_web_sm")
        except OSError:
            raise OSError(f"spaCy model '{lang}' not found. "
                          f"Install it with: python -m spacy download {lang}_core_web_sm")

        # Initialize Biotex
        self.biotex = Biotex(lang)

        # Determine validation column name
        if lang == "en":
            self.validation_col = "in_UMLS"
        elif lang == "fr":
            self.validation_col = "in_MeSH"
        else:
            self.validation_col = None

        # Store stop words for later use
        self.stop_words = set(self.nlp.Defaults.stop_words)

    def _is_stopword_term(self, term: str) -> bool:
        """Check if a term consists only of stop words."""
        tokens = term.split()
        if not tokens:
            return True
        return all(token.lower() in self.stop_words for token in tokens)

    def _has_good_pos(self, term: str, allowed_pos: set = {"NOUN", "ADJ", "VERB"}) -> bool:
        """Check if a term contains at least one word with an allowed POS tag."""
        doc = self.nlp(term)
        for token in doc:
            if token.pos_ in allowed_pos:
                return True
        return False

    def _filter_for_ir(self, terms: List[str]) -> List[str]:
        """
        Apply IR‑oriented filtering:
        - Remove terms that are only stop words.
        - Remove very short terms (single letters, etc.).
        - Remove terms consisting solely of numbers/punctuation.
        - Remove URLs.
        - Optionally keep only terms that have at least one content word (NOUN/ADJ/VERB).
        """
        filtered = []
        for term in terms:
            # 1. Exclude stop‑word terms
            if self._is_stopword_term(term):
                continue
            # 2. Exclude very short terms (single letters)
            if len(term) < 3:
                continue
            # 3. Exclude terms that are only numbers/punctuation
            if re.fullmatch(r'[\d\W]+', term):
                continue
            # 4. Exclude URLs
            if re.search(r'https?://', term):
                continue
            # 5. Optional: keep only if term has a content word (NOUN/ADJ/VERB)
            if not self._has_good_pos(term):
                continue
            filtered.append(term)
        return filtered

    def extract_terms(self, texts: List[str], filter_for_ir: bool = True) -> Tuple[List[str], List[float]]:
        """
        Extract terms using BioTex, keep only those validated in UMLS/MeSH,
        and optionally apply IR‑friendly filtering.

        Args:
            texts: List of document strings.
            filter_for_ir: If True, apply additional cleaning (stop words, POS, etc.).

        Returns:
            Tuple of (terms, frequencies) in descending order of score.
        """
        if not texts:
            return [], []

        # Step 1: Run BioTex extraction
        results_df = self.biotex.extract_term_corpus(texts, self.method)
        
        
        return results_df
        # Step 2: Filter by validation column (UMLS/MeSH)
        if self.validation_col and self.validation_col in results_df.columns:
            results_df = results_df[results_df[self.validation_col] == True]
        print(results_df)
        
        
        if results_df.empty:
            return [], []

        # Extract terms and frequencies
        all_terms = results_df.index.tolist()
        all_freqs = results_df["freq"].tolist()

        # Step 3: Apply IR filtering (if requested)
        if filter_for_ir:
            keep_mask = []
            for term in all_terms:
                if term in self._filter_for_ir([term]):   # simple check
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)
            # Filter both lists accordingly
            filtered_terms = [t for t, keep in zip(all_terms, keep_mask) if keep]
            filtered_freqs = [f for f, keep in zip(all_freqs, keep_mask) if keep]
            return filtered_terms, filtered_freqs
        else:
            return all_terms, all_freqs