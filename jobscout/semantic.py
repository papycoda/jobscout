"""Semantic matching using sentence embeddings for job-resume similarity."""

import hashlib
import logging
import os
import re
from typing import List, Set, Optional

import numpy as np

logger = logging.getLogger(__name__)

_transformer_instance = None


class _HashingSentenceTransformer:
    """Lightweight, offline-friendly embedder using hashed bag-of-words."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.is_fallback = True

    def _token_indices(self, text: str) -> List[int]:
        indices = []
        for token in re.findall(r"\w+", text.lower()):
            # Stable hash -> index in vector space
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            indices.append(idx)
        return indices

    def encode(
        self,
        texts,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False
    ):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=float)
            for idx in self._token_indices(text or ""):
                vec[idx] += 1.0

            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

            embeddings.append(vec)

        if convert_to_numpy:
            return np.vstack(embeddings) if embeddings else np.empty((0, self.dimension))
        return embeddings


def _get_model():
    """Lazy load the sentence transformer model."""
    global _transformer_instance

    if _transformer_instance is not None:
        return _transformer_instance

    model_name = os.getenv("JOBSCOUT_SEMANTIC_MODEL", "all-MiniLM-L6-v2")
    skip_download = os.getenv("JOBSCOUT_SKIP_SEMANTIC_DOWNLOAD") == "1" or bool(
        os.getenv("HF_HUB_OFFLINE") or os.getenv("TRANSFORMERS_OFFLINE")
    )

    if skip_download:
        logger.info("Semantic model download skipped; using lightweight hashing embedder")
        _transformer_instance = _HashingSentenceTransformer()
        return _transformer_instance

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading sentence transformer model: {model_name}")
        _transformer_instance = SentenceTransformer(model_name)
        logger.info("Sentence transformer model loaded successfully")
        return _transformer_instance
    except ImportError:
        logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
    except Exception as e:
        logger.warning(f"Failed to load sentence transformer '{model_name}': {e}. Falling back to hashing embedder.")

    _transformer_instance = _HashingSentenceTransformer()
    return _transformer_instance


def preload_semantic_model() -> bool:
    """
    Ensure a semantic model is available and warmed up.

    Returns:
        True if a model (real or fallback) is ready, False otherwise.
    """
    try:
        model = _get_model()
        if model is None:
            return False

        # Light warmup to trigger any lazy downloads/caches
        _ = compute_embeddings(["semantic model warmup"])
        return True
    except Exception as e:
        logger.warning(f"Semantic model preload failed: {e}")
        return False


def compute_embeddings(texts: List[str]) -> Optional[List]:
    """
    Compute embeddings for a list of texts.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors, or None if model unavailable
    """
    if not texts:
        return []

    model = _get_model()
    if model is None:
        return None

    try:
        return model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Unit length for cosine similarity
            show_progress_bar=False
        )
    except Exception as e:
        logger.error(f"Failed to compute embeddings: {e}")
        return None


def compute_similarity(resume_embedding, job_embedding) -> float:
    """
    Compute cosine similarity between resume and job embeddings.

    Both embeddings should be normalized (unit length).

    Args:
        resume_embedding: Resume embedding vector
        job_embedding: Job embedding vector

    Returns:
        Similarity score between -1 and 1
    """
    # Dot product of normalized vectors = cosine similarity
    similarity = float(np.dot(resume_embedding, job_embedding))
    return max(-1.0, min(1.0, similarity))


def compute_semantic_match_score(
    resume_text: str,
    job_description: str,
    resume_skills: Optional[Set[str]] = None
) -> Optional[float]:
    """
    Compute semantic similarity between resume and job description.

    Args:
        resume_text: Full resume text
        job_description: Job description text
        resume_skills: Optional set of skills for weighted matching

    Returns:
        Similarity score between -1 and 1, or None if unavailable
    """
    embeddings = compute_embeddings([resume_text, job_description])
    if embeddings is None:
        return None

    resume_emb, job_emb = embeddings[0], embeddings[1]
    return compute_similarity(resume_emb, job_emb)


class SemanticScorer:
    """
    Add semantic similarity scoring to the existing exact-match scoring.

    Blends semantic similarity with exact overlap for robust matching.
    """

    def __init__(self, semantic_weight: float = 0.5):
        """
        Initialize semantic scorer.

        Args:
            semantic_weight: Weight for semantic score (0-1).
                              0.5 = 50% semantic, 50% exact matching
        """
        self.semantic_weight = semantic_weight
        model = _get_model()
        self._enabled = model is not None
        self._using_fallback = getattr(model, "is_fallback", False) if model else False

        if not self._enabled:
            logger.warning("Semantic scoring disabled (sentence-transformers not available)")
        elif self._using_fallback:
            logger.info("Semantic scoring using lightweight hashing embedder (offline mode)")

    def is_enabled(self) -> bool:
        """Check if semantic scoring is available."""
        return self._enabled

    def compute_resume_embedding(self, resume_text: str) -> Optional[List]:
        """Compute and cache resume embedding."""
        if not self._enabled:
            return None
        return compute_embeddings([resume_text])

    def compute_semantic_scores(
        self,
        resume_text: str,
        job_descriptions: List[str]
    ) -> List[Optional[float]]:
        """
        Compute semantic similarity scores for multiple jobs.

        Args:
            resume_text: Resume text
            job_descriptions: List of job descriptions

        Returns:
            List of similarity scores (0-1), None for failed computations
        """
        if not self._enabled:
            return [None] * len(job_descriptions)

        # Batch compute all embeddings at once
        all_texts = [resume_text] + job_descriptions
        embeddings = compute_embeddings(all_texts)

        if embeddings is None:
            return [None] * len(job_descriptions)

        resume_emb = embeddings[0]
        scores = []

        for i in range(len(job_descriptions)):
            job_emb = embeddings[i + 1]
            similarity = compute_similarity(resume_emb, job_emb)
            scores.append(similarity)

        return scores

    def blend_scores(
        self,
        exact_score: float,
        semantic_score: Optional[float]
    ) -> float:
        """
        Blend exact matching score with semantic similarity.

        Args:
            exact_score: Original exact-match score (0-100)
            semantic_score: Semantic similarity score (0-1)

        Returns:
            Blended score (0-100)
        """
        if semantic_score is None:
            return exact_score

        if semantic_score < 0:
            return exact_score

        # Convert semantic score (0-1) to percentage
        semantic_pct = min(1.0, semantic_score) * 100

        # If semantic score is very low (< 0.1), it's likely poor embedding quality
        # (e.g., hashing embedder on generic text) rather than actual mismatch.
        # In this case, use exact score to avoid false negatives.
        if self._using_fallback and semantic_pct < 10:
            return exact_score

        # Weighted blend
        blended = (
            (1 - self.semantic_weight) * exact_score +
            self.semantic_weight * semantic_pct
        )

        return min(100.0, max(0.0, blended))


# Convenience function for quick semantic matching
def quick_match(resume_text: str, job_description: str) -> Optional[float]:
    """
    Quick semantic match between resume and job.

    Returns:
        Similarity score 0-1, or None if unavailable
    """
    return compute_semantic_match_score(resume_text, job_description)
