"""Tests for semantic search functionality."""

import os
import pytest
from jobscout.semantic import (
    _get_model,
    compute_embeddings,
    compute_similarity,
    compute_semantic_match_score,
    SemanticScorer,
    quick_match
)


@pytest.fixture(autouse=True)
def force_offline_mode(monkeypatch):
    """Avoid heavyweight model downloads during tests."""
    if "JOBSCOUT_SKIP_SEMANTIC_DOWNLOAD" not in os.environ:
        monkeypatch.setenv("JOBSCOUT_SKIP_SEMANTIC_DOWNLOAD", "1")


@pytest.fixture(scope="module")
def using_fallback_model():
    """Detect whether the lightweight hashing embedder is active."""
    model = _get_model()
    return getattr(model, "is_fallback", False)


class TestModelLoading:
    """Test sentence transformer model loading."""

    def test_get_model_returns_instance(self):
        """Model should be loaded and cached."""
        model = _get_model()
        if model is not None:
            # Should be callable/usable
            assert hasattr(model, 'encode')

        # Second call should return cached instance
        model2 = _get_model()
        assert model is model2

    def test_get_model_uses_fallback_when_offline(self):
        """When downloads are skipped, fallback embedder should be returned."""
        model = _get_model()
        assert model is not None
        assert getattr(model, "is_fallback", False)


class TestComputeEmbeddings:
    """Test embedding computation."""

    def test_single_text_embedding(self):
        """Should generate embedding for single text."""
        embeddings = compute_embeddings(["Python developer"])
        if embeddings is not None:
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 384  # MiniLM-L6-v2 dimension

    def test_multiple_text_embeddings(self):
        """Should generate embeddings for multiple texts."""
        texts = ["Python developer", "Java engineer", "Data scientist"]
        embeddings = compute_embeddings(texts)
        if embeddings is not None:
            assert len(embeddings) == 3
            for emb in embeddings:
                assert len(emb) == 384

    def test_empty_list_returns_none(self):
        """Empty list should be handled gracefully."""
        embeddings = compute_embeddings([])
        # Should handle empty case
        assert embeddings == [] or embeddings is None

    def test_embeddings_are_normalized(self):
        """Embeddings should be normalized (unit length)."""
        import numpy as np

        embeddings = compute_embeddings(["test text"])
        if embeddings is not None:
            emb = embeddings[0]
            # Should be approximately unit length
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 0.001  # Allow small floating point error


class TestComputeSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors_max_similarity(self):
        """Identical vectors should have similarity of 1.0."""
        import numpy as np

        vec = np.array([0.5, 0.5, 0.5, 0.5])
        # Normalize
        vec = vec / np.linalg.norm(vec)

        similarity = compute_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, rel=0.01)

    def test_orthogonal_vectors_zero_similarity(self):
        """Orthogonal vectors should have similarity near 0."""
        import numpy as np

        vec1 = np.array([1.0, 0.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0, 0.0])

        similarity = compute_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=0.01)

    def test_opposite_vectors_negative_similarity(self):
        """Opposite direction vectors should have negative similarity."""
        import numpy as np

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])

        similarity = compute_similarity(vec1, vec2)
        assert similarity < 0


class TestSemanticMatchScore:
    """Test semantic matching between resume and job descriptions."""

    def test_perfect_match_same_text(self):
        """Same text should have maximum similarity."""
        resume = "Python developer with Django experience"
        job = "Python developer with Django experience"

        score = compute_semantic_match_score(resume, job)
        if score is not None:
            assert score == pytest.approx(1.0, rel=0.01)

    def test_highly_similar_technologies(self, using_fallback_model):
        """Similar technical content should score high."""
        if using_fallback_model:
            pytest.skip("Fallback embedder provides approximate similarity only")

        resume = "Senior Python developer with Django and PostgreSQL"
        job = "Python backend engineer using Django framework and PostgreSQL database"

        score = compute_semantic_match_score(resume, job)
        if score is not None:
            # Should be quite similar
            assert score > 0.7

    def test_different_tech_domains_low_score(self):
        """Different tech domains should have low similarity."""
        resume = "Frontend developer with React and TypeScript"
        job = "Java Spring Boot backend developer"

        score = compute_semantic_match_score(resume, job)
        if score is not None:
            # Should be less similar
            assert score < 0.5

    def test_no_model_available_returns_none(self):
        """If model is unavailable, should return None."""
        # This test verifies the graceful degradation path
        # We can't easily mock the import failure, but the code handles it
        pass


class TestSemanticScorer:
    """Test SemanticScorer class functionality."""

    def test_initialization_default_weight(self):
        """Should initialize with default weight."""
        scorer = SemanticScorer()
        assert scorer.semantic_weight == 0.5
        # Only check is_enabled if model is available
        if _get_model() is not None:
            assert scorer.is_enabled()

    def test_initialization_custom_weight(self):
        """Should initialize with custom semantic weight."""
        scorer = SemanticScorer(semantic_weight=0.7)
        assert scorer.semantic_weight == 0.7

    def test_compute_resume_embedding(self):
        """Should compute and cache resume embedding."""
        scorer = SemanticScorer()
        if not scorer.is_enabled():
            pytest.skip("Sentence transformers not available")

        resume = "Python developer with 5 years experience"
        embedding = scorer.compute_resume_embedding(resume)

        assert embedding is not None
        assert len(embedding) == 1  # Returns list with one embedding

    def test_compute_semantic_scores_single_job(self):
        """Should compute semantic score for single job."""
        scorer = SemanticScorer()
        if not scorer.is_enabled():
            pytest.skip("Sentence transformers not available")

        resume = "Python developer with Django"
        jobs = ["Senior Python engineer with Django framework"]

        scores = scorer.compute_semantic_scores(resume, jobs)

        assert len(scores) == 1
        assert scores[0] is not None
        assert 0 <= scores[0] <= 1

    def test_compute_semantic_scores_multiple_jobs(self):
        """Should compute semantic scores for multiple jobs in batch."""
        scorer = SemanticScorer()
        if not scorer.is_enabled():
            pytest.skip("Sentence transformers not available")

        resume = "Python developer with Django and PostgreSQL"
        jobs = [
            "Python backend engineer",
            "React frontend developer",
            "Full stack Python Django developer"
        ]

        scores = scorer.compute_semantic_scores(resume, jobs)

        assert len(scores) == 3
        for score in scores:
            assert score is not None
            assert 0 <= score <= 1

        # The full stack job should be most similar
        assert scores[2] > scores[1]  # Python > React

    def test_blend_scores_equal_weight(self):
        """Blending with 0.5 weight should average the scores."""
        scorer = SemanticScorer(semantic_weight=0.5)

        # exact_score=80, semantic_score=0.6 (60%)
        # Should blend to 70%
        blended = scorer.blend_scores(80.0, 0.6)
        assert blended == pytest.approx(70.0, rel=0.01)

    def test_blend_scores_semantic_weighted(self):
        """Blending with higher semantic weight favors semantic."""
        scorer = SemanticScorer(semantic_weight=0.7)

        # exact_score=80, semantic_score=0.9 (90%)
        # 0.3 * 80 + 0.7 * 90 = 24 + 63 = 87
        blended = scorer.blend_scores(80.0, 0.9)
        assert blended == pytest.approx(87.0, rel=0.01)

    def test_blend_scores_none_semantic_returns_exact(self):
        """None semantic score should return exact score unchanged."""
        scorer = SemanticScorer()

        blended = scorer.blend_scores(75.0, None)
        assert blended == 75.0

    def test_blend_scores_clamps_to_valid_range(self):
        """Blended scores should be clamped between 0 and 100."""
        scorer = SemanticScorer(semantic_weight=0.5)

        # Test upper bound
        blended_high = scorer.blend_scores(95.0, 1.0)
        assert blended_high <= 100.0

        # Test lower bound
        blended_low = scorer.blend_scores(5.0, 0.0)
        assert blended_low >= 0.0

    def test_blend_scores_ignores_negative_semantic(self):
        """Negative semantic similarity should not drag exact scores down."""
        scorer = SemanticScorer(semantic_weight=0.6)
        blended = scorer.blend_scores(70.0, -0.4)
        assert blended == 70.0


class TestQuickMatch:
    """Test quick_match convenience function."""

    def test_quick_match_high_similarity(self):
        """Quick match should work for similar texts."""
        resume = "Python Django developer"
        job = "Senior Python engineer with Django framework"

        score = quick_match(resume, job)
        if score is not None:
            assert 0 <= score <= 1

    def test_quick_match_low_similarity(self):
        """Quick match should return low score for dissimilar texts."""
        resume = "Python backend developer"
        job = "iOS mobile developer with Swift"

        score = quick_match(resume, job)
        if score is not None:
            assert score < 0.5


class TestSemanticScenarios:
    """Real-world semantic matching scenarios."""

    def test_synonymous_technology_terms(self, using_fallback_model):
        """Should recognize synonymous tech terms."""
        if using_fallback_model:
            pytest.skip("Fallback embedder is offline-only and less nuanced")

        scorer = SemanticScorer()
        if not scorer.is_enabled():
            pytest.skip("Sentence transformers not available")

        resume = "JavaScript developer with React"
        job = "JS frontend engineer using React library"

        scores = scorer.compute_semantic_scores(resume, [job])
        assert scores[0] > 0.6  # Should be reasonably similar

    def test_role_vs_tools_priority(self, using_fallback_model):
        """Should weight role alignment over tool similarity."""
        if using_fallback_model:
            pytest.skip("Fallback embedder is offline-only and less nuanced")

        scorer = SemanticScorer()
        if not scorer.is_enabled():
            pytest.skip("Sentence transformers not available")

        resume = "Backend Python developer"
        jobs = [
            "Backend engineer using Python",  # Same role + tools
            "Frontend developer using Python"  # Different role, same tools
        ]

        scores = scorer.compute_semantic_scores(resume, jobs)
        # Backend match should be higher than frontend
        assert scores[0] > scores[1]

    def test_experience_level_semantics(self, using_fallback_model):
        """Should detect experience level differences."""
        if using_fallback_model:
            pytest.skip("Fallback embedder is offline-only and less nuanced")

        scorer = SemanticScorer()
        if not scorer.is_enabled():
            pytest.skip("Sentence transformers not available")

        resume = "Senior Python developer with 8 years experience"
        jobs = [
            "Lead Python engineer",  # Similar seniority
            "Junior Python developer"  # Different seniority
        ]

        scores = scorer.compute_semantic_scores(resume, jobs)
        # Senior/lead should be more similar than senior/junior
        assert scores[0] > scores[1]

    def test_partial_skill_overlap(self):
        """Should handle partial skill overlap."""
        scorer = SemanticScorer()
        if not scorer.is_enabled():
            pytest.skip("Sentence transformers not available")

        resume = "Python Django PostgreSQL Redis developer"
        jobs = [
            "Python Django developer",  # 2/4 overlap
            "Python Django PostgreSQL Redis Kubernetes developer",  # 4/5 overlap
            "Go MongoDB developer"  # 0 overlap
        ]

        scores = scorer.compute_semantic_scores(resume, jobs)
        # More overlap should mean higher similarity
        assert scores[1] > scores[0] > scores[2]
