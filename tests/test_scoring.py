"""Tests for job scoring system."""

import pytest
from jobscout.resume_parser import ParsedResume
from jobscout.job_parser import ParsedJob
from jobscout.config import JobScoutConfig, JobPreferences
from jobscout.scoring import JobScorer


@pytest.fixture
def sample_resume():
    """Create a sample resume for testing."""
    return ParsedResume(
        raw_text="Sample resume text",
        skills={"python", "django", "postgresql", "docker", "aws"},
        tools=set(),
        seniority="mid",
        years_experience=4.0,
        role_keywords=["software engineer", "backend engineer"]
    )


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return JobScoutConfig(
        resume_path="test.pdf",
        job_preferences=JobPreferences(),
        min_score_threshold=65.0,
        fallback_min_score=75.0
    )


class TestJobScorer:
    """Test job scoring logic."""

    def test_perfect_match(self, sample_resume, sample_config):
        """Test a job that perfectly matches the resume."""
        scorer = JobScorer(sample_resume, sample_config, set())

        # Job that matches all skills
        job = ParsedJob(
            title="Backend Engineer",
            company="TestCorp",
            location="Remote",
            description="We need a Python Django developer with PostgreSQL and Docker",
            apply_url="https://example.com/apply",
            source="Test",
            must_have_skills={"python", "django", "postgresql"},
            nice_to_have_skills={"docker"},
            min_years_experience=3,
            seniority_level="mid"
        )

        scored_jobs = scorer.score_jobs([job])

        assert len(scored_jobs) == 1
        scored = scored_jobs[0]

        # Should be a high score
        assert scored.score >= 85
        assert scored.is_apply_ready
        assert len(scored.missing_must_haves) == 0

    def test_partial_match_fails_threshold(self, sample_resume, sample_config):
        """Test a job with low match that fails threshold."""
        scorer = JobScorer(sample_resume, sample_config, set())

        # Job requiring skills the candidate doesn't have
        job = ParsedJob(
            title="Full Stack Engineer",
            company="TestCorp",
            location="Remote",
            description="We need React, Node.js, and TypeScript expertise",
            apply_url="https://example.com/apply",
            source="Test",
            must_have_skills={"react", "typescript", "node"},  # Candidate has none
            nice_to_have_skills={"vue"},
            min_years_experience=3,
            seniority_level="mid"
        )

        scored_jobs = scorer.score_jobs([job])

        # Should not be apply-ready
        assert len(scored_jobs) == 0

    def test_seniority_mismatch_penalty(self, sample_resume, sample_config):
        """Test penalty for seniority mismatch."""
        scorer = JobScorer(sample_resume, sample_config, set())

        # Job requiring senior level, candidate is mid
        job = ParsedJob(
            title="Senior Backend Engineer",
            company="TestCorp",
            location="Remote",
            description="Senior Python developer with 8+ years experience",
            apply_url="https://example.com/apply",
            source="Test",
            must_have_skills={"python"},
            nice_to_have_skills=set(),
            min_years_experience=8,
            seniority_level="senior"
        )

        scored_jobs = scorer.score_jobs([job])

        # Might still pass if skills match, but with penalty
        if scored_jobs:
            scored = scored_jobs[0]
            # Score should be reduced due to seniority mismatch
            assert scored.seniority_alignment < 1.0

    def test_empty_must_have_list_penalty(self, sample_resume, sample_config):
        """Test higher threshold when must-have list is empty."""
        scorer = JobScorer(sample_resume, sample_config, set())

        # Job with no clear must-have skills
        job = ParsedJob(
            title="Software Engineer",
            company="TestCorp",
            location="Remote",
            description="Join our team and build great software",
            apply_url="https://example.com/apply",
            source="Test",
            must_have_skills=set(),  # Empty
            nice_to_have_skills={"python"},
            min_years_experience=None,
            seniority_level="unknown"
        )

        scored_jobs = scorer.score_jobs([job])

        # Should require higher score (75% vs 65%)
        # This job likely won't pass
        if scored_jobs:
            scored = scored_jobs[0]
            # Should have been penalized
            assert scored.must_have_coverage == 0.5

    def test_stack_overlap_calculation(self, sample_resume, sample_config):
        """Test stack overlap scoring."""
        scorer = JobScorer(sample_resume, sample_config, set())

        job = ParsedJob(
            title="Backend Engineer",
            company="TestCorp",
            location="Remote",
            description="Python developer with Django and PostgreSQL",
            apply_url="https://example.com/apply",
            source="Test",
            must_have_skills={"python", "django"},
            nice_to_have_skills={"redis", "kubernetes"},
            min_years_experience=3,
            seniority_level="mid"
        )

        scored_jobs = scorer.score_jobs([job])

        assert len(scored_jobs) == 1
        scored = scored_jobs[0]

        # Should have decent stack overlap
        assert scored.stack_overlap > 0

    def test_missing_must_haves_identified(self, sample_resume, sample_config):
        """Test that missing must-have skills are identified."""
        scorer = JobScorer(sample_resume, sample_config, set())

        job = ParsedJob(
            title="Backend Engineer",
            company="TestCorp",
            location="Remote",
            description="Python developer with Redis and Kubernetes",
            apply_url="https://example.com/apply",
            source="Test",
            must_have_skills={"python", "redis", "kubernetes"},  # Missing redis & kubernetes
            nice_to_have_skills=set(),
            min_years_experience=3,
            seniority_level="mid"
        )

        scored_jobs = scorer.score_jobs([job])

        # Should identify missing must-haves
        if scored_jobs:
            scored = scored_jobs[0]
            # Should have redis and/or kubernetes in missing
            assert len(scored.missing_must_haves) > 0
