"""Conservative job scoring system."""

import logging
from dataclasses import dataclass
from typing import List, Set
from .job_parser import ParsedJob


logger = logging.getLogger(__name__)


@dataclass
class ScoredJob:
    """Job with score and breakdown."""
    job: ParsedJob
    score: float
    is_apply_ready: bool

    # Score breakdown
    must_have_coverage: float  # 0-1
    stack_overlap: float  # 0-1
    seniority_alignment: float  # 0-1

    # Details
    missing_must_haves: Set[str]
    matching_skills: Set[str]


class JobScorer:
    """
    Conservative job scoring system.

    Scoring weights:
    1. Must-have coverage (60%)
    2. Stack overlap (25%)
    3. Seniority alignment (15%)

    Apply-ready thresholds:
    - Score ≥ 65%
    - AND must-have coverage ≥ 60%

    If must-have list is empty:
    - Require score ≥ 75%
    """

    def __init__(self, resume, config, preferred_stack: Set[str]):
        """Initialize scorer with resume and preferences."""
        self.resume = resume
        self.config = config
        self.preferred_stack = preferred_stack

        # Combine resume skills + preferred stack for matching
        self.all_candidate_skills = resume.skills.copy()
        self.all_candidate_skills.update(preferred_stack)

    def score_jobs(self, jobs: List[ParsedJob]) -> List[ScoredJob]:
        """Score all jobs and return apply-ready ones."""
        scored_jobs = []

        for job in jobs:
            scored = self._score_job(job)
            if scored.is_apply_ready:
                scored_jobs.append(scored)
            else:
                logger.debug(
                    f"Job '{job.title}' at {job.company} scored {scored.score:.1f}% "
                    f"(below threshold or must-have coverage too low)"
                )

        # Sort by score descending
        scored_jobs.sort(key=lambda j: j.score, reverse=True)

        logger.info(f"Found {len(scored_jobs)} apply-ready jobs")
        return scored_jobs

    def _score_job(self, job: ParsedJob) -> ScoredJob:
        """Score a single job."""
        # 1. Calculate must-have coverage (60% weight)
        coverage, missing_must_haves, matching_skills = self._calculate_must_have_coverage(job)

        # 2. Calculate stack overlap (25% weight)
        overlap = self._calculate_stack_overlap(job)

        # 3. Calculate seniority alignment (15% weight)
        seniority = self._calculate_seniority_alignment(job)

        # Apply weights
        score = (coverage * 60) + (overlap * 25) + (seniority * 15)

        # Apply penalty if must-have list is empty
        if not job.must_have_skills:
            coverage = 0.5
            score -= 10
            logger.debug(f"Job has no must-have skills, applied -10 penalty")

        # Clamp score to 0-100
        score = max(0, min(100, score))

        # Determine if apply-ready
        is_apply_ready = self._is_apply_ready(score, coverage, job)

        return ScoredJob(
            job=job,
            score=score,
            is_apply_ready=is_apply_ready,
            must_have_coverage=coverage,
            stack_overlap=overlap,
            seniority_alignment=seniority,
            missing_must_haves=missing_must_haves,
            matching_skills=matching_skills
        )

    def _calculate_must_have_coverage(self, job: ParsedJob) -> tuple[float, Set[str], Set[str]]:
        """
        Calculate must-have skill coverage.

        Returns: (coverage_ratio, missing_must_haves, matching_skills)
        """
        must_haves = job.must_have_skills

        if not must_haves:
            # Empty must-have list = handled in _score_job
            return 0.0, set(), set()

        # Find which must-haves the candidate has
        matching = set()
        missing = set()

        for skill in must_haves:
            if skill in self.all_candidate_skills:
                matching.add(skill)
            else:
                missing.add(skill)

        coverage = len(matching) / len(must_haves)
        return coverage, missing, matching

    def _calculate_stack_overlap(self, job: ParsedJob) -> float:
        """
        Calculate stack overlap.

        Overlap between (job must + nice skills) and (resume + preferred stack).
        """
        # All job skills (must + nice)
        job_skills = job.must_have_skills | job.nice_to_have_skills

        if not job_skills:
            return 0.5  # Neutral score if no skills identified

        # Calculate overlap
        overlap = job_skills & self.all_candidate_skills
        overlap_ratio = len(overlap) / len(job_skills)

        return overlap_ratio

    def _calculate_seniority_alignment(self, job: ParsedJob) -> float:
        """
        Calculate seniority alignment.

        - Penalize if job requires 8+ years and resume <5
        - Mild penalty if job is junior and resume is clearly senior
        """
        if job.seniority_level == "unknown" or self.resume.seniority == "unknown":
            return 0.7  # Neutral score when unknown

        # Perfect match
        if job.seniority_level == self.resume.seniority:
            return 1.0

        # Calculate alignment based on seniority levels
        alignment = 0.7  # Default to decent alignment

        # Job wants senior, candidate is junior/mid
        if job.seniority_level == "senior" and self.resume.seniority in ["junior", "mid"]:
            # Check years-based penalty
            if job.min_years_experience and job.min_years_experience >= 8:
                if self.resume.years_experience < 5:
                    alignment = 0.3  # Strong penalty
                else:
                    alignment = 0.5  # Moderate penalty
            else:
                alignment = 0.6

        # Job is junior, candidate is senior
        elif job.seniority_level == "junior" and self.resume.seniority == "senior":
            alignment = 0.6  # Mild penalty (may be overqualified)

        # Job is mid, candidate is junior
        elif job.seniority_level == "mid" and self.resume.seniority == "junior":
            alignment = 0.5

        # Job is junior/mid, candidate is senior (not penalized)
        elif self.resume.seniority == "senior":
            alignment = 0.9  # Senior can step down

        return alignment

    def _is_apply_ready(self, score: float, must_have_coverage: float, job: ParsedJob) -> bool:
        """
        Determine if job is apply-ready.

        Criteria:
        - Score ≥ 65%
        - AND must-have coverage ≥ 60%

        If must-have list is empty:
        - Require score ≥ 75%
        """
        # If no must-haves, use higher threshold
        if not job.must_have_skills:
            return score >= self.config.fallback_min_score

        # Normal case: check both score and coverage
        score_ok = score >= self.config.min_score_threshold
        coverage_ok = must_have_coverage >= 0.6

        return score_ok and coverage_ok
