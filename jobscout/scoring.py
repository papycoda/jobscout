"""Conservative job scoring system with stronger emphasis on languages/frameworks and role fit."""

import logging
from dataclasses import dataclass, field
from typing import List, Set, Dict
from .job_parser import ParsedJob


logger = logging.getLogger(__name__)

# High-signal skills (languages & frameworks) get extra weight in scoring
LANGUAGE_SKILLS = {
    "python", "javascript", "typescript", "java", "go", "c#", "c++", "ruby",
    "php", "rust", "swift", "kotlin", "scala"
}
FRAMEWORK_SKILLS = {
    "django", "fastapi", "flask", "spring", "express", "nestjs",
    "react", "vue", "angular", "svelte"
}
HIGH_VALUE_SKILLS = LANGUAGE_SKILLS | FRAMEWORK_SKILLS

# Role signals used to align candidate intent with job intent
ROLE_KEYWORDS: Dict[str, List[str]] = {
    "backend": ["backend", "back-end", "server", "api"],
    "frontend": ["frontend", "front-end", "ui", "javascript", "react", "vue", "angular"],
    "fullstack": ["fullstack", "full-stack", "full stack"],
    "devops": ["devops", "sre", "site reliability", "platform"],
    "data": ["data", "ml", "machine learning", "analytics"],
    "mobile": ["mobile", "ios", "android"],
}
ROLE_SKILL_HINTS: Dict[str, Set[str]] = {
    "backend": {
        "python", "java", "go", "ruby", "php", "rust", "scala",
        "django", "fastapi", "flask", "spring", "express", "nestjs",
        "postgresql", "mysql", "sqlite", "mongodb",
    },
    "frontend": {"react", "vue", "angular", "svelte", "javascript", "typescript"},
    "devops": {"kubernetes", "docker", "terraform", "aws", "gcp", "azure"},
    "data": {"postgresql", "mysql", "sqlite", "mongodb", "redis"},
    "mobile": {"swift", "kotlin"},
}


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
    role_alignment: float = 0.0  # 0-1

    # Details
    missing_must_haves: Set[str] = field(default_factory=set)
    matching_skills: Set[str] = field(default_factory=set)


class JobScorer:
    """
    Conservative job scoring system tuned for languages/frameworks and role intent.

    Scoring weights:
    1. Must-have coverage (55% - weighted toward languages/frameworks)
    2. Stack overlap (30% - weighted toward languages/frameworks)
    3. Role alignment (10% - backend/frontend/etc alignment)
    4. Seniority alignment (5%)

    Apply-ready thresholds:
    - Score ≥ min_score_threshold (default: 60%)
    - AND must-have coverage ≥ 60% (only if must-haves exist)
    - AND at least 2 matching skills
    - AND no hard role mismatch when roles are known

    If must-have list is empty:
    - Skip must-have coverage gate
    - Require score ≥ min_score_threshold (same as normal case)
    - Small penalty applied (-3 points)

    Penalties:
    - Unknown posting date: -3 points
    - Empty must-have list: -3 points
    """

    def __init__(self, resume, config, preferred_stack: Set[str]):
        """Initialize scorer with resume and preferences."""
        self.resume = resume
        self.config = config
        self.preferred_stack = preferred_stack

        # Combine resume skills + preferred stack for matching
        self.all_candidate_skills = resume.skills.copy()
        self.all_candidate_skills.update(preferred_stack)

        # Pre-compute candidate role signals
        resume_role_text = " ".join(resume.role_keywords or [])
        self.candidate_roles = self._extract_roles(resume_role_text, self.all_candidate_skills)

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
        # 1. Calculate must-have coverage (55% weight)
        coverage, missing_must_haves, matching_skills = self._calculate_must_have_coverage(job)

        # 2. Calculate stack overlap (30% weight)
        overlap = self._calculate_stack_overlap(job)

        # 3. Calculate seniority alignment (5% weight)
        seniority = self._calculate_seniority_alignment(job)

        # 4. Calculate role alignment (10% weight) to keep role intent in view
        role_alignment = self._calculate_role_alignment(job)

        # Apply weights
        score = (
            (coverage * 55)  # prioritize must-have coverage with extra weight on languages/frameworks
            + (overlap * 30)  # emphasize full stack overlap
            + (role_alignment * 10)  # consider role fit (backend/frontend/etc)
            + (seniority * 5)  # still consider seniority, but lighter
        )

        # Apply small penalty for unknown posting dates
        if not job.posted_date:
            score -= 3  # Small penalty for unknown date
            logger.debug("Job has unknown posting date, applied -3 penalty")

        # Apply penalty if must-have list is empty
        if not job.must_have_skills:
            coverage = 0.5
            score -= 3  # Reduced penalty from -10 to -3
            logger.debug("Job has no must-have skills, applied -3 penalty (coverage gate skipped)")

        # Clamp score to 0-100
        score = max(0, min(100, score))

        # Determine if apply-ready
        job_roles = self._extract_roles(job.title + " " + job.description, job.must_have_skills | job.nice_to_have_skills)
        is_apply_ready = self._is_apply_ready(score, coverage, job, job_roles)

        return ScoredJob(
            job=job,
            score=score,
            is_apply_ready=is_apply_ready,
            must_have_coverage=coverage,
            stack_overlap=overlap,
            seniority_alignment=seniority,
            role_alignment=role_alignment,
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

        # Weighted coverage: languages/frameworks count a bit more
        total_weight = sum(self._skill_weight(s) for s in must_haves)
        matched_weight = sum(self._skill_weight(s) for s in matching)
        coverage = matched_weight / total_weight if total_weight else 0.0
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
        total_weight = sum(self._skill_weight(s) for s in job_skills)
        overlap_weight = sum(self._skill_weight(s) for s in overlap)
        overlap_ratio = overlap_weight / total_weight if total_weight else 0.0

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

    def _is_apply_ready(self, score: float, must_have_coverage: float, job: ParsedJob, job_roles: Set[str]) -> bool:
        """
        Determine if job is apply-ready.

        Criteria:
        - Score ≥ min_score_threshold (60%)
        - AND must-have coverage ≥ 60% (only if must-haves exist)
        - AND at least 2-3 matching languages/frameworks
        - AND no hard role mismatch when roles are known

        If must-have list is empty:
        - Skip the must-have coverage gate
        - Require score ≥ min_score_threshold (same as normal case)
        - AND at least 2-3 matching total skills
        - AND role alignment not a hard mismatch (when roles are known)
        """
        # Calculate total matching skills (must-have + nice-to-have)
        all_job_skills = job.must_have_skills | job.nice_to_have_skills
        matching_skills = all_job_skills & self.all_candidate_skills
        matching_count = len(matching_skills)

        # If both sides have explicit roles and they do not overlap, treat as mismatch
        role_overlap = job_roles & self.candidate_roles
        role_known_mismatch = bool(job_roles) and bool(self.candidate_roles) and not role_overlap

        # Require at least 2 matching skills
        min_matching_skills = 2

        # If no must-haves, skip the coverage gate and use standard threshold
        if not job.must_have_skills:
            score_ok = score >= self.config.min_score_threshold
            has_enough_matches = matching_count >= min_matching_skills
            return score_ok and has_enough_matches and not role_known_mismatch

        # Normal case: check score, coverage, AND matching skill count
        score_ok = score >= self.config.min_score_threshold
        coverage_ok = must_have_coverage >= 0.6
        has_enough_matches = matching_count >= min_matching_skills

        return score_ok and coverage_ok and has_enough_matches and not role_known_mismatch

    def _skill_weight(self, skill: str) -> float:
        """Give languages/frameworks a bit more influence in scoring."""
        return 1.3 if skill in HIGH_VALUE_SKILLS else 1.0

    def _extract_roles(self, text: str, skills: Set[str]) -> Set[str]:
        """Extract coarse role buckets from text and skill hints."""
        roles = set()
        text_lower = text.lower() if text else ""

        explicit_role_found = False
        for role, keywords in ROLE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                roles.add(role)
                explicit_role_found = True

        # Only fall back to skill hints when explicit roles are not present
        if not explicit_role_found:
            for role, hints in ROLE_SKILL_HINTS.items():
                if skills & hints:
                    roles.add(role)

        # If both front/back present, mark as fullstack
        if "backend" in roles and "frontend" in roles:
            roles.add("fullstack")

        return roles

    def _calculate_role_alignment(self, job: ParsedJob) -> float:
        """Score how well the job's role focus matches the candidate's intent."""
        job_roles = self._extract_roles(job.title + " " + job.description, job.must_have_skills | job.nice_to_have_skills)

        if not job_roles or not self.candidate_roles:
            return 0.6  # Neutral when unclear

        overlap = job_roles & self.candidate_roles
        if overlap:
            # Strong alignment if roles overlap
            return 1.0

        # Mild penalty for mismatch when both sides are clear
        return 0.4
