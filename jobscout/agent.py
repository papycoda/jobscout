"""Agent-based job search and matching."""

import logging
from typing import List, Optional
from .llm_client import LLMClient
from .agent_models import CVProfile, JobEvaluation, AgentConfig, JobSearchResult
from .job_sources.base import JobListing


logger = logging.getLogger(__name__)


class JobScoutAgent:
    """Agent that handles CV analysis, job search, and matching."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize agent with configuration.

        Args:
            config: Agent configuration (uses defaults if not provided)
        """
        self.config = config or AgentConfig()

        # Initialize LLM client
        self.llm = LLMClient(
            provider=self.config.llm_provider,
            api_key=self.config.llm_api_key,
            model=self.config.llm_model,
        )

        self.cv_profile: Optional[CVProfile] = None

    def analyze_cv(self, cv_text: str) -> CVProfile:
        """
        Analyze CV and extract candidate profile.

        Args:
            cv_text: Full text of the CV/resume

        Returns:
            CVProfile with extracted information
        """
        logger.info("Analyzing CV...")

        prompt = self._cv_analysis_prompt(cv_text)
        system_prompt = self._cv_analysis_system_prompt()

        try:
            response = self.llm.generate_json(prompt, system_prompt)
            profile = self._parse_cv_profile(response)

            self.cv_profile = profile
            logger.info(f"CV analyzed: {profile.role_primary} developer, {len(profile.skills)} skills, {profile.years_experience} years experience")
            return profile

        except Exception as e:
            logger.error(f"CV analysis failed: {e}")
            # Return minimal profile on failure
            return CVProfile(
                role_primary="unknown",
                seniority="unknown",
                skills=set(),
            )

    def generate_search_queries(self, profile: CVProfile) -> List[str]:
        """
        Generate diverse search queries based on CV profile.

        Args:
            profile: Candidate's CV profile

        Returns:
            List of 10-15 search terms
        """
        logger.info("Generating search queries...")

        prompt = self._search_query_prompt(profile)
        system_prompt = "You are a job search expert. Generate diverse search terms."

        try:
            response = self.llm.generate_json(prompt, system_prompt)
            queries = response.get("search_queries", [])

            # Ensure we have enough queries
            if len(queries) < 5:
                logger.warning(f"Only got {len(queries)} queries, using profile keywords as backup")
                queries.extend(profile.search_keywords)

            logger.info(f"Generated {len(queries)} search queries")
            return queries

        except Exception as e:
            logger.warning(f"Query generation failed: {e}, using profile keywords")
            return profile.search_keywords or ["software engineer", "developer"]

    def evaluate_job(
        self,
        job: JobListing,
        profile: Optional[CVProfile] = None,
    ) -> JobEvaluation:
        """
        Evaluate a job against the candidate's profile.

        Args:
            job: Job listing to evaluate
            profile: Candidate's CV profile (uses cached if not provided)

        Returns:
            JobEvaluation with match score and reasoning
        """
        if profile is None:
            profile = self.cv_profile

        if profile is None:
            raise ValueError("No profile available. Call analyze_cv() first.")

        prompt = self._job_evaluation_prompt(job, profile)
        system_prompt = self._job_evaluation_system_prompt()

        try:
            response = self.llm.generate_json(prompt, system_prompt)

            # Normalize score
            score = float(response.get("match_score", 0))
            score = max(0, min(100, score))

            return JobEvaluation.from_job_listing(
                job,
                match_score=score,
                is_match=bool(response.get("is_match", False)),
                role_aligned=response.get("role_aligned", True),
                seniority_aligned=response.get("seniority_aligned", True),
                location_compatible=response.get("location_compatible", True),
                required_skills_matched=response.get("required_skills_matched", []),
                required_skills_missing=response.get("required_skills_missing", []),
                summary=response.get("summary", ""),
                concerns=response.get("concerns", []),
                why_match=response.get("why_match", ""),
                salary=response.get("salary"),
            )

        except Exception as e:
            logger.warning(f"Job evaluation failed for {job.title}: {e}")
            # Return minimal evaluation on failure
            return JobEvaluation.from_job_listing(
                job,
                match_score=0,
                is_match=False,
                summary="Evaluation failed",
            )

    def evaluate_jobs_batch(
        self,
        jobs: List[JobListing],
        profile: Optional[CVProfile] = None,
    ) -> List[JobEvaluation]:
        """
        Evaluate multiple jobs.

        Args:
            jobs: List of job listings to evaluate
            profile: Candidate's CV profile (uses cached if not provided)

        Returns:
            List of JobEvaluation objects
        """
        if profile is None:
            profile = self.cv_profile

        if profile is None:
            raise ValueError("No profile available. Call analyze_cv() first.")

        evaluations = []
        for i, job in enumerate(jobs):
            try:
                eval_result = self.evaluate_job(job, profile)
                evaluations.append(eval_result)

                if (i + 1) % 20 == 0:
                    logger.info(f"Evaluated {i + 1}/{len(jobs)} jobs...")

            except Exception as e:
                logger.warning(f"Failed to evaluate {job.title}: {e}")
                continue

        return evaluations

    def rank_jobs(
        self,
        evaluations: List[JobEvaluation],
        limit: Optional[int] = None,
    ) -> List[JobEvaluation]:
        """
        Rank and filter job evaluations.

        Args:
            evaluations: List of job evaluations
            limit: Max number to return (defaults to config.max_results)

        Returns:
            Sorted list of top matches (is_match=True only)
        """
        if limit is None:
            limit = self.config.max_results

        # Filter to matches only
        matches = [e for e in evaluations if e.is_match]

        # Sort by score descending
        matches.sort(key=lambda e: e.match_score, reverse=True)

        # Return top N
        return matches[:limit]

    def run_search(
        self,
        cv_text: str,
        jobs: List[JobListing],
    ) -> JobSearchResult:
        """
        Run complete search: analyze CV, evaluate jobs, rank matches.

        Args:
            cv_text: Full text of CV/resume
            jobs: List of job listings to evaluate

        Returns:
            JobSearchResult with matches and statistics
        """
        import time
        start = time.time()

        # Step 1: Analyze CV
        profile = self.analyze_cv(cv_text)

        # Step 2: Evaluate all jobs
        evaluations = self.evaluate_jobs_batch(jobs, profile)

        # Step 3: Rank and filter
        ranked = self.rank_jobs(evaluations)

        elapsed = time.time() - start

        return JobSearchResult(
            matches=ranked,
            filtered_count=len(evaluations) - len([e for e in evaluations if e.is_match]),
            total_evaluated=len(evaluations),
            search_time_seconds=elapsed,
        )

    # Prompt methods

    def _cv_analysis_prompt(self, cv_text: str) -> str:
        """Generate prompt for CV analysis."""
        # Truncate CV if too long
        cv_text = cv_text[:12000]

        return f"""Analyze this resume and extract the following information.

RESUME:
{cv_text}

Return ONLY valid JSON with this exact structure:
{{
    "name": "Full name if found or null",
    "role_primary": "One of: backend, frontend, fullstack, data, devops, mobile",
    "seniority": "One of: junior, mid, senior, lead, principal, unknown",
    "skills": ["skill1", "skill2", ...],
    "languages": ["python", "javascript", ...],
    "frameworks": ["django", "react", ...],
    "databases": ["postgresql", "mongodb", ...],
    "infrastructure": ["docker", "aws", ...],
    "years_experience": number (as float, e.g., 5.5),
    "companies": ["company1", "company2"],
    "search_keywords": ["python backend engineer", "api developer", "python developer", ...],
    "preferred_locations": ["remote", "san francisco", ...],
    "target_companies": ["google", "stripe", ...]
}}

Requirements:
- Extract ALL technical skills mentioned
- Generate 10-15 diverse search keywords including: job titles, technologies, and domains
- Normalize all names to lowercase (e.g., "React.js" → "react")
- Infer seniority from experience, not just job titles
- If years_experience is unclear, estimate from career history
"""

    def _cv_analysis_system_prompt(self) -> str:
        """System prompt for CV analysis."""
        return """You are an expert technical recruiter and career coach.
Extract accurate information from resumes.
Normalize skill names to lowercase (e.g., "React.js" → "react", "PostgreSQL" → "postgresql").
Infer the primary role from the overall experience, not just keywords.
For search keywords, include: job titles, technologies, frameworks, and domain-specific terms (e.g., "ai engineer", "fintech")."""

    def _search_query_prompt(self, profile: CVProfile) -> str:
        """Generate prompt for search query generation."""
        return f"""Based on this candidate profile, generate 12-15 diverse search terms for job boards.

CANDIDATE PROFILE:
- Role: {profile.role_primary}
- Seniority: {profile.seniority}
- Years Experience: {profile.years_experience}
- Top Skills: {', '.join(list(profile.skills)[:15])}
- Languages: {', '.join(list(profile.languages))}
- Frameworks: {', '.join(list(profile.frameworks))}
- Current Keywords: {', '.join(profile.search_keywords[:8] if profile.search_keywords else [])}

Generate search terms across these categories:
1. Job titles (e.g., "python backend engineer", "senior backend developer")
2. Technologies (e.g., "python developer", "react native developer")
3. Domains/Specializations (e.g., "ai engineer", "ml engineer", "api platform engineer", "fintech backend")
4. Combined terms (e.g., "python django developer", "fullstack python react")

Return ONLY valid JSON:
{{
    "search_queries": ["term1", "term2", ...]
}}

Requirements:
- All terms should be lowercase
- Include variations of the same technology
- Think about how jobs are actually titled on job boards
"""

    def _job_evaluation_prompt(self, job: JobListing, profile: CVProfile) -> str:
        """Generate prompt for job evaluation."""
        # Truncate description if too long
        description = job.description[:6000]

        return f"""Evaluate if this job is a match for the candidate.

CANDIDATE PROFILE:
- Role: {profile.role_primary}
- Seniority: {profile.seniority}
- Years Experience: {profile.years_experience}
- Skills: {', '.join(list(profile.skills)[:25])}
- Languages: {', '.join(list(profile.languages))}
- Frameworks: {', '.join(list(profile.frameworks))}
- Databases: {', '.join(list(profile.databases))}

JOB POSTING:
Title: {job.title}
Company: {job.company}
Location: {job.location}
Description: {description}

Evaluate objectively:

1. ROLE ALIGNMENT: Is this the right type of role? (backend/frontend/data/devops/mobile)
2. SKILLS MATCH: Does the candidate have the REQUIRED skills? List what's matched and missing.
3. SENIORITY: Is the experience level appropriate?
4. LOCATION: Is the location compatible?

Return ONLY valid JSON:
{{
    "match_score": number (0-100),
    "is_match": true/false,
    "role_aligned": true/false,
    "seniority_aligned": true/false,
    "location_compatible": true/false,
    "required_skills_matched": ["skill1", "skill2"],
    "required_skills_missing": [],
    "summary": "Brief explanation (1-2 sentences)",
    "concerns": ["any red flags or concerns"],
    "why_match": "Why this is a good fit if score > 60",
    "salary": "Extracted salary range if mentioned or null"
}}

Scoring guidelines:
- 90-100: Excellent match - almost all requirements met
- 75-89: Good match - most requirements met, minor gaps
- 60-74: Possible match - some gaps but worth considering
- Below 60: Not a match

A job is NOT a match (is_match=false) if:
- Wrong role domain (e.g., pure data science for web backend developer)
- Missing critical required skills (core technologies)
- Significant seniority mismatch (e.g., junior role for senior candidate, or vice versa)
"""

    def _job_evaluation_system_prompt(self) -> str:
        """System prompt for job evaluation."""
        return """You are an objective job match evaluator.
Compare candidate qualifications to job requirements WITHOUT bias.
Focus on actual fit, not aspirational matches.
Consider that missing "nice to have" skills is fine.
Missing core "must have" skills is a problem.
Be conservative — it's better to miss a marginal match than include a bad one.
Normalize all skill names to lowercase."""