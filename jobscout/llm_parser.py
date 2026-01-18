"""LLM-enhanced job parser using multiple providers."""

import json
import logging
from typing import Optional, Set, Dict, Any
from dataclasses import dataclass
from types import SimpleNamespace
from datetime import datetime

from .llm_providers import MultiProviderLLMClient, DEFAULT_MODEL, get_model
from .job_parser import ParsedJob, JobParser


logger = logging.getLogger(__name__)


# Canonical skill dictionary for consistency with resume parser
CANONICAL_SKILLS = {
    # Languages
    "python", "javascript", "typescript", "java", "go", "c#", "c++", "ruby", "php", "rust", "swift", "kotlin", "scala",
    # Frameworks
    "django", "fastapi", "flask", "spring", "express", "nestjs", "react", "vue", "angular", "svelte",
    # Infrastructure
    "docker", "kubernetes", "aws", "gcp", "azure", "terraform", "ansible", "chef", "puppet",
    # Databases
    "postgresql", "mysql", "sqlite", "mongodb", "redis", "elasticsearch", "dynamodb", "cassandra", "cockroachdb",
    # Messaging
    "kafka", "rabbitmq", "sqs", "pubsub",
    # Testing/CI
    "pytest", "unittest", "junit", "jest", "mocha", "github_actions", "gitlab_ci", "jenkins", "travis", "circleci",
}


@dataclass
class LLMParseResult:
    """Result from LLM parsing."""
    must_have_skills: Set[str]
    nice_to_have_skills: Set[str]
    seniority_level: str
    min_years_experience: Optional[float]
    deal_breakers: list[str]
    confidence: float
    role_keywords: list[str]


class LLMJobParser:
    """
    Parse job descriptions using multiple LLM providers with user context.

    This parser uses a much more sophisticated prompt that includes:
    - User's skills and experience
    - Job metadata (title, company, location)
    - Canonical skill dictionary for consistency
    - Few-shot examples for guidance
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        fallback_parser: Optional[JobParser] = None
    ):
        """
        Initialize LLM job parser.

        Args:
            api_key: API key for the LLM provider
            model: Model ID to use (default: gpt-5-mini)
            fallback_parser: Regex parser to use as fallback
        """
        self.model_id = model
        self.fallback_parser = fallback_parser or JobParser()

        try:
            self.client = MultiProviderLLMClient(model_id=model, api_key=api_key)
            model_info = self.client.get_model_info()
            logger.info(f"LLM parser initialized with {model_info['name']} ({model_info['provider']})")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise

    def parse(
        self,
        job_description: str,
        job_metadata: Optional[Dict] = None,
        user_skills: Optional[Set[str]] = None,
        user_seniority: str = "unknown",
        user_years_experience: float = 0.0,
        title_skills: Optional[Set[str]] = None
    ) -> ParsedJob:
        """
        Parse job description using LLM with user context.

        Args:
            job_description: Full job description text
            job_metadata: Optional dict with title, company, etc.
            user_skills: Optional set of user's skills for context
            user_seniority: Optional user's seniority level
            user_years_experience: Optional user's years of experience

        Returns:
            ParsedJob with extracted requirements
        """
        try:
            llm_result = self._parse_with_llm(
                job_description,
                job_metadata,
                user_skills or set(),
                user_seniority,
                user_years_experience,
                title_skills or set()
            )
            logger.info(f"LLM parsing successful (confidence: {llm_result.confidence:.2f})")

            return ParsedJob(
                title=job_metadata.get("title", "Unknown") if job_metadata else "Unknown",
                company=job_metadata.get("company", "Unknown") if job_metadata else "Unknown",
                location=job_metadata.get("location", "Unknown") if job_metadata else "Unknown",
                description=job_description,
                apply_url=job_metadata.get("apply_url", "") if job_metadata else "",
                source=job_metadata.get("source", "LLM") if job_metadata else "LLM",
                must_have_skills=llm_result.must_have_skills,
                nice_to_have_skills=llm_result.nice_to_have_skills,
                min_years_experience=llm_result.min_years_experience,
                seniority_level=llm_result.seniority_level,
                job_roles=self._infer_job_roles(job_description, job_metadata),
                posted_date=self._normalize_posted_date(job_metadata),
                role_keywords=set(llm_result.role_keywords),
            )

        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}. Falling back to regex parser.")
            if self.fallback_parser:
                return self._fallback_parse(job_description, job_metadata)
            raise

    def _parse_with_llm(
        self,
        job_description: str,
        job_metadata: Optional[Dict],
        user_skills: Set[str],
        user_seniority: str,
        user_years_experience: float,
        title_skills: Set[str] = set()
    ) -> LLMParseResult:
        """Parse job description using LLM API with user context."""

        # Build user context section
        user_context = f"""Candidate Profile:
- Skills: {sorted(user_skills) if user_skills else 'Not provided'}
- Seniority: {user_seniority}
- Years of Experience: {user_years_experience}"""

        # Build job metadata section
        job_info = f"""Job Information:
- Title: {job_metadata.get('title', 'Unknown') if job_metadata else 'Unknown'}
- Company: {job_metadata.get('company', 'Unknown') if job_metadata else 'Unknown'}
- Location: {job_metadata.get('location', 'Unknown') if job_metadata else 'Unknown'}"""

        # Add title skills if available (title is most reliable indicator)
        title_context = ""
        if title_skills:
            title_context = f"\n- Skills from Title: {sorted(title_skills)}"

        system_prompt = f"""You are an expert job matcher. Extract only the signals that matter for fit and search.

Rules (apply in order):
1) Use canonical skills only. Map variants to canonical list below; if no clear mapping, drop it.
2) Separate must-have vs nice-to-have using explicit language in the posting.
3) Ignore soft skills and generic terms (e.g., "communication", "self-starter", "database" without a specific technology).
4) Detect deal-breakers (visa/clearance/relocation/onsite-only/geography/certifications).
5) Recommend 3-5 short role keywords (2-4 words) aligned to the posting (e.g., "backend engineer", "fullstack engineer", "data engineer", "devops engineer"). Lowercase.
6) Confidence: 1.0 = explicit lists; 0.8 = mostly clear; 0.6 = vague; 0.4 = marketing fluff. One decimal place.

Canonical Skills (exact names):
Languages: python, javascript, typescript, java, go, c#, c++, ruby, php, rust, swift, kotlin, scala
Frameworks: django, fastapi, flask, spring, express, nestjs, react, vue, angular, svelte
Infrastructure: docker, kubernetes, aws, gcp, azure, terraform, ansible, chef, puppet
Databases: postgresql, mysql, sqlite, mongodb, redis, elasticsearch, dynamodb, cassandra, cockroachdb
Messaging: kafka, rabbitmq, sqs, pubsub
Testing/CI: pytest, unittest, junit, jest, mocha, github_actions, gitlab_ci, jenkins, travis, circleci

Seniority levels: junior (0-2y), mid (3-5y), senior (6+y; titles with Senior/Lead/Principal/Staff).

Return ONLY raw JSON, no code fences, no markdown."""

        # Few-shot examples to guide extraction
        examples = """Example:
Job: "Senior Python Engineer - Remote"
Description: "We're looking for a Senior Python Engineer with 8+ years experience. Required: Python, Django, PostgreSQL. Deploy with Docker/Kubernetes. AWS is a plus. Remote, but must be US-based; no visa sponsorship."

Result:
{
    "schema_version": "1.1",
    "must_have_skills": ["python", "django", "postgresql", "docker", "kubernetes"],
    "nice_to_have_skills": ["aws"],
    "seniority_level": "senior",
    "min_years_experience": 8.0,
    "deal_breakers": ["no visa sponsorship", "requires US-based candidates"],
    "role_keywords": ["backend engineer", "senior python engineer"],
    "confidence": 1.0
}
"""

        user_prompt = f"""{user_context}

{job_info}{title_context}

{examples}

**YOUR TASK:** Analyze the job description below and extract the requirements.

- Use only canonical skills. If a mentioned tech has no canonical match, omit it.
- Lowercase everything.
- Distinguish must-have vs nice-to-have from explicit wording.
- Extract deal-breakers that would disqualify candidates.
- Recommend 3-5 role keywords aligned to this posting.
- Return valid JSON only (no code fences). If unsure, return empty arrays and "unknown" seniority.

Job Description (may be long):
{job_description}

Return JSON in this exact format:
{{
    "schema_version": "1.1",
    "must_have_skills": ["skill1", "skill2", "..."],
    "nice_to_have_skills": ["skill1", "skill2", "..."],
    "seniority_level": "junior|mid|senior|unknown",
    "min_years_experience": <number or null>,
    "deal_breakers": ["dealbreaker1", "..."],
    "role_keywords": ["role1", "role2", "..."],
    "confidence": <0.0 to 1.0>
}}"""

        try:
            content = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,  # This model only supports temperature=1.0
                max_tokens=1500,
            )

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)

            return LLMParseResult(
                must_have_skills=set(skill.lower() for skill in result.get("must_have_skills", [])),
                nice_to_have_skills=set(skill.lower() for skill in result.get("nice_to_have_skills", [])),
                seniority_level=result.get("seniority_level", "unknown").lower(),
                min_years_experience=result.get("min_years_experience"),
                deal_breakers=result.get("deal_breakers", []),
                confidence=round(float(result.get("confidence", 0.8)), 1),
                role_keywords=[rk.strip().lower() for rk in result.get("role_keywords", []) if isinstance(rk, str) and rk.strip()]
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"LLM response: {content}")
            raise
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def _fallback_parse(self, job_description: str, job_metadata: Optional[Dict]) -> ParsedJob:
        """Fallback to regex-based parsing."""
        from .job_sources.base import JobListing

        # Create a mock job listing for the fallback parser
        mock_job = JobListing(
            title=job_metadata.get("title", "Unknown") if job_metadata else "Unknown",
            company=job_metadata.get("company", "Unknown") if job_metadata else "Unknown",
            location=job_metadata.get("location", "Unknown") if job_metadata else "Unknown",
            description=job_description,
            apply_url=job_metadata.get("apply_url", "") if job_metadata else "",
            source="Fallback",
        )

        return self.fallback_parser.parse(mock_job)

    def check_deal_breakers(
        self,
        job_description: str,
        user_constraints: Dict[str, Any]
    ) -> list[str]:
        """
        Check if job has deal-breakers for the user.

        Args:
            job_description: Job description text
            user_constraints: Dict with keys like:
                - requires_visa_sponsorship: bool
                - location_preference: "remote"|"hybrid"|"onsite"|"any"
                - max_commute_hours: float

        Returns:
            List of deal-breaker reasons (empty if none)
        """
        try:
            prompt = f"""Check if this job has deal-breakers for a candidate with these constraints:

User Constraints:
{json.dumps(user_constraints, indent=2)}

Job Description:
{job_description[:2000]}

Return JSON:
{{
    "has_deal_breakers": true/false,
    "reasons": ["reason1", "..."]
}}"""

            content = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a career advisor checking job compatibility."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
            )

            # Parse JSON response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)

            if result.get("has_deal_breakers"):
                return result.get("reasons", [])
            return []

        except Exception as e:
            logger.warning(f"Deal-breaker check failed: {e}")
            return []

    def _infer_job_roles(self, job_description: str, job_metadata: Optional[Dict]) -> Set[str]:
        """
        Reuse the regex role inference to keep downstream filters/scoring consistent.
        """
        if not self.fallback_parser:
            return set()

        stub_job = SimpleNamespace(
            title=job_metadata.get("title", "Unknown") if job_metadata else "Unknown",
            description=job_description,
        )
        try:
            return self.fallback_parser._extract_job_roles(stub_job)
        except Exception as exc:
            logger.debug(f"Could not infer job roles from LLM parse: {exc}")
            return set()

    def _normalize_posted_date(self, job_metadata: Optional[Dict]) -> Optional[str]:
        """Return posted_date in ISO format if present in metadata."""
        if not job_metadata:
            return None

        posted = job_metadata.get("posted_date")
        if isinstance(posted, datetime):
            return posted.isoformat()
        if isinstance(posted, str):
            return posted
        return None
