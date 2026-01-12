"""LLM-enhanced job parser using multiple providers."""

import json
import logging
from typing import Optional, Set, Dict, Any
from dataclasses import dataclass

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
        user_years_experience: float = 0.0
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
                user_years_experience
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
        user_years_experience: float
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

        system_prompt = """You are an expert job matcher and career advisor. Your task is to analyze job postings and extract requirements in a way that helps match candidates to jobs.

**Your goal**: Extract the skills and requirements that ACTUALLY MATTER for determining if a candidate is a good fit.

**Key Principles**:
1. **Be Specific**: Extract concrete technologies (e.g., "postgresql" not "databases", "react" not "javascript frameworks")
2. **Use Canonical Names**: Map variations to standard names (e.g., "node.js" → "javascript", "postgres" → "postgresql")
3. **Distinguish Must-Have vs Nice-to-Have**:
   - Must-have: Explicitly required skills (e.g., "required", "must have", "you need", "we're looking for")
   - Nice-to-have: Preferred but optional (e.g., "nice to have", "bonus", "plus", "preferred")
4. **Consider Context**: A skill mentioned once in a 20-page description is less important than one mentioned 5 times
5. **Look for Deal-Breakers**: Requirements that would disqualify a candidate (visa, relocation, specific certifications)

**Canonical Skills Reference** (use these exact names):
Languages: python, javascript, typescript, java, go, c#, c++, ruby, php, rust, swift, kotlin, scala
Frameworks: django, fastapi, flask, spring, express, nestjs, react, vue, angular, svelte
Infrastructure: docker, kubernetes, aws, gcp, azure, terraform, ansible, chef, puppet
Databases: postgresql, mysql, sqlite, mongodb, redis, elasticsearch, dynamodb, cassandra, cockroachdb
Messaging: kafka, rabbitmq, sqs, pubsub
Testing/CI: pytest, unittest, junit, jest, mocha, github_actions, gitlab_ci, jenkins, travis, circleci

**Seniority Levels**:
- "junior": 0-2 years, titles like "Junior", "Entry Level", "Associate"
- "mid": 3-5 years, titles like "Mid-Level", "Intermediate", "Software Engineer"
- "senior": 6+ years, titles like "Senior", "Lead", "Principal", "Staff"

**Deal-Breakers** (hard requirements that disqualify):
- Visa sponsorship required/not provided
- Onsite-only requirements
- Specific certifications (e.g., "must have AWS Solutions Architect Professional")
- Industry-specific requirements (e.g., "healthcare experience required")
- Geographic restrictions

**Confidence Scoring**:
- 1.0: All requirements clearly stated, explicit must-have list
- 0.8: Most requirements clear, some ambiguity
- 0.6: Requirements vague or unclear
- 0.4: Very poor quality job description

Return ONLY valid JSON. No markdown, no explanations."""

        # Few-shot examples to guide extraction
        examples = """**EXAMPLE 1:**
Job: "Senior Python Engineer - Remote"
Description: "We're looking for a Senior Python Engineer with 5+ years of experience. You must have strong Python skills, Django experience, and PostgreSQL knowledge. AWS experience is a plus. You should be familiar with Docker and Kubernetes for deployment. Required: 8+ years total experience."

Result:
{
    "must_have_skills": ["python", "django", "postgresql", "docker", "kubernetes"],
    "nice_to_have_skills": ["aws"],
    "seniority_level": "senior",
    "min_years_experience": 8.0,
    "deal_breakers": [],
    "confidence": 1.0
}

**EXAMPLE 2:**
Job: "Full Stack Developer"
Description: "Join our team as a Full Stack Developer! We use React, Node.js, and MongoDB. Experience with TypeScript is preferred but not required. You should know SQL and have worked with cloud platforms before. Nice to have: GraphQL, Redis."

Result:
{
    "must_have_skills": ["react", "javascript", "mongodb", "postgresql"],
    "nice_to_have_skills": ["typescript", "graphql", "redis"],
    "seniority_level": "unknown",
    "min_years_experience": null,
    "deal_breakers": [],
    "confidence": 0.8
}

**EXAMPLE 3:**
Job: "Backend Engineer (Python/Django)"
Description: "Must be authorized to work in US without sponsorship. Required: 3+ years Python, Django REST Framework, PostgreSQL. Bonus points for: Redis, Celery, Docker. This is an onsite role in San Francisco - no remote."

Result:
{
    "must_have_skills": ["python", "django", "postgresql"],
    "nice_to_have_skills": ["redis", "docker"],
    "seniority_level": "mid",
    "min_years_experience": 3.0,
    "deal_breakers": ["requires US work authorization (no sponsorship)", "onsite only (San Francisco)"],
    "confidence": 1.0
}
"""

        user_prompt = f"""{user_context}

{job_info}

{examples}

**YOUR TASK:**
Analyze the job description below and extract the requirements.

**IMPORTANT**:
- Map skills to canonical names from the reference list above
- Use lowercase for all skill names
- Distinguish between must-have and nice-to-have based on wording
- Extract deal-breakers that would disqualify the candidate
- Set confidence based on how clearly requirements are stated

Job Description:
{job_description}

Return JSON in this exact format:
{{
    "must_have_skills": ["skill1", "skill2", "..."],
    "nice_to_have_skills": ["skill1", "skill2", "..."],
    "seniority_level": "junior|mid|senior|unknown",
    "min_years_experience": <number or null>,
    "deal_breakers": ["dealbreaker1", "..."],
    "confidence": <0.0 to 1.0>
}}"""

        try:
            content = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0 if self.model_id != "gpt-4o-mini" else 1.0,  # Some models only support 1.0
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
                confidence=float(result.get("confidence", 0.8))
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
