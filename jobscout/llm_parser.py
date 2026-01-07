"""LLM-enhanced job parser using multiple providers."""

import json
import logging
from typing import Optional, Set, Dict, Any
from dataclasses import dataclass

from .llm_providers import MultiProviderLLMClient, DEFAULT_MODEL, get_model
from .job_parser import ParsedJob, JobParser


logger = logging.getLogger(__name__)


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
    Parse job descriptions using multiple LLM providers.

    Supports OpenAI (o3-mini, o1, gpt-4o, gpt-4o-mini) and DeepSeek.
    Falls back to regex-based parsing if LLM fails or is unavailable.
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
            model: Model ID to use (default: gpt-4o-mini)
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

    def parse(self, job_description: str, job_metadata: Optional[Dict] = None) -> ParsedJob:
        """
        Parse job description using LLM.

        Args:
            job_description: Full job description text
            job_metadata: Optional dict with title, company, etc.

        Returns:
            ParsedJob with extracted requirements

        Raises:
            Exception: If parsing fails and no fallback available
        """
        try:
            llm_result = self._parse_with_llm(job_description)
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

    def _parse_with_llm(self, job_description: str) -> LLMParseResult:
        """Parse job description using LLM API."""
        system_prompt = """You are an expert job description analyzer. Extract structured information from job postings.

Your task:
1. Identify MUST-HAVE skills (explicitly required)
2. Identify NICE-TO-HAVE skills (preferred but optional)
3. Determine seniority level (junior/mid/senior)
4. Extract minimum years of experience if stated
5. Identify deal-breakers (e.g., visa requirements, onsite requirements)

Rules:
- Be specific about technologies (e.g., "postgresql" not "databases")
- Include versions if specified (e.g., "python 3.8+")
- Group related skills (e.g., "react", "typescript", "javascript" are separate)
- If seniority is unclear, use "unknown"
- Only extract years if explicitly stated (don't infer from seniority)
- Deal-breakers are hard requirements that would disqualify a candidate

Return ONLY valid JSON. No markdown, no explanations."""

        user_prompt = f"""Analyze this job description:

{job_description[:4000]}

Return JSON in this exact format:
{{
    "must_have_skills": ["skill1", "skill2", "..."],
    "nice_to_have_skills": ["skill1", "skill2", "..."],
    "seniority_level": "junior|mid|senior|unknown",
    "min_years_experience": <number or null>,
    "deal_breakers": ["requirement1", "..."],
    "confidence": <0.0 to 1.0>
}}"""

        try:
            content = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=1000,
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
