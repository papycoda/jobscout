"""Parse job descriptions to extract requirements."""

import re
import logging
from dataclasses import dataclass
from typing import Set, List, Optional, TYPE_CHECKING
from .resume_parser import SKILL_DICT

if TYPE_CHECKING:
    from .config import JobScoutConfig


logger = logging.getLogger(__name__)


@dataclass
class ParsedJob:
    """Parsed job requirements."""
    title: str
    company: str
    location: str
    description: str
    apply_url: str
    source: str

    # Extracted requirements
    must_have_skills: Set[str]
    nice_to_have_skills: Set[str]

    # Seniority requirements
    min_years_experience: Optional[float] = None
    seniority_level: str = "unknown"  # junior, mid, senior, unknown

    # Metadata
    posted_date: Optional[str] = None


class JobParser:
    """Parse job descriptions to extract structured requirements."""

    def __init__(self, config: Optional["JobScoutConfig"] = None):
        """
        Initialize parser with skill patterns and optional LLM support.

        Args:
            config: Optional config for LLM parser settings
        """
        self._build_patterns()
        self._build_section_patterns()
        self.llm_parser = None
        self.use_llm = False

        if config and config.use_llm_parser and config.openai_api_key:
            try:
                from .llm_parser import LLMJobParser
                self.llm_parser = LLMJobParser(
                    api_key=config.openai_api_key,
                    model=config.openai_model,
                    fallback_parser=self
                )
                self.use_llm = True
                logger.info(f"LLM parser enabled with model: {config.openai_model}")
            except ImportError as e:
                logger.warning(f"Failed to initialize LLM parser: {e}. Using regex parser.")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM parser: {e}. Using regex parser.")

    def _build_patterns(self):
        """Build regex patterns for skill matching."""
        self.patterns = {}
        for canonical, variants in SKILL_DICT.items():
            pattern = r'\b(' + '|'.join(re.escape(v) for v in variants) + r')\b'
            self.patterns[canonical] = re.compile(pattern, re.IGNORECASE)

    def _build_section_patterns(self):
        """Build patterns to identify requirement sections."""
        # Patterns that indicate must-have requirements
        self.must_have_patterns = [
            re.compile(r'(requirements|qualifications|what you\'ll bring|you have)', re.IGNORECASE),
            re.compile(r'(must have|required|essential)', re.IGNORECASE),
        ]

        # Patterns that indicate nice-to-have requirements
        self.nice_to_have_patterns = [
            re.compile(r'(nice to have|bonus|preferred|plus|desirable)', re.IGNORECASE),
        ]

        # Patterns that indicate experience requirements
        self.experience_patterns = [
            re.compile(r'(\d+)\+?\s*years?\s*(of)?\s*experience', re.IGNORECASE),
            re.compile(r'(\d+)\+?\s*years?\s*(of)?\s*(professional|work)', re.IGNORECASE),
        ]

        self.primary_requirement_headers = [
            "requirements",
            "qualifications",
            "what you'll bring",
            "you have",
            "must have",
            "required",
            "essential",
        ]
        self.secondary_requirement_headers = [
            "responsibilities",
            "about the role",
        ]
        self.nice_to_have_headers = [
            "nice to have",
            "bonus",
            "preferred",
            "plus",
            "desirable",
        ]

        stop_headers = (
            self.primary_requirement_headers
            + self.secondary_requirement_headers
            + self.nice_to_have_headers
        )
        stop_pattern = r'\n\s*(' + '|'.join(re.escape(h) for h in stop_headers) + r')\s*\n'
        self.section_stop_pattern = re.compile(stop_pattern, re.IGNORECASE)

    def parse(self, job) -> ParsedJob:
        """
        Parse job listing to extract requirements.

        Uses LLM parser if enabled and available, otherwise falls back to regex.
        """
        # Try LLM parser if enabled
        if self.use_llm and self.llm_parser:
            try:
                job_metadata = {
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "apply_url": job.apply_url,
                    "source": job.source,
                }
                return self.llm_parser.parse(job.description, job_metadata)
            except Exception as e:
                logger.warning(f"LLM parsing failed for {job.title} at {job.company}: {e}. Falling back to regex.")

        # Fallback to regex-based parsing
        description = job.description.lower()

        # Find requirements section
        requirements_text = self._extract_requirements_section(job.description)

        # Extract skills from requirements section
        must_have = self._extract_skills(requirements_text)
        nice_to_have = self._extract_nice_to_have(job.description)

        # Extract seniority requirements
        years, seniority = self._extract_seniority_requirements(job.description)

        return ParsedJob(
            title=job.title,
            company=job.company,
            location=job.location,
            description=job.description,
            apply_url=job.apply_url,
            source=job.source,
            must_have_skills=must_have,
            nice_to_have_skills=nice_to_have,
            min_years_experience=years,
            seniority_level=seniority,
            posted_date=job.posted_date.isoformat() if job.posted_date else None
        )

    def _extract_requirements_section(self, full_description: str) -> str:
        """Extract the requirements/qualifications section from description."""
        section = self._extract_section_by_headers(full_description, self.primary_requirement_headers)
        if section:
            return section

        section = self._extract_section_by_headers(full_description, self.secondary_requirement_headers)
        if section:
            return section

        return full_description

    def _extract_skills(self, text: str) -> Set[str]:
        """Extract must-have skills from text."""
        found_skills = set()

        for canonical, pattern in self.patterns.items():
            if pattern.search(text):
                found_skills.add(canonical)

        return found_skills

    def _extract_nice_to_have(self, full_description: str) -> Set[str]:
        """Extract nice-to-have skills from description."""
        found_skills = set()

        section = self._extract_section_by_headers(full_description, self.nice_to_have_headers)
        if section:
            found_skills.update(self._extract_skills(section))

        return found_skills

    def _extract_section_by_headers(self, full_description: str, headers: List[str]) -> str:
        """Extract the first section that follows any header in the list."""
        if not headers:
            return ""

        header_pattern = '|'.join(re.escape(h) for h in headers)
        sections = re.split(
            rf'\n\s*({header_pattern})\s*\n',
            full_description,
            flags=re.IGNORECASE
        )

        if len(sections) < 3:
            return ""

        for i in range(1, len(sections), 2):
            content = sections[i + 1] if i + 1 < len(sections) else ""
            if content:
                return self._truncate_section(content)

        return ""

    def _truncate_section(self, section_text: str) -> str:
        """Trim a section at the next header boundary."""
        parts = self.section_stop_pattern.split(section_text)
        if parts:
            return parts[0].strip()
        return section_text.strip()

    def _extract_seniority_requirements(self, description: str) -> tuple[Optional[float], str]:
        """Extract experience and seniority requirements."""
        # Extract years of experience
        years = None
        for pattern in self.experience_patterns:
            matches = pattern.findall(description)
            for match in matches:
                try:
                    years_val = float(match[0])
                    if years is None or years_val > years:
                        years = years_val
                except (ValueError, IndexError):
                    continue

        # Infer seniority level from title and description
        description_lower = description.lower()

        # Check for explicit seniority keywords
        if any(keyword in description_lower for keyword in ['senior', 'sr.', 'sr ', 'lead', 'principal', 'staff']):
            seniority = "senior"
        elif any(keyword in description_lower for keyword in ['junior', 'jr.', 'jr ', 'entry level', 'associate']):
            seniority = "junior"
        elif any(keyword in description_lower for keyword in ['mid-level', 'midlevel', 'intermediate']):
            seniority = "mid"
        else:
            seniority = "unknown"

        # Use years as additional signal
        if years:
            if years >= 7 and seniority == "unknown":
                seniority = "senior"
            elif years >= 3 and seniority == "unknown":
                seniority = "mid"
            elif years < 3 and seniority == "unknown":
                seniority = "junior"

        return years, seniority
