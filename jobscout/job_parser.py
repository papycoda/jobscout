"""Parse job descriptions to extract requirements."""

import re
import logging
from dataclasses import dataclass
from typing import Set, List, Optional
from .resume_parser import SKILL_DICT


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

    def __init__(self):
        """Initialize parser with skill patterns."""
        self._build_patterns()
        self._build_section_patterns()

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

    def parse(self, job) -> ParsedJob:
        """Parse job listing to extract requirements."""
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
        # Split by common headers
        sections = re.split(
            r'\n\s*(requirements|qualifications|what you\'?ll bring|you have|responsibilities|about the role)\s*\n',
            full_description,
            flags=re.IGNORECASE
        )

        # Look for section that starts with requirements-related words
        for i, section in enumerate(sections):
            if i > 0 and section:
                # Check if this section is about requirements
                section_lower = section.lower()
                if any(pattern.search(section_lower) for pattern in self.must_have_patterns):
                    return section

        # Fallback: return full description
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

        # Look for nice-to-have section
        sections = re.split(
            r'\n\s*(nice to have|bonus|preferred|plus|desirable)\s*\n',
            full_description,
            flags=re.IGNORECASE
        )

        for i, section in enumerate(sections):
            if i > 0 and section:
                # Extract skills from this section
                skills = self._extract_skills(section)
                found_skills.update(skills)

        return found_skills

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
