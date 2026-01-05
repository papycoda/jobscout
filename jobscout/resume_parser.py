"""Resume parsing and skill extraction."""

import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
import docx
import pdfplumber


# Canonical skill dictionary for matching
SKILL_DICT = {
    # Languages
    "python": ["python", "py"],
    "javascript": ["javascript", "js", "nodejs", "node.js"],
    "typescript": ["typescript", "ts"],
    "java": ["java"],
    "go": ["go", "golang"],
    "c#": ["c#", "csharp"],
    "c++": ["c++", "cpp"],
    "ruby": ["ruby", "rails"],
    "php": ["php"],
    "rust": ["rust"],
    "swift": ["swift"],
    "kotlin": ["kotlin"],
    "scala": ["scala"],

    # Frameworks & Libraries
    "django": ["django"],
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "spring": ["spring boot", "springboot", "spring"],
    "express": ["express", "express.js"],
    "nestjs": ["nestjs", "nest.js"],
    "react": ["react", "reactjs", "react.js"],
    "vue": ["vue", "vue.js", "vuejs"],
    "angular": ["angular"],
    "svelte": ["svelte"],

    # Infrastructure
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s", "k8"],
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "azure": ["azure", "microsoft azure"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "chef": ["chef"],
    "puppet": ["puppet"],

    # Databases
    "postgresql": ["postgresql", "postgres", "psql"],
    "mysql": ["mysql"],
    "sqlite": ["sqlite"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "elasticsearch": ["elasticsearch", "elastic search"],
    "dynamodb": ["dynamodb"],
    "cassandra": ["cassandra"],
    "cockroachdb": ["cockroachdb", "cockroach"],

    # Messaging & Queues
    "kafka": ["kafka"],
    "rabbitmq": ["rabbitmq", "rabbit mq"],
    "sqs": ["sqs", "amazon sqs"],
    "pubsub": ["pubsub", "google pubsub"],

    # Testing & CI/CD
    "pytest": ["pytest"],
    "unittest": ["unittest", "unit test"],
    "junit": ["junit"],
    "jest": ["jest"],
    "mocha": ["mocha"],
    "github_actions": ["github actions", "github actions ci"],
    "gitlab_ci": ["gitlab ci", "gitlab-ci"],
    "jenkins": ["jenkins"],
    "travis": ["travis ci"],
    "circleci": ["circleci"],
}


# Seniority keywords for inference
SENIORITY_KEYWORDS = {
    "junior": ["junior", "jr", "entry level", "associate"],
    "mid": ["mid-level", "midlevel", "mid level", "intermediate"],
    "senior": ["senior", "sr", "lead", "principal", "staff"],
}

ROLE_TITLE_HINTS = [
    "engineer",
    "developer",
    "architect",
    "manager",
    "scientist",
    "analyst",
    "devops",
    "sre",
]

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass
class ParsedResume:
    """Extracted resume data."""
    raw_text: str
    skills: Set[str] = field(default_factory=set)
    tools: Set[str] = field(default_factory=set)
    seniority: str = "unknown"  # junior, mid, senior, unknown
    years_experience: float = 0.0
    role_keywords: List[str] = field(default_factory=list)


class ResumeParser:
    """Parse resumes and extract skills using canonical dictionary."""

    def __init__(self):
        """Initialize parser with skill patterns."""
        self._build_patterns()

    def _build_patterns(self):
        """Build regex patterns for skill matching."""
        self.patterns = {}
        for canonical, variants in SKILL_DICT.items():
            # Build pattern that matches whole words only
            pattern = r'\b(' + '|'.join(re.escape(v) for v in variants) + r')\b'
            self.patterns[canonical] = re.compile(pattern, re.IGNORECASE)

    def parse(self, file_path: str) -> ParsedResume:
        """Parse resume from file (PDF, DOCX, or TXT)."""
        path = Path(file_path)

        if path.suffix.lower() == '.pdf':
            text = self._parse_pdf(path)
        elif path.suffix.lower() == '.docx':
            text = self._parse_docx(path)
        elif path.suffix.lower() == '.txt':
            text = self._parse_txt(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return self._extract_from_text(text)

    def _parse_pdf(self, path: Path) -> str:
        """Extract text from PDF."""
        text_chunks = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_chunks.append(page_text)
        return "\n".join(text_chunks)

    def _parse_docx(self, path: Path) -> str:
        """Extract text from DOCX."""
        doc = docx.Document(path)
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)

    def _parse_txt(self, path: Path) -> str:
        """Extract text from TXT."""
        return path.read_text(encoding='utf-8', errors='ignore')

    def _extract_from_text(self, text: str) -> ParsedResume:
        """Extract skills and metadata from text."""
        # Extract skills using canonical dictionary
        found_skills = set()
        for canonical, pattern in self.patterns.items():
            if pattern.search(text):
                found_skills.add(canonical)

        # Extract years of experience
        years = self._extract_years_experience(text)

        # Infer seniority
        seniority = self._infer_seniority(text, years)

        # Extract role keywords
        role_keywords = self._extract_role_keywords(text)

        return ParsedResume(
            raw_text=text,
            skills=found_skills,
            tools=set(),  # Tools are extracted as skills
            seniority=seniority,
            years_experience=years,
            role_keywords=role_keywords
        )

    def _extract_years_experience(self, text: str) -> float:
        """Extract approximate years of experience."""
        patterns = [
            r'(\d+(?:\.\d+)?)\+?\s*years?\s*(of)?\s*experience',
            r'experience?\s*:\s*(\d+(?:\.\d+)?)\+?\s*years?',
            r'(\d+(?:\.\d+)?)\s*years?\s*(of)?\s*(professional|work|industry)?\s*experience',
        ]

        max_years = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    years = float(match[0])
                    max_years = max(max_years, years)
                except (ValueError, IndexError):
                    continue

        if max_years > 0:
            return max_years
        return self._extract_years_from_date_ranges(text)

    def _infer_seniority(self, text: str, years: float) -> str:
        """Infer seniority from text and experience."""
        text_lower = text.lower()

        # Prefer keywords in role/title lines to avoid false positives.
        title_level = self._infer_seniority_from_titles(text_lower)
        if title_level:
            return title_level

        # Check for explicit seniority keywords, highest precedence first.
        for level in ["senior", "mid", "junior"]:
            for keyword in SENIORITY_KEYWORDS[level]:
                if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                    return level

        # Fallback to years-based inference
        if years >= 7:
            return "senior"
        elif years >= 3:
            return "mid"
        elif years > 0:
            return "junior"
        else:
            return "unknown"

    def _infer_seniority_from_titles(self, text: str) -> Optional[str]:
        """Infer seniority from lines that look like role titles."""
        title_lines = [
            line for line in text.splitlines()
            if any(hint in line for hint in ROLE_TITLE_HINTS)
        ]
        if not title_lines:
            return None

        found_levels = set()
        for line in title_lines:
            for level in ["senior", "mid", "junior"]:
                keywords = SENIORITY_KEYWORDS[level]
                pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"
                if re.search(pattern, line):
                    found_levels.add(level)

        if "senior" in found_levels:
            return "senior"
        if "mid" in found_levels:
            return "mid"
        if "junior" in found_levels:
            return "junior"
        return None

    def _extract_years_from_date_ranges(self, text: str) -> float:
        """Infer years of experience from date ranges in work history."""
        section_text = self._extract_experience_section(text)
        ranges = self._find_date_ranges(section_text)
        if not ranges and section_text != text:
            ranges = self._find_date_ranges(text)

        total_months = self._merge_month_ranges(ranges)
        if total_months <= 0:
            return 0.0
        return round(total_months / 12.0, 1)

    def _extract_experience_section(self, text: str) -> str:
        """Best-effort slice of the experience section."""
        section_start = re.search(
            r"\b(experience|work experience|professional experience|employment|work history)\b",
            text,
            re.IGNORECASE,
        )
        if not section_start:
            return text

        start_index = section_start.start()
        section_text = text[start_index:]

        section_end = re.search(
            r"\b(education|skills|projects|certifications|summary|profile|awards|publications|volunteer|interests)\b",
            section_text,
            re.IGNORECASE,
        )
        if section_end:
            return section_text[:section_end.start()]
        return section_text

    def _find_date_ranges(self, text: str) -> List[Tuple[int, int]]:
        """Find date ranges and return month index intervals."""
        date_token = (
            r"(?:[A-Za-z]{3,9}\s+\d{4}"
            r"|\d{4}[/-]\d{1,2}"
            r"|\d{1,2}[/-]\d{4}"
            r"|\d{4})"
        )
        dash_pattern = "-|\u2013|\u2014|to|through|until"
        range_pattern = re.compile(
            rf"(?P<start>{date_token})\s*(?:{dash_pattern})\s*"
            rf"(?P<end>{date_token}|present|current|now)",
            re.IGNORECASE,
        )

        now = datetime.utcnow()
        now_index = now.year * 12 + (now.month - 1)
        ranges = []
        for match in range_pattern.finditer(text):
            start_token = match.group("start")
            end_token = match.group("end")

            start = self._parse_date_token(start_token, is_end=False)
            if not start:
                continue
            if end_token.lower() in {"present", "current", "now"}:
                end_index = now_index
            else:
                end = self._parse_date_token(end_token, is_end=True)
                if not end:
                    continue
                end_index = end[0] * 12 + (end[1] - 1)

            start_index = start[0] * 12 + (start[1] - 1)
            if start_index > end_index:
                continue
            ranges.append((start_index, end_index))

        return ranges

    def _parse_date_token(self, token: str, is_end: bool) -> Optional[Tuple[int, int]]:
        """Parse a date token into (year, month)."""
        cleaned = re.sub(r"[.,]", "", token.strip().lower())

        # Month name + year
        match = re.match(r"([a-z]{3,9})\s+(\d{4})$", cleaned)
        if match:
            month_name = match.group(1)
            year = int(match.group(2))
            month = MONTHS.get(month_name)
            if month and self._is_valid_year(year):
                return year, month

        # MM/YYYY
        match = re.match(r"(\d{1,2})[/-](\d{4})$", cleaned)
        if match:
            month = int(match.group(1))
            year = int(match.group(2))
            if 1 <= month <= 12 and self._is_valid_year(year):
                return year, month

        # YYYY/MM
        match = re.match(r"(\d{4})[/-](\d{1,2})$", cleaned)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            if 1 <= month <= 12 and self._is_valid_year(year):
                return year, month

        # Year only
        match = re.match(r"(\d{4})$", cleaned)
        if match:
            year = int(match.group(1))
            if self._is_valid_year(year):
                month = 12 if is_end else 1
                return year, month

        return None

    def _is_valid_year(self, year: int) -> bool:
        """Check if year is in a plausible range."""
        current_year = datetime.utcnow().year
        return 1970 <= year <= current_year + 1

    def _merge_month_ranges(self, ranges: List[Tuple[int, int]]) -> int:
        """Merge overlapping month ranges and return total months."""
        if not ranges:
            return 0

        ranges.sort()
        merged = [ranges[0]]
        for start, end in ranges[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        total_months = 0
        for start, end in merged:
            total_months += end - start + 1
        return total_months

    def _extract_role_keywords(self, text: str) -> List[str]:
        """Extract role/title keywords from resume."""
        # Common software engineering roles
        role_patterns = [
            r'\b(software engineer|backend engineer|front-?end engineer|full-?stack engineer)\b',
            r'\b(data scientist|data engineer|machine learning engineer|ml engineer)\b',
            r'\b(devops engineer|site reliability engineer|sre)\b',
            r'\b(software developer|application developer|web developer)\b',
            r'\b(technical lead|engineering lead|staff engineer|principal engineer)\b',
            r'\b(backend developer|frontend developer|fullstack developer)\b',
        ]

        found_roles = []
        text_lower = text.lower()

        for pattern in role_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            found_roles.extend(matches)

        # Deduplicate while preserving order
        seen = set()
        unique_roles = []
        for role in found_roles:
            if role.lower() not in seen:
                seen.add(role.lower())
                unique_roles.append(role)

        return unique_roles[:5]  # Limit to top 5 most relevant
