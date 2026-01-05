"""Resume parsing and skill extraction."""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Set, Dict
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
            r'(\d+)\+?\s*years?\s*(of)?\s*experience',
            r'experience?\s*:\s*(\d+)\+?\s*years?',
            r'(\d+)\s*years?\s*(of)?\s*(professional|work|industry)?\s*experience',
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

        return max_years

    def _infer_seniority(self, text: str, years: float) -> str:
        """Infer seniority from text and experience."""
        text_lower = text.lower()

        # Check for explicit seniority keywords
        for level, keywords in SENIORITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
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
