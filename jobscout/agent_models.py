"""Data models for agent-based JobScout."""

from dataclasses import dataclass, field
from typing import Optional, List, Set
from datetime import datetime


@dataclass
class CVProfile:
    """Candidate profile extracted from CV."""
    # Core identity
    name: Optional[str] = None
    role_primary: str = ""  # backend, frontend, fullstack, data, devops, mobile
    seniority: str = "unknown"  # junior, mid, senior, unknown, lead, principal

    # Skills (normalized to lowercase)
    skills: Set[str] = field(default_factory=set)
    languages: Set[str] = field(default_factory=set)
    frameworks: Set[str] = field(default_factory=set)
    databases: Set[str] = field(default_factory=set)
    infrastructure: Set[str] = field(default_factory=set)

    # Experience
    years_experience: float = 0
    companies: List[str] = field(default_factory=list)

    # For job search
    search_keywords: List[str] = field(default_factory=list)

    # Preferences
    preferred_locations: List[str] = field(default_factory=list)
    target_companies: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "role_primary": self.role_primary,
            "seniority": self.seniority,
            "skills": sorted(self.skills),
            "languages": sorted(self.languages),
            "frameworks": sorted(self.frameworks),
            "databases": sorted(self.databases),
            "infrastructure": sorted(self.infrastructure),
            "years_experience": self.years_experience,
            "companies": self.companies,
            "search_keywords": self.search_keywords,
            "preferred_locations": self.preferred_locations,
            "target_companies": self.target_companies,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CVProfile":
        """Create from dictionary."""
        return cls(
            name=data.get("name"),
            role_primary=data.get("role_primary", ""),
            seniority=data.get("seniority", "unknown"),
            skills=set(data.get("skills", [])),
            languages=set(data.get("languages", [])),
            frameworks=set(data.get("frameworks", [])),
            databases=set(data.get("databases", [])),
            infrastructure=set(data.get("infrastructure", [])),
            years_experience=float(data.get("years_experience", 0)),
            companies=data.get("companies", []),
            search_keywords=data.get("search_keywords", []),
            preferred_locations=data.get("preferred_locations", []),
            target_companies=data.get("target_companies", []),
        )


@dataclass
class JobEvaluation:
    """Agent's evaluation of a job against a CV profile."""
    job_title: str = ""
    company: str = ""
    location: str = ""
    url: str = ""
    source: str = ""
    posted_date: Optional[str] = None
    salary: Optional[str] = None

    # Matching results
    match_score: float = 0  # 0-100
    is_match: bool = False

    # Alignment
    role_aligned: bool = True
    seniority_aligned: bool = True
    location_compatible: bool = True

    # Skills
    required_skills_matched: List[str] = field(default_factory=list)
    required_skills_missing: List[str] = field(default_factory=list)

    # Explanation
    summary: str = ""
    concerns: List[str] = field(default_factory=list)
    why_match: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_title": self.job_title,
            "company": self.company,
            "location": self.location,
            "url": self.url,
            "source": self.source,
            "posted_date": self.posted_date,
            "salary": self.salary,
            "match_score": round(self.match_score, 1),
            "is_match": self.is_match,
            "role_aligned": self.role_aligned,
            "seniority_aligned": self.seniority_aligned,
            "location_compatible": self.location_compatible,
            "required_skills_matched": self.required_skills_matched,
            "required_skills_missing": self.required_skills_missing,
            "summary": self.summary,
            "concerns": self.concerns,
            "why_match": self.why_match,
        }

    @classmethod
    def from_job_listing(cls, job, **kwargs) -> "JobEvaluation":
        """Create from JobListing with evaluation data."""
        return cls(
            job_title=job.title,
            company=job.company,
            location=job.location,
            url=job.apply_url,
            source=job.source,
            posted_date=job.posted_date.isoformat() if hasattr(job, 'posted_date') and job.posted_date else None,
            **kwargs
        )


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    # LLM settings
    llm_provider: str = "openai"
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None

    # Job search
    location: str = "remote"
    max_results: int = 7
    sources: List[str] = field(default_factory=lambda: ["company_boards", "remoteok"])

    # Company boards
    company_boards_include_all: bool = True
    company_boards_specific: List[str] = field(default_factory=list)


@dataclass
class JobSearchResult:
    """Result of a job search operation."""
    matches: List[JobEvaluation] = field(default_factory=list)
    filtered_count: int = 0
    total_evaluated: int = 0
    search_time_seconds: float = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "matches": [j.to_dict() for j in self.matches],
            "filtered_count": self.filtered_count,
            "total_evaluated": self.total_evaluated,
            "search_time_seconds": round(self.search_time_seconds, 2),
        }
