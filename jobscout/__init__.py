"""
JobScout: A conservative job-search assistant.

Agent-based system that uses LLM to analyze CVs and match jobs.
"""

__version__ = "2.0.0"

# Core agent modules (new in v2.0)
from jobscout.llm_client import LLMClient, get_default_client, LLMResponse
from jobscout.agent_models import CVProfile, JobEvaluation, AgentConfig, JobSearchResult
from jobscout.agent import JobScoutAgent
from jobscout.emailer_agent import AgentEmailer

# Job sources - import directly to avoid circular import issues
from jobscout.job_sources.base import JobListing, JobSource, strip_html_tags
from jobscout.job_sources.fetcher import JobFetcher, fetch_jobs_for_profile

# Define DEFAULT_SOURCES here to avoid import issues
DEFAULT_SOURCES = ["company_boards", "remoteok", "weworkremotely", "remotive"]

__all__ = [
    # Agent core
    "JobScoutAgent",
    "CVProfile",
    "JobEvaluation",
    "AgentConfig",
    "JobSearchResult",
    # LLM client
    "LLMClient",
    "get_default_client",
    "LLMResponse",
    # Email
    "AgentEmailer",
    # Job sources
    "JobListing",
    "JobSource",
    "strip_html_tags",
    "JobFetcher",
    "fetch_jobs_for_profile",
    "DEFAULT_SOURCES",
]
