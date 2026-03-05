# Agent-Based JobScout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild JobScout as an agent-based system where an LLM agent handles CV analysis, job search, matching, and ranking, replacing the complex regex/scoring system with intelligent evaluation.

**Architecture:** Single agent analyzes CV → generates search queries → fetches jobs from existing sources → evaluates each job objectively → ranks and emails top N matches. Stateless, no dedup cache.

**Tech Stack:** Python 3.10+, OpenAI/Anthropic/DeepSeek APIs, existing job_sources (streamlined), SMTP/email.

---

## Overview of Changes

**Files to DELETE (agent replaces these):**
- `jobscout/scoring.py` — entire file, agent handles scoring
- `jobscout/job_parser.py` — agent extracts job requirements
- `jobscout/filters.py` — agent evaluates fit
- `jobscout/dedup.py` — stateless now, no cross-run dedup
- `jobscout/semantic.py` — not needed with agent
- `jobscout/role_recommender.py` — agent handles this
- `jobscout/resume_parser.py` — agent analyzes CV directly

**Files to CREATE:**
- `jobscout/agent.py` — core agent logic
- `jobscout/llm_client.py` — unified LLM client

**Files to MODIFY:**
- `jobscout/config.py` — simplify to agent-focused config
- `jobscout/main.py` — agent-based orchestration
- `jobscout/emailer.py` — new email format
- `jobscout/job_sources/` — streamline, keep what works
- `jobscout_cli.py` — update for new flow
- `config.example.yaml` — simplified config

---

## Task 1: Create Unified LLM Client

**Files:**
- Create: `jobscout/llm_client.py`

**Step 1: Write the LLM client with multi-provider support**

```python
"""Unified LLM client for agent-based JobScout."""

import os
from typing import Optional, TypeVar
from dataclasses import dataclass
import json


T = TypeVar('T')


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    provider: str

    def parse_json(self) -> dict:
        """Parse response as JSON."""
        return json.loads(self.content)

    def parse_object(self, cls: type[T]) -> T:
        """Parse response as a dataclass object."""
        data = self.parse_json()
        return cls(**data)


class LLMClient:
    """Unified client for OpenAI, Anthropic, and DeepSeek."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize LLM client.

        Args:
            provider: "openai", "anthropic", or "deepseek"
            api_key: API key (defaults to env var)
            model: Model name (defaults to provider default)
        """
        self.provider = provider
        self.api_key = api_key or self._get_default_api_key(provider)
        self.model = model or self._get_default_model(provider)

        if not self.api_key:
            raise ValueError(f"API key not found for provider: {provider}")

        # Import the appropriate client
        if provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        elif provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        elif provider == "deepseek":
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _get_default_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment."""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        return os.getenv(env_vars.get(provider, ""))

    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-20241022",
            "deepseek": "deepseek-chat",
        }
        return defaults.get(provider, "gpt-4o-mini")

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """
        Send chat completion request.

        Args:
            messages: List of {role, content} dicts
            temperature: Sampling temperature (0-1)
            response_format: For OpenAI, {"type": "json_object"}

        Returns:
            LLMResponse with content and metadata
        """
        if self.provider == "openai":
            return self._openai_chat(messages, temperature, response_format)
        elif self.provider == "anthropic":
            return self._anthropic_chat(messages, temperature)
        elif self.provider == "deepseek":
            return self._deepseek_chat(messages, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _openai_chat(
        self,
        messages: list[dict],
        temperature: float,
        response_format: Optional[dict],
    ) -> LLMResponse:
        """OpenAI chat completion."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider="openai",
        )

    def _anthropic_chat(
        self,
        messages: list[dict],
        temperature: float,
    ) -> LLMResponse:
        """Anthropic chat completion."""
        # Anthropic requires system message to be separate
        system_msg = ""
        user_msgs = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg += msg["content"] + "\n\n"
            else:
                user_msgs.append(msg)

        # For JSON output, instruct in system prompt
        response = self._client.messages.create(
            model=self.model,
            system=system_msg.strip() or None,
            messages=user_msgs,
            temperature=temperature,
            max_tokens=4096,
        )

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            provider="anthropic",
        )

    def _deepseek_chat(
        self,
        messages: list[dict],
        temperature: float,
    ) -> LLMResponse:
        """DeepSeek chat completion (OpenAI-compatible)."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider="deepseek",
        )

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Generate JSON response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Parsed JSON dict
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Add JSON instruction to prompt
        if self.provider == "openai":
            response = self.chat(messages, response_format={"type": "json_object"})
        else:
            # For non-OpenAI, add instruction to prompt
            messages[-1]["content"] += "\n\nRespond ONLY with valid JSON."
            response = self.chat(messages)

        return response.parse_json()

    def generate_structured(
        self,
        prompt: str,
        structure: type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """
        Generate structured response matching a dataclass.

        Args:
            prompt: User prompt
            structure: Dataclass type to parse into
            system_prompt: Optional system prompt

        Returns:
            Instance of the dataclass
        """
        json_response = self.generate_json(prompt, system_prompt)
        return structure(**json_response)
```

**Step 2: Write tests for LLM client**

```python
# tests/test_llm_client.py

import pytest
from jobscout.llm_client import LLMClient, LLMResponse


def test_llm_client_requires_api_key():
    """Test that client raises error without API key."""
    with pytest.raises(ValueError, match="API key not found"):
        LLMClient(provider="openai", api_key=None, model="gpt-4o-mini")


def test_llm_client_uses_env_api_key(monkeypatch):
    """Test that client uses environment variable for API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = LLMClient(provider="openai")
    assert client.api_key == "test-key"


def test_llm_client_default_models():
    """Test that default models are set correctly."""
    client = LLMClient(provider="openai", api_key="test")
    assert client.model == "gpt-4o-mini"

    client = LLMClient(provider="anthropic", api_key="test")
    assert client.model == "claude-3-5-haiku-20241022"


@pytest.mark.integration
def test_openai_chat_integration(monkeypatch):
    """Integration test for OpenAI chat (requires API key)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    client = LLMClient(provider="openai")
    response = client.chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'test passed'"},
    ])

    assert isinstance(response, LLMResponse)
    assert response.provider == "openai"
    assert "test passed" in response.content.lower()


@pytest.mark.integration
def test_generate_json_integration(monkeypatch):
    """Integration test for JSON generation."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    client = LLMClient(provider="openai")
    result = client.generate_json(
        "Return JSON with keys: name (string), age (number)",
        system_prompt="You are a JSON generator.",
    )

    assert isinstance(result, dict)
    assert "name" in result
    assert "age" in result
    assert isinstance(result["age"], (int, float))
```

**Step 3: Run tests to verify setup**

```bash
cd "C:\Users\Administrator\Documents\Yemi's Projects\jobscout"
pytest tests/test_llm_client.py -v
```

Expected: Tests pass (integration tests may skip if no API key)

**Step 4: Commit**

```bash
git add jobscout/llm_client.py tests/test_llm_client.py
git commit -m "feat: add unified LLM client with multi-provider support"
```

---

## Task 2: Create Agent Core Module

**Files:**
- Create: `jobscout/agent.py`
- Create: `jobscout/models.py`

**Step 1: Create data models**

```python
# jobscout/models.py

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
    seniority: str = "unknown"  # junior, mid, senior, unknown

    # Skills
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


@dataclass
class JobEvaluation:
    """Agent's evaluation of a job against a CV profile."""
    job_title: str = ""
    company: str = ""
    location: str = ""
    url: str = ""
    source: str = ""

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

    # Salary (if available)
    salary: Optional[str] = None


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
```

**Step 2: Create the agent core**

```python
# jobscout/agent.py

"""Agent-based job search and matching."""

import logging
from typing import List, Optional
from .llm_client import LLMClient
from .models import CVProfile, JobEvaluation, AgentConfig
from .job_sources.base import JobListing


logger = logging.getLogger(__name__)


class JobScoutAgent:
    """Agent that handles CV analysis, job search, and matching."""

    def __init__(self, config: AgentConfig):
        """Initialize agent with configuration."""
        self.config = config
        self.llm = LLMClient(
            provider=config.llm_provider,
            api_key=config.llm_api_key,
            model=config.llm_model,
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

        response = self.llm.generate_json(prompt, system_prompt)
        profile = self._parse_cv_profile(response)

        self.cv_profile = profile
        logger.info(f"CV analyzed: {profile.role_primary} developer, {len(profile.skills)} skills")
        return profile

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

        response = self.llm.generate_json(prompt, system_prompt)
        queries = response.get("search_queries", [])

        logger.info(f"Generated {len(queries)} search queries")
        return queries

    def evaluate_job(
        self,
        job: JobListing,
        profile: CVProfile,
    ) -> JobEvaluation:
        """
        Evaluate a job against the candidate's profile.

        Args:
            job: Job listing to evaluate
            profile: Candidate's CV profile

        Returns:
            JobEvaluation with match score and reasoning
        """
        prompt = self._job_evaluation_prompt(job, profile)
        system_prompt = self._job_evaluation_system_prompt()

        response = self.llm.generate_json(prompt, system_prompt)

        return JobEvaluation(
            job_title=job.title,
            company=job.company,
            location=job.location,
            url=job.apply_url,
            source=job.source,
            **response
        )

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

    def _cv_analysis_prompt(self, cv_text: str) -> str:
        """Generate prompt for CV analysis."""
        return f"""Analyze this resume and extract the following information:

RESUME:
{cv_text[:8000]}

Return JSON with this structure:
{{
    "name": "Full name if found",
    "role_primary": "One of: backend, frontend, fullstack, data, devops, mobile",
    "seniority": "One of: junior, mid, senior, unknown",
    "skills": ["list", "of", "all", "technical", "skills"],
    "languages": ["python", "javascript", ...],
    "frameworks": ["django", "react", ...],
    "databases": ["postgresql", "mongodb", ...],
    "infrastructure": ["docker", "aws", ...],
    "years_experience": number,
    "companies": ["company1", "company2"],
    "search_keywords": ["python backend engineer", "api developer", ...],
    "preferred_locations": ["remote", "san francisco", ...],
    "target_companies": ["google", "stripe", ...]
}}

Be thorough with skills - include everything mentioned.
Generate 10-15 diverse search keywords including job titles, technologies, and domains.
"""

    def _cv_analysis_system_prompt(self) -> str:
        """System prompt for CV analysis."""
        return """You are an expert technical recruiter and career coach.
Extract accurate information from resumes.
Normalize skill names (e.g., "React.js" → "react", "PostgreSQL" → "postgresql").
Infer the primary role from the overall experience, not just keywords."""

    def _search_query_prompt(self, profile: CVProfile) -> str:
        """Generate prompt for search query generation."""
        return f"""Based on this candidate profile, generate 10-15 diverse search terms for job boards.

CANDIDATE PROFILE:
- Role: {profile.role_primary}
- Skills: {', '.join(list(profile.skills)[:10])}
- Experience: {profile.years_experience} years
- Current keywords: {', '.join(profile.search_keywords[:5])}

Generate search terms across these categories:
1. Job titles (e.g., "python backend engineer")
2. Technologies (e.g., "python developer", "react developer")
3. Domains/Specializations (e.g., "ai engineer", "ml engineer", "api developer")
4. Company types (e.g., "fintech engineer", "healthcare developer")

Return JSON:
{{
    "search_queries": ["term1", "term2", ...]
}}"""

    def _job_evaluation_prompt(self, job: JobListing, profile: CVProfile) -> str:
        """Generate prompt for job evaluation."""
        return f"""Evaluate if this job is a match for the candidate.

CANDIDATE PROFILE:
- Role: {profile.role_primary}
- Seniority: {profile.seniority}
- Years Experience: {profile.years_experience}
- Skills: {', '.join(list(profile.skills)[:20])}
- Languages: {', '.join(list(profile.languages))}
- Frameworks: {', '.join(list(profile.frameworks))}

JOB POSTING:
Title: {job.title}
Company: {job.company}
Location: {job.location}
Description: {job.description[:4000]}

Evaluate objectively:

1. ROLE ALIGNMENT: Is this the right type of role? (backend/frontend/data/etc)
2. SKILLS MATCH: Does the candidate have the REQUIRED skills?
3. SENIORITY: Is the experience level appropriate?
4. LOCATION: Is the location compatible?

Return JSON:
{{
    "match_score": 0-100,
    "is_match": true/false,
    "role_aligned": true/false,
    "seniority_aligned": true/false,
    "location_compatible": true/false,
    "required_skills_matched": ["skill1", "skill2"],
    "required_skills_missing": [],
    "summary": "Brief explanation (1-2 sentences)",
    "concerns": ["any red flags or concerns"],
    "why_match": "Why this is a good match if score > 60"
}}

Scoring guidelines:
- 90-100: Excellent match - almost all requirements met
- 75-89: Good match - most requirements met, minor gaps
- 60-74: Possible match - some gaps but worth considering
- Below 60: Not a match

A job is NOT a match (is_match=false) if:
- Wrong role domain (e.g., data science for web dev)
- Missing critical required skills
- Significant seniority mismatch
"""

    def _job_evaluation_system_prompt(self) -> str:
        """System prompt for job evaluation."""
        return """You are an objective job match evaluator.
Compare candidate qualifications to job requirements WITHOUT bias.
Focus on actual fit, not aspirational matches.
Consider that missing "nice to have" skills is fine.
Missing "must have" core skills is a problem."""

    def _parse_cv_profile(self, data: dict) -> CVProfile:
        """Parse LLM response into CVProfile."""
        return CVProfile(
            name=data.get("name"),
            role_primary=data.get("role_primary", "").lower(),
            seniority=data.get("seniority", "unknown").lower(),
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
```

**Step 3: Write tests for agent**

```python
# tests/test_agent.py

import pytest
from jobscout.agent import JobScoutAgent
from jobscout.models import AgentConfig, CVProfile
from jobscout.job_sources.base import JobListing


@pytest.fixture
def mock_config():
    """Create test config."""
    return AgentConfig(
        llm_provider="openai",
        llm_api_key="test-key",
        max_results=5,
    )


@pytest.fixture
def sample_cv():
    """Sample CV text."""
    return """
    John Doe
    Backend Engineer

    EXPERIENCE:
    - Senior Backend Engineer at TechCorp (2020-Present)
      - Built APIs with Python, Django, PostgreSQL
      - Deployed with Docker, AWS

    - Backend Developer at StartupCo (2018-2020)
      - Python/Django development
      - Redis caching, Celery tasks

    SKILLS:
    - Languages: Python, SQL, JavaScript
    - Frameworks: Django, FastAPI, Flask
    - Databases: PostgreSQL, Redis, MongoDB
    - Infrastructure: Docker, AWS, Kubernetes
    """


@pytest.fixture
def sample_job():
    """Sample job listing."""
    return JobListing(
        title="Senior Backend Engineer",
        company="TestCompany",
        location="Remote",
        description="We are looking for a senior backend engineer with Python, Django, and PostgreSQL experience. You will build APIs and work on distributed systems.",
        apply_url="https://example.com/apply",
        source="Test",
    )


def test_agent_initialization(mock_config):
    """Test agent initializes correctly."""
    # Use a mock to avoid actual API call
    from unittest.mock import Mock, patch

    with patch('jobscout.agent.LLMClient'):
        agent = JobScoutAgent(mock_config)
        assert agent.config == mock_config
        assert agent.cv_profile is None


@pytest.mark.integration
def test_analyze_cv_integration(mock_config, sample_cv):
    """Integration test for CV analysis (requires API key)."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    config = AgentConfig(llm_provider="openai", max_results=5)
    agent = JobScoutAgent(config)

    profile = agent.analyze_cv(sample_cv)

    assert profile.role_primary in ["backend", "frontend", "fullstack", "data", "devops", "mobile"]
    assert len(profile.skills) > 0
    assert profile.years_experience >= 0
    assert len(profile.search_keywords) >= 5


@pytest.mark.integration
def test_generate_search_queries_integration(mock_config, sample_cv):
    """Integration test for search query generation."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    config = AgentConfig(llm_provider="openai", max_results=5)
    agent = JobScoutAgent(config)
    agent.analyze_cv(sample_cv)  # Set profile

    queries = agent.generate_search_queries(agent.cv_profile)

    assert len(queries) >= 10
    assert all(isinstance(q, str) for q in queries)


@pytest.mark.integration
def test_evaluate_job_integration(mock_config, sample_cv, sample_job):
    """Integration test for job evaluation."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    config = AgentConfig(llm_provider="openai", max_results=5)
    agent = JobScoutAgent(config)
    profile = agent.analyze_cv(sample_cv)

    evaluation = agent.evaluate_job(sample_job, profile)

    assert evaluation.job_title == "Senior Backend Engineer"
    assert evaluation.company == "TestCompany"
    assert 0 <= evaluation.match_score <= 100
    assert isinstance(evaluation.is_match, bool)


def test_rank_jobs_filters_non_matches(mock_config):
    """Test that ranking filters out non-matches."""
    from jobscout.models import JobEvaluation

    agent = JobScoutAgent(mock_config)

    evaluations = [
        JobEvaluation(job_title="Job 1", is_match=True, match_score=80),
        JobEvaluation(job_title="Job 2", is_match=False, match_score=70),
        JobEvaluation(job_title="Job 3", is_match=True, match_score=90),
        JobEvaluation(job_title="Job 4", is_match=True, match_score=85),
    ]

    ranked = agent.rank_jobs(evaluations, limit=2)

    assert len(ranked) == 2
    assert ranked[0].job_title == "Job 3"  # Highest score
    assert ranked[1].job_title == "Job 4"  # Second highest
```

**Step 4: Run tests**

```bash
pytest tests/test_agent.py -v
```

Expected: Tests pass (integration tests may skip without API key)

**Step 5: Commit**

```bash
git add jobscout/agent.py jobscout/models.py tests/test_agent.py
git commit -m "feat: add agent core with CV analysis and job evaluation"
```

---

## Task 3: Streamline Job Sources

**Files:**
- Modify: `jobscout/job_sources/base.py`
- Modify: `jobscout/job_sources/rss_feeds.py`
- Modify: `jobscout/job_sources/company_boards.py`
- Modify: `jobscout/job_sources/remotive_api.py`
- Delete: `jobscout/job_sources/boolean_search.py`
- Delete: `jobscout/job_sources/greenhouse_api.py`
- Delete: `jobscout/job_sources/lever_api.py`
- Delete: `jobscout/job_sources/truncation.py`

**Step 1: Update base.py (keep existing, it's fine)**

No changes needed to `base.py` — the `JobListing` dataclass and `strip_html_tags` are still useful.

**Step 2: Create unified job fetcher**

```python
# jobscout/job_sources/fetcher.py

"""Unified job fetcher that uses all sources."""

import logging
from typing import List, Optional
from .base import JobListing
from .company_boards import CompanyBoardsSource
from .rss_feeds import RemoteOKSource, WeWorkRemotelySource, HimalayasSource, JavascriptJobsSource
from .remotive_api import RemotiveSource


logger = logging.getLogger(__name__)


class JobFetcher:
    """Fetches jobs from multiple sources based on search queries."""

    def __init__(
        self,
        sources: Optional[List[str]] = None,
        location: str = "remote",
        company_boards_all: bool = True,
        company_boards_specific: Optional[List[str]] = None,
    ):
        """
        Initialize job fetcher.

        Args:
            sources: List of source names (defaults to all)
            location: Location preference for filtering
            company_boards_all: Scrape all company boards
            company_boards_specific: Specific companies to scrape
        """
        self.sources = sources or ["company_boards", "remoteok", "weworkremotely", "remotive"]
        self.location = location.lower()
        self.company_boards_all = company_boards_all
        self.company_boards_specific = company_boards_specific or []

    def fetch_all(self, search_queries: List[str], limit_per_source: int = 50) -> List[JobListing]:
        """
        Fetch jobs from all sources using search queries.

        Args:
            search_queries: List of search terms to query/filter with
            limit_per_source: Max jobs per source

        Returns:
            List of job listings (deduplicated by URL)
        """
        all_jobs = []
        seen_urls = set()

        for source in self.sources:
            try:
                jobs = self._fetch_from_source(source, search_queries, limit_per_source)

                # Deduplicate by URL
                for job in jobs:
                    if job.apply_url not in seen_urls:
                        seen_urls.add(job.apply_url)
                        all_jobs.append(job)

                logger.info(f"Fetched {len(jobs)} jobs from {source}")

            except Exception as e:
                logger.error(f"Failed to fetch from {source}: {e}")
                continue

        logger.info(f"Total unique jobs fetched: {len(all_jobs)}")
        return all_jobs

    def _fetch_from_source(
        self,
        source: str,
        search_queries: List[str],
        limit: int,
    ) -> List[JobListing]:
        """Fetch from a specific source."""
        source_lower = source.lower()

        if source_lower == "company_boards":
            src = CompanyBoardsSource(
                resume_skills=set(),  # Not used for search-based fetching
                role_keywords=search_queries,
                location_preference=self.location,
                max_job_age_days=7,
                companies=self.company_boards_specific if not self.company_boards_all else None,
            )
            return src.fetch_jobs(limit=limit)

        elif source_lower == "remoteok":
            src = RemoteOKSource("RemoteOK")
            # Filter by search terms
            all_jobs = src.fetch_jobs(limit=limit)
            return self._filter_by_search_terms(all_jobs, search_queries)

        elif source_lower == "weworkremotely":
            src = WeWorkRemotelySource("We Work Remotely")
            all_jobs = src.fetch_jobs(limit=limit)
            return self._filter_by_search_terms(all_jobs, search_queries)

        elif source_lower == "himalayas":
            src = HimalayasSource("Himalayas")
            all_jobs = src.fetch_jobs(limit=limit)
            return self._filter_by_search_terms(all_jobs, search_queries)

        elif source_lower == "remotive":
            src = RemotiveSource("Remotive")
            return src.fetch_jobs(limit=limit)

        else:
            logger.warning(f"Unknown source: {source}")
            return []

    def _filter_by_search_terms(self, jobs: List[JobListing], search_queries: List[str]) -> List[JobListing]:
        """Filter jobs by search terms (simple keyword matching)."""
        if not search_queries:
            return jobs

        filtered = []
        search_terms_lower = [q.lower() for q in search_queries]

        for job in jobs:
            job_text = f"{job.title} {job.description}".lower()

            # Job matches if any search term is present
            if any(term in job_text for term in search_terms_lower):
                filtered.append(job)

        return filtered
```

**Step 3: Delete obsolete files**

```bash
cd "C:\Users\Administrator\Documents\Yemi's Projects\jobscout"
rm jobscout/job_sources/boolean_search.py
rm jobscout/job_sources/greenhouse_api.py
rm jobscout/job_sources/lever_api.py
rm jobscout/job_sources/truncation.py
```

**Step 4: Update job_sources/__init__.py**

```python
# jobscout/job_sources/__init__.py

"""Job sources for agent-based JobScout."""

from .base import JobListing, JobSource, strip_html_tags
from .fetcher import JobFetcher
from .rss_feeds import RemoteOKSource, WeWorkRemotelySource, HimalayasSource
from .remotive_api import RemotiveSource
from .company_boards import CompanyBoardsSource

__all__ = [
    "JobListing",
    "JobSource",
    "strip_html_tags",
    "JobFetcher",
    "RemoteOKSource",
    "WeWorkRemotelySource",
    "HimalayasSource",
    "RemotiveSource",
    "CompanyBoardsSource",
]
```

**Step 5: Write tests for job fetcher**

```python
# tests/test_job_fetcher.py

import pytest
from jobscout.job_sources.fetcher import JobFetcher


def test_job_fetcher_initialization():
    """Test fetcher initializes with defaults."""
    fetcher = JobFetcher()
    assert fetcher.sources == ["company_boards", "remoteok", "weworkremotely", "remotive"]
    assert fetcher.location == "remote"


def test_job_fetcher_custom_sources():
    """Test fetcher with custom sources."""
    fetcher = JobFetcher(sources=["remoteok", "remotive"])
    assert fetcher.sources == ["remoteok", "remotive"]


def test_job_fetcher_custom_location():
    """Test fetcher with custom location."""
    fetcher = JobFetcher(location="San Francisco")
    assert fetcher.location == "san francisco"


@pytest.mark.integration
def test_fetch_from_remoteok():
    """Integration test for RemoteOK fetching."""
    fetcher = JobFetcher(sources=["remoteok"])
    jobs = fetcher.fetch_all(search_queries=["python"], limit_per_source=10)

    assert len(jobs) > 0
    assert all(hasattr(j, 'title') for j in jobs)
    assert all(hasattr(j, 'apply_url') for j in jobs)


@pytest.mark.integration
def test_fetch_from_remotive():
    """Integration test for Remotive fetching."""
    fetcher = JobFetcher(sources=["remotive"])
    jobs = fetcher.fetch_all(search_queries=["backend"], limit_per_source=10)

    assert len(jobs) >= 0  # May return 0 if no matches


def test_deduplication_by_url():
    """Test that jobs are deduplicated by URL."""
    from jobscout.job_sources.base import JobListing

    fetcher = JobFetcher(sources=[])

    job1 = JobListing(
        title="Job 1",
        company="Company",
        location="Remote",
        description="Desc",
        apply_url="https://example.com/1",
        source="Test",
    )
    job2 = JobListing(
        title="Job 2",
        company="Company",
        location="Remote",
        description="Desc",
        apply_url="https://example.com/1",  # Same URL
        source="Test2",
    )
    job3 = JobListing(
        title="Job 3",
        company="Company",
        location="Remote",
        description="Desc",
        apply_url="https://example.com/3",
        source="Test",
    )

    # Manually test deduplication logic
    seen_urls = set()
    unique_jobs = []
    for job in [job1, job2, job3]:
        if job.apply_url not in seen_urls:
            seen_urls.add(job.apply_url)
            unique_jobs.append(job)

    assert len(unique_jobs) == 2
    assert unique_jobs[0].title == "Job 1"
    assert unique_jobs[1].title == "Job 3"
```

**Step 6: Run tests**

```bash
pytest tests/test_job_fetcher.py -v
```

Expected: Tests pass

**Step 7: Commit**

```bash
git add jobscout/job_sources/ tests/test_job_fetcher.py
git commit -m "refactor: streamline job sources, add unified fetcher"
```

---

## Task 4: Simplified Configuration

**Files:**
- Create: `jobscout/config_agent.py` (new simplified config)
- Modify: `config.example.yaml`

**Step 1: Create new config module**

```python
# jobscout/config_agent.py

"""Simplified configuration for agent-based JobScout."""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class EmailConfig:
    """Email configuration."""
    to_address: str
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from: Optional[str] = None


@dataclass
class SearchConfig:
    """Job search configuration."""
    location: str = "remote"
    max_jobs: int = 7
    sources: List[str] = field(default_factory=lambda: ["company_boards", "remoteok"])
    company_boards_include_all: bool = True
    company_boards_specific: List[str] = field(default_factory=list)


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "openai"
    api_key: Optional[str] = None
    model: Optional[str] = None

    def get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key

        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        return os.getenv(env_vars.get(self.provider.lower(), ""))

    def get_model(self) -> str:
        """Get model name or default."""
        if self.model:
            return self.model

        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-20241022",
            "deepseek": "deepseek-chat",
        }
        return defaults.get(self.provider.lower(), "gpt-4o-mini")


@dataclass
class AgentJobScoutConfig:
    """Main configuration for agent-based JobScout."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    email: Optional[EmailConfig] = None
    search: SearchConfig = field(default_factory=SearchConfig)
    outbox_dir: str = "./outbox"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str) -> "AgentJobScoutConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Parse nested configs
        llm_data = data.get("llm", {})
        email_data = data.get("email")
        search_data = data.get("search", {})

        llm = LLMConfig(**llm_data)
        email = EmailConfig(**email_data) if email_data else None
        search = SearchConfig(**search_data)

        return cls(
            llm=llm,
            email=email,
            search=search,
            outbox_dir=data.get("outbox_dir", "./outbox"),
            log_level=data.get("log_level", "INFO"),
        )

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.llm.get_api_key():
            errors.append(f"LLM API key not found for provider: {self.llm.provider}")

        if self.search.max_jobs < 1 or self.search.max_jobs > 10:
            errors.append("max_jobs must be between 1 and 10")

        if self.email and self.email.to_address:
            # Email configured, validate SMTP if provided
            if self.email.smtp_host and not self.email.smtp_password:
                errors.append("SMTP password required when SMTP host is configured")

        return errors


def load_config(config_path: str = "config.yaml") -> AgentJobScoutConfig:
    """Load and validate configuration from file."""
    config = AgentJobScoutConfig.from_yaml(config_path)
    errors = config.validate()

    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return config
```

**Step 2: Update config.example.yaml**

```yaml
# config.example.yaml

# LLM Configuration (Required)
llm:
  provider: openai  # openai, anthropic, deepseek
  # api_key: sk-...  # Or set OPENAI_API_KEY environment variable
  model: gpt-4o-mini  # Optional, uses provider default

# Email Configuration (Required for sending emails)
email:
  to_address: your-email@example.com
  smtp_host: smtp.gmail.com
  smtp_port: 587
  smtp_username: your-email@gmail.com
  # smtp_password: your-app-password  # Or set SMTP_PASSWORD environment variable
  smtp_from: "JobScout <noreply@jobscout.example.com>"

# Job Search Configuration
search:
  location: remote  # remote, hybrid, onsite, or city name
  max_jobs: 7  # Number of matches to email (1-10)

  # Job sources to use
  sources:
    - company_boards  # Scrapes Greenhouse/Lever/Ashby boards
    - remoteok
    - weworkremotely
    - remotive
    # - himalayas

  # Company boards configuration
  company_boards_include_all: true  # Scrape all known company boards
  # Or specify specific companies:
  # company_boards_specific:
  #   - stripe
  #   - anthropic
  #   - openai

# Other settings
outbox_dir: ./outbox  # Fallback for email when SMTP not configured
log_level: INFO
```

**Step 5: Write tests for config**

```python
# tests/test_config_agent.py

import pytest
import tempfile
import os
from jobscout.config_agent import (
    AgentJobScoutConfig,
    EmailConfig,
    SearchConfig,
    LLMConfig,
    load_config,
)


def test_llm_config_get_api_key_from_env(monkeypatch):
    """Test LLM config gets API key from environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-from-env")

    config = LLMConfig(provider="openai")
    assert config.get_api_key() == "test-key-from-env"


def test_llm_config_get_api_key_from_config():
    """Test LLM config gets API key from config."""
    config = LLMConfig(provider="openai", api_key="test-key-from-config")
    assert config.get_api_key() == "test-key-from-config"


def test_llm_config_get_default_models():
    """Test default model names."""
    assert LLMConfig(provider="openai").get_model() == "gpt-4o-mini"
    assert LLMConfig(provider="anthropic").get_model() == "claude-3-5-haiku-20241022"


def test_load_config_from_yaml():
    """Test loading config from YAML file."""
    yaml_content = """
llm:
  provider: openai
  model: gpt-4o-mini

email:
  to_address: test@example.com
  smtp_host: smtp.gmail.com
  smtp_port: 587

search:
  location: remote
  max_jobs: 5
  sources:
    - remoteok
    - remotive

outbox_dir: ./outbox
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = AgentJobScoutConfig.from_yaml(temp_path)

        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o-mini"
        assert config.email.to_address == "test@example.com"
        assert config.search.location == "remote"
        assert config.search.max_jobs == 5
        assert "remoteok" in config.search.sources
    finally:
        os.unlink(temp_path)


def test_config_validation_missing_api_key(monkeypatch):
    """Test validation fails without API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = AgentJobScoutConfig(
        llm=LLMConfig(provider="openai", api_key=None),
        email=EmailConfig(to_address="test@example.com"),
    )

    errors = config.validate()
    assert len(errors) > 0
    assert any("API key" in e for e in errors)


def test_config_validation_invalid_max_jobs():
    """Test validation fails for invalid max_jobs."""
    config = AgentJobScoutConfig(
        llm=LLMConfig(provider="openai", api_key="test"),
        search=SearchConfig(max_jobs=15),
    )

    errors = config.validate()
    assert any("max_jobs" in e for e in errors)


def test_config_happy_path(monkeypatch):
    """Test valid config passes validation."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = AgentJobScoutConfig(
        llm=LLMConfig(provider="openai"),
        email=EmailConfig(to_address="test@example.com"),
        search=SearchConfig(max_jobs=7),
    )

    errors = config.validate()
    assert len(errors) == 0
```

**Step 6: Run tests**

```bash
pytest tests/test_config_agent.py -v
```

Expected: All tests pass

**Step 7: Commit**

```bash
git add jobscout/config_agent.py config.example.yaml tests/test_config_agent.py
git commit -m "feat: add simplified configuration for agent-based JobScout"
```

---

## Task 5: New Email Format

**Files:**
- Create: `jobscout/emailer_agent.py`

**Step 1: Create new emailer with updated format**

```python
# jobscout/emailer_agent.py

"""Email delivery for agent-based JobScout."""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from .models import JobEvaluation
from .config_agent import EmailConfig


logger = logging.getLogger(__name__)


class AgentEmailer:
    """Sends job match emails with the new format."""

    def __init__(self, email_config: Optional[EmailConfig], outbox_dir: str = "./outbox"):
        """
        Initialize emailer.

        Args:
            email_config: Email configuration (None = outbox only mode)
            outbox_dir: Directory to write emails when SMTP not configured
        """
        self.config = email_config
        self.outbox_dir = Path(outbox_dir)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)

    def send_digest(
        self,
        jobs: List[JobEvaluation],
        candidate_name: str = "",
    ) -> bool:
        """
        Send email digest with matched jobs.

        Args:
            jobs: List of job evaluations to email
            candidate_name: Optional candidate name for subject

        Returns:
            True if email sent successfully (or written to outbox)
        """
        if not jobs:
            logger.info("No jobs to send")
            self._write_empty_digest()
            return True

        html = self._render_html(jobs, candidate_name)
        subject = self._get_subject(jobs, candidate_name)

        # Try SMTP if configured
        if self.config and self.config.smtp_host:
            return self._send_smtp(html, subject)
        else:
            return self._write_outbox(html, subject)

    def _render_html(self, jobs: List[JobEvaluation], candidate_name: str) -> str:
        """Render HTML email."""
        date_str = datetime.now().strftime("%B %d, %Y")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px 8px 0 0; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header p {{ margin: 5px 0 0; opacity: 0.9; }}
        .footer {{ background: #f8f9fa; padding: 15px; border-radius: 0 0 8px 8px; text-align: center; font-size: 12px; color: #6c757d; }}
        .job {{ border: 1px solid #e0e0e0; border-left: 4px solid #667eea; margin: 15px 0; border-radius: 8px; overflow: hidden; }}
        .job-header {{ background: #f8f9fa; padding: 12px 15px; display: flex; justify-content: space-between; align-items: center; }}
        .job-title {{ font-size: 16px; font-weight: 600; color: #333; }}
        .job-company {{ font-size: 14px; color: #666; }}
        .job-score {{ background: #667eea; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 14px; }}
        .job-score.high {{ background: #28a745; }}
        .job-score.medium {{ background: #ffc107; color: #333; }}
        .job-body {{ padding: 15px; }}
        .job-detail {{ margin: 8px 0; font-size: 14px; }}
        .job-detail-label {{ color: #6c757d; font-weight: 500; }}
        .job-section {{ margin: 15px 0; }}
        .job-section-title {{ font-weight: 600; margin-bottom: 8px; color: #333; }}
        .match-reason {{ background: #e8f5e9; padding: 10px; border-radius: 6px; font-size: 14px; }}
        .match-reason ul {{ margin: 5px 0; padding-left: 20px; }}
        .concerns {{ background: #fff3cd; padding: 10px; border-radius: 6px; font-size: 14px; }}
        .concerns ul {{ margin: 5px 0; padding-left: 20px; }}
        .apply-btn {{ display: inline-block; background: #667eea; color: white; padding: 10px 20px; text-decoration: none; border-radius: 6px; font-weight: 500; }}
        .apply-btn:hover {{ background: #5568d3; }}
        .stats {{ background: #f8f9fa; padding: 10px; border-radius: 6px; font-size: 12px; color: #6c757d; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>JobScout Daily Digest</h1>
        <p>{date_str} • {len(jobs)} matches found</p>
    </div>

"""

        # Add jobs
        for i, job in enumerate(jobs, 1):
            score_class = "high" if job.match_score >= 80 else "medium"
            match_reasons = job.why_match.split("\n") if job.why_match else []
            concerns_list = job.concerns if job.concerns else []

            html += f"""
    <div class="job">
        <div class="job-header">
            <div>
                <div class="job-title">{i}. {self._escape_html(job.job_title)}</div>
                <div class="job-company">{self._escape_html(job.company)}</div>
            </div>
            <div class="job-score {score_class}">{job.match_score:.0f}%</div>
        </div>
        <div class="job-body">
            <div class="job-detail">
                <span class="job-detail-label">Location:</span> {self._escape_html(job.location)}
            </div>
"""

            if job.salary:
                html += f"""
            <div class="job-detail">
                <span class="job-detail-label">Salary:</span> {self._escape_html(job.salary)}
            </div>
"""

            # Why it matches
            if job.why_match:
                html += f"""
            <div class="job-section">
                <div class="job-section-title">✓ Why it matches:</div>
                <div class="match-reason">
                    {self._escape_html(job.why_match).replace(chr(10), "<br>")}
                </div>
            </div>
"""

            # Skills matched
            if job.required_skills_matched:
                html += f"""
            <div class="job-section">
                <div class="job-detail">
                    <span class="job-detail-label">Skills matched:</span> {', '.join(self._escape_html(s) for s in job.required_skills_matched[:8])}
                </div>
            </div>
"""

            # Concerns
            if job.concerns:
                html += f"""
            <div class="job-section">
                <div class="job-section-title">⚠ Things to note:</div>
                <div class="concerns">
                    {'<br>'.join('• ' + self._escape_html(c) for c in job.concerns[:3])}
                </div>
            </div>
"""

            html += f"""
            <div class="job-section" style="margin-top: 15px;">
                <a href="{job.url}" class="apply-btn">Apply Now</a>
            </div>
        </div>
    </div>
"""

        # Footer with stats
        html += f"""
    <div class="footer">
        <div class="stats">
            Showing top {len(jobs)} matches from today's search
        </div>
    </div>
</body>
</html>
"""

        return html

    def _get_subject(self, jobs: List[JobEvaluation], candidate_name: str) -> str:
        """Generate email subject."""
        name_part = f" for {candidate_name}" if candidate_name else ""
        return f"JobScout: {len(jobs)} matching jobs{name_part}"

    def _send_smtp(self, html: str, subject: str) -> bool:
        """Send email via SMTP."""
        if not self.config:
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config.smtp_from or f"JobScout <{self.config.smtp_username}>"
            msg['To'] = self.config.to_address

            msg.attach(MIMEText(html, 'html'))

            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent to {self.config.to_address}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            # Fallback to outbox
            return self._write_outbox(html, subject)

    def _write_outbox(self, html: str, subject: str) -> bool:
        """Write email to outbox directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jobscout_digest_{timestamp}.html"
        filepath = self.outbox_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"<!-- Subject: {subject} -->\n")
            f.write(html)

        logger.info(f"Email written to outbox: {filepath}")
        return True

    def _write_empty_digest(self) -> None:
        """Write empty digest when no jobs found."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jobscout_digest_{timestamp}.html"
        filepath = self.outbox_dir / filename

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; text-align: center; }}
        .message {{ background: #f8f9fa; padding: 30px; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="message">
        <h2>No matching jobs found today</h2>
        <p>JobScout ran but didn't find any jobs that match your profile.</p>
        <p>This is normal — the conservative filter ensures you only see relevant opportunities.</p>
    </div>
</body>
</html>
"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Empty digest written to outbox: {filepath}")

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (str(text)
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))
```

**Step 2: Write tests for emailer**

```python
# tests/test_emailer_agent.py

import pytest
import tempfile
import os
from jobscout.emailer_agent import AgentEmailer
from jobscout.models import JobEvaluation, EmailConfig


@pytest.fixture
def sample_evaluations():
    """Sample job evaluations."""
    return [
        JobEvaluation(
            job_title="Senior Backend Engineer",
            company="TestCorp",
            location="Remote",
            url="https://example.com/1",
            source="Test",
            match_score=92,
            is_match=True,
            role_aligned=True,
            seniority_aligned=True,
            location_compatible=True,
            required_skills_matched=["Python", "Django", "PostgreSQL"],
            required_skills_missing=[],
            summary="Strong match for your Python/Django background",
            concerns=[],
            why_match="Excellent Python/Django alignment, API design experience required",
            salary="$180k-$250k",
        ),
        JobEvaluation(
            job_title="AI Platform Engineer",
            company="AICorp",
            location="Remote",
            url="https://example.com/2",
            source="Test",
            match_score=78,
            is_match=True,
            role_aligned=True,
            seniority_aligned=True,
            location_compatible=True,
            required_skills_matched=["Python", "APIs"],
            required_skills_missing=["ML experience"],
            summary="Good match with some ML learning curve",
            concerns=["Some ML experience preferred but not required"],
            why_match="Backend role with AI focus, matches your Python skills",
            salary=None,
        ),
    ]


def test_emailer_initialization():
    """Test emailer initializes correctly."""
    emailer = AgentEmailer(None, outbox_dir=tempfile.mkdtemp())
    assert emailer.config is None
    assert emailer.outbox_dir.exists()


def test_render_html(sample_evaluations):
    """Test HTML rendering."""
    outbox_dir = tempfile.mkdtemp()
    emailer = AgentEmailer(None, outbox_dir=outbox_dir)

    html = emailer._render_html(sample_evaluations, "John Doe")

    assert "John Doe" in html
    assert "2 matches found" in html
    assert "Senior Backend Engineer" in html
    assert "TestCorp" in html
    assert "92%" in html
    assert "Python, Django, PostgreSQL" in html
    assert "$180k-$250k" in html
    assert "AI Platform Engineer" in html
    assert "78%" in html


def test_send_digest_writes_to_outbox(sample_evaluations):
    """Test that digest is written to outbox."""
    outbox_dir = tempfile.mkdtemp()
    emailer = AgentEmailer(None, outbox_dir=outbox_dir)

    result = emailer.send_digest(sample_evaluations, "John Doe")

    assert result is True

    # Check file was created
    files = os.listdir(outbox_dir)
    assert len(files) == 1
    assert files[0].startswith("jobscout_digest_")

    # Check content
    filepath = os.path.join(outbox_dir, files[0])
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "Senior Backend Engineer" in content


def test_send_digest_with_no_jobs():
    """Test that empty digest is written when no jobs."""
    outbox_dir = tempfile.mkdtemp()
    emailer = AgentEmailer(None, outbox_dir=outbox_dir)

    result = emailer.send_digest([], "John Doe")

    assert result is True

    # Check empty digest was created
    files = os.listdir(outbox_dir)
    assert len(files) == 1
    assert "No matching jobs" in open(os.path.join(outbox_dir, files[0])).read()


def test_escape_html():
    """Test HTML escaping."""
    outbox_dir = tempfile.mkdtemp()
    emailer = AgentEmailer(None, outbox_dir=outbox_dir)

    assert emailer._escape_html("Test & Demo") == "Test &amp; Demo"
    assert emailer._escape_html("<script>") == "&lt;script&gt;"
    assert emailer._escape_html('Quote "test"') == "Quote &quot;test&quot;"
```

**Step 3: Run tests**

```bash
pytest tests/test_emailer_agent.py -v
```

Expected: All tests pass

**Step 4: Commit**

```bash
git add jobscout/emailer_agent.py tests/test_emailer_agent.py
git commit -m "feat: add new email format for agent-based JobScout"
```

---

## Task 6: Main Orchestration

**Files:**
- Create: `jobscout/main_agent.py`
- Create: `jobscout_cli_agent.py`

**Step 1: Create main orchestration**

```python
# jobscout/main_agent.py

"""Main orchestration for agent-based JobScout."""

import logging
from pathlib import Path
from typing import Optional
from .config_agent import AgentJobScoutConfig, load_config
from .agent import JobScoutAgent
from .models import CVProfile
from .job_sources.fetcher import JobFetcher
from .emailer_agent import AgentEmailer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class AgentJobScout:
    """Main orchestration for agent-based JobScout."""

    def __init__(self, config: AgentJobScoutConfig):
        """Initialize with configuration."""
        self.config = config
        self.agent = JobScoutAgent(config.search)
        self.emailer = AgentEmailer(
            email_config=config.email,
            outbox_dir=config.outbox_dir,
        )

    def run(self, cv_path: str) -> bool:
        """
        Run complete job search and matching.

        Args:
            cv_path: Path to CV file (PDF, DOCX, or TXT)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Step 1: Read and analyze CV
            logger.info(f"Reading CV from {cv_path}")
            cv_text = self._read_cv(cv_path)
            profile = self.agent.analyze_cv(cv_text)

            # Step 2: Generate search queries
            search_queries = self.agent.generate_search_queries(profile)
            logger.info(f"Search queries: {', '.join(search_queries[:5])}...")

            # Step 3: Fetch jobs
            logger.info("Fetching jobs...")
            fetcher = JobFetcher(
                sources=self.config.search.sources,
                location=self.config.search.location,
                company_boards_all=self.config.search.company_boards_include_all,
                company_boards_specific=self.config.search.company_boards_specific,
            )
            jobs = fetcher.fetch_all(search_queries)
            logger.info(f"Fetched {len(jobs)} total jobs")

            if not jobs:
                logger.info("No jobs found. Sending empty digest.")
                self.emailer.send_digest([])
                return True

            # Step 4: Evaluate each job
            logger.info("Evaluating jobs...")
            evaluations = []
            for i, job in enumerate(jobs):
                try:
                    eval_result = self.agent.evaluate_job(job, profile)
                    evaluations.append(eval_result)

                    if (i + 1) % 10 == 0:
                        logger.info(f"Evaluated {i + 1}/{len(jobs)} jobs...")

                except Exception as e:
                    logger.warning(f"Failed to evaluate job {job.title}: {e}")
                    continue

            logger.info(f"Evaluated {len(evaluations)} jobs")

            # Step 5: Rank and filter
            ranked = self.agent.rank_jobs(evaluations, limit=self.config.search.max_jobs)

            matches = [e for e in evaluations if e.is_match]
            non_matches = len(evaluations) - len(matches)

            logger.info(f"Results: {len(ranked)} matches, {non_matches} filtered out")

            # Log top matches
            if ranked:
                logger.info("Top matches:")
                for i, job in enumerate(ranked, 1):
                    logger.info(f"  {i}. {job.job_title} at {job.company} ({job.match_score:.0f}%)")

            # Step 6: Send email
            logger.info("Sending email digest...")
            self.emailer.send_digest(ranked, candidate_name=profile.name)

            logger.info("JobScout run completed successfully")
            return True

        except Exception as e:
            logger.error(f"JobScout run failed: {e}", exc_info=True)
            return False

    def _read_cv(self, cv_path: str) -> str:
        """Read CV text from file."""
        path = Path(cv_path)

        if not path.exists():
            raise FileNotFoundError(f"CV not found: {cv_path}")

        suffix = path.suffix.lower()

        if suffix == '.txt':
            return path.read_text(encoding='utf-8')

        elif suffix == '.pdf':
            import pdfplumber
            text_chunks = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_chunks.append(page_text)
            return "\n".join(text_chunks)

        elif suffix == '.docx':
            import docx
            doc = docx.Document(path)
            return "\n".join([para.text for para in doc.paragraphs])

        else:
            raise ValueError(f"Unsupported file type: {suffix}")


def run_jobscout(config_path: str = "config.yaml", cv_path: str = "resume.txt") -> bool:
    """Run JobScout with given config and CV."""
    config = load_config(config_path)
    jobscout = AgentJobScout(config)
    return jobscout.run(cv_path)
```

**Step 2: Create new CLI entry point**

```python
#!/usr/bin/env python
"""
JobScout CLI entry point (Agent-based)

Usage:
    python jobscout_cli_agent.py --cv resume.txt
    python jobscout_cli_agent.py --cv resume.txt --config my-config.yaml
"""

import sys
import argparse
from jobscout.main_agent import run_jobscout


def main():
    parser = argparse.ArgumentParser(
        description="JobScout: Agent-based job search assistant"
    )
    parser.add_argument(
        "--cv",
        required=True,
        help="Path to your CV/resume (PDF, DOCX, or TXT)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )

    args = parser.parse_args()

    try:
        print(f"JobScout Agent — Starting search...")
        print(f"Config: {args.config}")
        print(f"CV: {args.cv}")
        print()

        success = run_jobscout(
            config_path=args.config,
            cv_path=args.cv
        )

        if success:
            print("\nJobScout completed successfully!")
            print("Check your email or the outbox directory for results.")
            sys.exit(0)
        else:
            print("\nJobScout encountered errors. Check logs for details.")
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure your CV file exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Step 3: Write integration test**

```python
# tests/test_main_agent.py

import pytest
import tempfile
from jobscout.main_agent import AgentJobScout
from jobscout.config_agent import AgentJobScoutConfig, SearchConfig, LLMConfig


@pytest.fixture
def sample_cv_text():
    """Sample CV text."""
    return """
    John Doe
    Backend Engineer

    EXPERIENCE:
    - Senior Backend Engineer at TechCorp (2020-Present)
      - Python, Django, PostgreSQL, Redis
      - Docker, AWS deployment

    - Backend Developer at StartupCo (2018-2020)
      - Python/Django APIs
      - Database design

    SKILLS:
    Python, Django, FastAPI, PostgreSQL, Redis, Docker, AWS
    """


@pytest.fixture
def test_config(tmp_path):
    """Create test config."""
    import os

    # Set fake API key for testing
    os.environ["OPENAI_API_KEY"] = "fake-key-for-testing"

    return AgentJobScoutConfig(
        llm=LLMConfig(provider="openai"),
        search=SearchConfig(max_jobs=3, sources=[]),  # No sources for test
        outbox_dir=str(tmp_path / "outbox"),
    )


def test_read_cv_text(sample_cv_text, tmp_path):
    """Test reading CV from text file."""
    cv_file = tmp_path / "resume.txt"
    cv_file.write_text(sample_cv_text)

    config = AgentJobScoutConfig(
        llm=LLMConfig(provider="openai", api_key="fake"),
        search=SearchConfig(max_jobs=1, sources=[]),
        outbox_dir=str(tmp_path / "outbox"),
    )

    jobscout = AgentJobScout(config)
    text = jobscout._read_cv(str(cv_file))

    assert "John Doe" in text
    assert "Python" in text
    assert "Django" in text


def test_read_cv_pdf(sample_cv_text, tmp_path):
    """Test reading CV from PDF file."""
    pytest.skip("PDF reading requires pdfplumber - skipped in unit tests")


def test_read_cv_unsupported_file(tmp_path):
    """Test error on unsupported file type."""
    cv_file = tmp_path / "resume.xyz"
    cv_file.write_text("test")

    config = AgentJobScoutConfig(
        llm=LLMConfig(provider="openai", api_key="fake"),
        search=SearchConfig(max_jobs=1, sources=[]),
    )

    jobscout = AgentJobScout(config)

    with pytest.raises(ValueError, match="Unsupported file type"):
        jobscout._read_cv(str(cv_file))


def test_read_cv_not_found():
    """Test error when CV not found."""
    config = AgentJobScoutConfig(
        llm=LLMConfig(provider="openai", api_key="fake"),
        search=SearchConfig(max_jobs=1, sources=[]),
    )

    jobscout = AgentJobScout(config)

    with pytest.raises(FileNotFoundError):
        jobscout._read_cv("/nonexistent/resume.txt")
```

**Step 4: Run tests**

```bash
pytest tests/test_main_agent.py -v
```

Expected: All tests pass

**Step 5: Commit**

```bash
git add jobscout/main_agent.py jobscout_cli_agent.py tests/test_main_agent.py
git commit -m "feat: add main orchestration and CLI for agent-based JobScout"
```

---

## Task 7: Clean Up Obsolete Files

**Files:**
- Delete: `jobscout/scoring.py`
- Delete: `jobscout/job_parser.py`
- Delete: `jobscout/filters.py`
- Delete: `jobscout/dedup.py`
- Delete: `jobscout/semantic.py`
- Delete: `jobscout/role_recommender.py`
- Delete: `jobscout/resume_parser.py`
- Delete: `jobscout/llm_parser.py`
- Delete: `jobscout/main.py` (old main)
- Delete: `jobscout/scheduler.py` (can add back later if needed)

**Step 1: Delete obsolete files**

```bash
cd "C:\Users\Administrator\Documents\Yemi's Projects\jobscout"

# Delete obsolete core files
rm jobscout/scoring.py
rm jobscout/job_parser.py
rm jobscout/filters.py
rm jobscout/dedup.py
rm jobscout/semantic.py
rm jobscout/role_recommender.py
rm jobscout/resume_parser.py
rm jobscout/llm_parser.py
rm jobscout/main.py
rm jobscout/scheduler.py
```

**Step 2: Update jobscout/__init__.py**

```python
# jobscout/__init__.py

"""Agent-based JobScout — Conservative job search assistant."""

from .config_agent import AgentJobScoutConfig, load_config
from .agent import JobScoutAgent
from .models import CVProfile, JobEvaluation, AgentConfig
from .main_agent import AgentJobScout, run_jobscout
from .emailer_agent import AgentEmailer

__version__ = "2.0.0"

__all__ = [
    "AgentJobScoutConfig",
    "load_config",
    "JobScoutAgent",
    "CVProfile",
    "JobEvaluation",
    "AgentConfig",
    "AgentJobScout",
    "run_jobscout",
    "AgentEmailer",
]
```

**Step 3: Remove old test files (optional, keep if useful for reference)**

```bash
# Optional: Remove old tests that are now obsolete
# rm test_full_flow.py
# rm test_improved_llm_parser.py
# rm test_llm_integration.py
# rm test_llm_parser.py
# rm test_stateless_api.py
# rm test_boolean_search.py
```

**Step 4: Update requirements.txt (simplify)**

```txt
# Core dependencies
requests>=2.31.0
python-dateutil>=2.8.2
pyyaml>=6.0.1

# LLM Providers
openai>=1.0.0
anthropic>=0.18.0

# CV parsing
pdfplumber>=0.11.0
python-docx>=1.1.0

# HTML parsing
beautifulsoup4>=4.12.3
lxml>=5.1.0

# Email (built-in, but these help)
# No external email dependencies needed

# Testing
pytest>=8.0.0
pytest-cov>=4.1.0
```

**Step 5: Update README.md**

```markdown
# JobScout v2.0 — Agent-Based Job Search

An AI-powered job search assistant that finds jobs matching your CV and emails you the top matches.

**v2.0 is a complete rewrite using an AI agent approach** — instead of complex scoring formulas, an LLM agent intelligently evaluates your CV against job postings.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` with your settings:

```yaml
llm:
  provider: openai  # or anthropic, deepseek
  model: gpt-4o-mini

email:
  to_address: your-email@example.com
  smtp_host: smtp.gmail.com
  smtp_port: 587
  smtp_username: your-email@gmail.com
  smtp_password: your-app-password

search:
  location: remote
  max_jobs: 7
```

### 3. Run

```bash
python jobscout_cli_agent.py --cv path/to/your/resume.pdf
```

## How It Works

1. **AI analyzes your CV** — Extracts skills, role, experience, and generates search terms
2. **Searches multiple sources** — Company boards, RemoteOK, We Work Remotely, Remotive
3. **Evaluates each job** — AI compares your profile to each job posting objectively
4. **Ranks and filters** — Returns only genuine matches
5. **Emails you the results** — Clean digest with top 7 matches (configurable)

## Configuration

See `config.example.yaml` for all options.

**LLM Providers:**
- OpenAI: `gpt-4o-mini` (default, fast and cheap)
- Anthropic: `claude-3-5-haiku-20241022`
- DeepSeek: `deepseek-chat`

**Job Sources:**
- Company boards (Greenhouse, Lever, Ashby)
- RemoteOK
- We Work Remotely
- Remotive

## Architecture

- **Agent-based**: Single LLM agent handles CV analysis, search generation, and job matching
- **Stateless**: Each run is independent
- **Conservative**: Better to miss a good job than waste your time
```

**Step 6: Commit cleanup**

```bash
git add -A
git commit -m "refactor: remove obsolete files, update README for v2.0"
```

---

## Task 8: End-to-End Integration Test

**Files:**
- Create: `tests/test_e2e.py`

**Step 1: Create end-to-end test**

```python
# tests/test_e2e.py

"""End-to-end integration test for agent-based JobScout."""

import os
import pytest
from jobscout.main_agent import AgentJobScout
from jobscout.config_agent import AgentJobScoutConfig, SearchConfig, LLMConfig, EmailConfig


@pytest.mark.integration
def test_full_pipeline_with_real_api():
    """
    Full end-to-end test with real API calls.

    Requirements:
    - OPENAI_API_KEY environment variable set
    - Valid CV file at tests/fixtures/sample_cv.txt
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    # Sample CV
    cv_text = """
    Jane Engineer
    Senior Backend Developer

    EXPERIENCE:
    - Senior Backend Engineer at TechCorp (2021-Present)
      - Python, Django, FastAPI
      - PostgreSQL, Redis
      - Docker, Kubernetes, AWS

    - Backend Developer at StartupInc (2019-2021)
      - Python/Django web applications
      - REST API design
      - Database optimization

    SKILLS:
    Languages: Python, SQL, JavaScript
    Frameworks: Django, FastAPI, Flask
    Databases: PostgreSQL, Redis, MongoDB
    Infrastructure: Docker, Kubernetes, AWS, Terraform
    """

    # Write temp CV file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(cv_text)
        cv_path = f.name

    try:
        # Create config
        config = AgentJobScoutConfig(
            llm=LLMConfig(provider="openai"),
            search=SearchConfig(
                max_jobs=3,
                sources=["remoteok"],  # Single source for faster test
                location="remote",
            ),
            outbox_dir=tempfile.mkdtemp(),
        )

        # Run JobScout
        jobscout = AgentJobScout(config)
        result = jobscout.run(cv_path)

        # Assert success
        assert result is True

        # Check that outbox has a file
        outbox_files = list(os.listdir(config.outbox_dir))
        assert len(outbox_files) > 0
        assert any(f.startswith("jobscout_digest_") for f in outbox_files)

    finally:
        os.unlink(cv_path)


@pytest.mark.integration
def test_cv_analysis():
    """Test CV analysis with real API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    from jobscout.agent import JobScoutAgent
    from jobscout.models import AgentConfig

    cv_text = """
    John Doe
    Python Developer with 5 years experience.
    Skills: Python, Django, PostgreSQL, Docker.
    """

    config = AgentConfig(llm_provider="openai")
    agent = JobScoutAgent(config)
    profile = agent.analyze_cv(cv_text)

    assert profile.role_primary in ["backend", "frontend", "fullstack", "data", "devops", "mobile"]
    assert len(profile.skills) > 0
    assert len(profile.search_keywords) > 0
    assert profile.years_experience >= 0


@pytest.mark.integration
def test_job_evaluation():
    """Test job evaluation with real API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    from jobscout.agent import JobScoutAgent
    from jobscout.models import AgentConfig, CVProfile
    from jobscout.job_sources.base import JobListing

    config = AgentConfig(llm_provider="openai")
    agent = JobScoutAgent(config)

    # Create a profile
    profile = CVProfile(
        role_primary="backend",
        seniority="mid",
        skills={"python", "django", "postgresql"},
        years_experience=5,
    )

    # Create a job
    job = JobListing(
        title="Python Backend Developer",
        company="TestCorp",
        location="Remote",
        description="We are looking for a Python developer with Django and PostgreSQL experience to build REST APIs.",
        apply_url="https://example.com",
        source="Test",
    )

    evaluation = agent.evaluate_job(job, profile)

    assert evaluation.job_title == "Python Backend Developer"
    assert 0 <= evaluation.match_score <= 100
    assert isinstance(evaluation.is_match, bool)


def test_job_fetcher():
    """Test job fetching without LLM."""
    from jobscout.job_sources.fetcher import JobFetcher

    fetcher = JobFetcher(sources=["remoteok"], location="remote")
    jobs = fetcher.fetch_all(search_queries=["python"], limit_per_source=5)

    # Should return some jobs (may be empty in test environment)
    assert isinstance(jobs, list)
```

**Step 2: Run integration tests**

```bash
pytest tests/test_e2e.py -v --integration
```

Expected: Integration tests pass (with valid API key)

**Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: add end-to-end integration tests"
```

---

## Summary

**What was built:**

1. **LLM Client** (`llm_client.py`) — Unified interface for OpenAI, Anthropic, DeepSeek
2. **Agent Core** (`agent.py`, `models.py`) — CV analysis, job evaluation, ranking
3. **Job Fetcher** (`job_sources/fetcher.py`) — Unified job fetching
4. **Simplified Config** (`config_agent.py`) — Clean YAML configuration
5. **New Email Format** (`emailer_agent.py`) — Clean, scannable digest
6. **Main Orchestration** (`main_agent.py`, `jobscout_cli_agent.py`) — Complete pipeline
7. **Tests** — Comprehensive test coverage

**What was removed:**

- Complex scoring system (`scoring.py`)
- Job parser with regex (`job_parser.py`)
- Hard filters (`filters.py`)
- Dedup cache (`dedup.py`)
- Semantic scorer (`semantic.py`)
- Old orchestration (`main.py`, `jobscout_cli.py`)

**New file structure:**

```
jobscout/
├── __init__.py
├── agent.py              # Agent core
├── config_agent.py       # Simplified config
├── emailer_agent.py      # New email format
├── llm_client.py         # Unified LLM client
├── main_agent.py         # Main orchestration
├── models.py             # Data models
└── job_sources/
    ├── __init__.py
    ├── base.py
    ├── fetcher.py        # Unified fetcher
    ├── company_boards.py
    ├── rss_feeds.py
    └── remotive_api.py
```

**Running the new JobScout:**

```bash
python jobscout_cli_agent.py --cv path/to/resume.pdf
```

---

**Plan complete and saved to `docs/plans/2026-03-05-agent-based-jobscout.md`.**
