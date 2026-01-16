"""Pydantic models for FastAPI requests and responses."""

from pydantic import BaseModel
from typing import List, Optional, Set
from datetime import datetime


class HealthResponse(BaseModel):
    ok: bool


class ConfigResponse(BaseModel):
    config_yaml: str


class ConfigUpdateRequest(BaseModel):
    config_yaml: str


class ResumeUploadResponse(BaseModel):
    resume_path: str
    skills: List[str]
    metadata: dict


class SearchResponse(BaseModel):
    run_id: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str  # 'pending' | 'running' | 'completed' | 'failed'
    message: Optional[str] = None
    jobs_found: Optional[int] = None
    jobs_matched: Optional[int] = None


class JobBreakdown(BaseModel):
    must_have_coverage: float  # 0-1
    stack_overlap: float  # 0-1
    seniority_alignment: float  # 0-1


class SeniorityDetails(BaseModel):
    expected: str
    found: str
    explanation: str


class JobDetails(BaseModel):
    id: str
    title: str
    company: str
    location: str
    posted_at: Optional[str]
    apply_url: str
    source: str
    snippet: str
    score_total: float  # 0-100
    breakdown: JobBreakdown
    must_have: dict
    stack: dict
    seniority: SeniorityDetails


class Pagination(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int


class JobsListResponse(BaseModel):
    jobs: List[JobDetails]
    pagination: Optional[Pagination] = None


class FilteredJobDetails(BaseModel):
    id: str
    title: str
    company: str
    location: str
    apply_url: str
    source: str
    snippet: str
    score_total: Optional[float]
    reasons: List[str]
    reason_summary: Optional[str] = None
    reason_detail: Optional[str] = None


class FilteredJobsListResponse(BaseModel):
    filtered_jobs: List[FilteredJobDetails]
    pagination: Optional[Pagination] = None


class SendDigestRequest(BaseModel):
    run_id: Optional[str] = None


class SendDigestResponse(BaseModel):
    digest_id: str
    mode: str  # "smtp", "outbox", or "noop"


class SendTestEmailResponse(BaseModel):
    ok: bool


class DigestMetadata(BaseModel):
    id: str
    created_at: str
    subject: str
    mode: str


class DigestsListResponse(BaseModel):
    digests: List[DigestMetadata]


class DigestResponse(BaseModel):
    id: str
    html: str
    created_at: str
    subject: str


# LLM Models
class LLMModelInfo(BaseModel):
    id: str
    name: str
    description: str
    supports_reasoning: bool
    cost_estimate: str


class LLMProviderModels(BaseModel):
    provider: str
    models: List[LLMModelInfo]


class AvailableModelsResponse(BaseModel):
    models: List[LLMProviderModels]
    default_model: str


class UpdateLLMModelRequest(BaseModel):
    model_id: str
    api_key: Optional[str] = None  # If not provided, uses stored key


class UpdateLLMModelResponse(BaseModel):
    success: bool
    model_id: str
    model_name: str


# Metrics
class MetricsResponse(BaseModel):
    total_searches: int
    total_resume_uploads: int
    total_emails_sent: int
    recent_searches: List[dict]
    avg_jobs_per_search: float
    avg_match_rate: float
