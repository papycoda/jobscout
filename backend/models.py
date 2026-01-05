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


class JobsListResponse(BaseModel):
    jobs: List[JobDetails]


class FilteredJobDetails(BaseModel):
    id: str
    title: str
    company: str
    location: str
    source: str
    snippet: str
    score_total: Optional[float]
    reasons: List[str]


class FilteredJobsListResponse(BaseModel):
    filtered_jobs: List[FilteredJobDetails]


class SendDigestRequest(BaseModel):
    run_id: Optional[str] = None


class SendDigestResponse(BaseModel):
    digest_id: str
    mode: str  # "smtp" or "outbox"


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
