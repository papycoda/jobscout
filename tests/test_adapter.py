"""Tests for backend adapter behavior."""

from datetime import datetime

from backend.adapter import JobScoutAdapter
from jobscout.config import JobScoutConfig, JobPreferences, EmailConfig, ScheduleConfig
from jobscout.job_sources.base import JobListing


def test_empty_result_handles_filtered_jobs(tmp_path):
    """Ensure empty-result path doesn't reference undefined variables."""
    config = JobScoutConfig(
        resume_path="unused.txt",
        email=EmailConfig(enabled=False),
        schedule=ScheduleConfig(),
        job_preferences=JobPreferences(),
        outbox_dir=str(tmp_path / "outbox")
    )
    resume_profile = {
        "skills": ["python"],
        "tools": [],
        "seniority": "mid",
        "years_experience": 3,
        "role_keywords": ["backend engineer"]
    }
    adapter = JobScoutAdapter(config, resume_profile=resume_profile)

    job = JobListing(
        title="Backend Engineer",
        company="TestCo",
        location="Remote",
        description="Great job",
        apply_url="https://example.com/apply",
        source="TestSource"
    )
    filtered_jobs = [{"job": job, "reasons": ["location mismatch"]}]

    result = adapter._empty_result(filtered_jobs, datetime.now())

    assert result["jobs"] == []
    assert result["metadata"]["status"] == "completed"
    assert result["metadata"]["matching"] == 0
    assert len(result["filtered_jobs"]) == 1
    assert result["filtered_jobs"][0]["apply_url"] == "https://example.com/apply"
    assert result["filtered_jobs"][0]["reasons"] == ["location mismatch"]
    assert result["filtered_jobs"][0]["reason_summary"] == "Location mismatch"
    assert result["filtered_jobs"][0]["reason_detail"] == "location mismatch"
    assert result["filtered_jobs"][0]["score_total"] is None
