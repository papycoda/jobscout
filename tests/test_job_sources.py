"""Tests for job sources with mocking."""

import pytest
from jobscout.job_sources.base import JobListing
from jobscout.job_sources.rss_feeds import RemoteOKSource
from jobscout.job_sources.remotive_api import RemotiveSource


class TestJobSources:
    """Test job source fetching."""

    def test_remoteok_source_structure(self):
        """Test RemoteOK source can be instantiated and has correct structure."""
        source = RemoteOKSource("RemoteOK")

        assert source.name == "RemoteOK"
        assert hasattr(source, 'fetch_jobs')

    def test_remotive_source_structure(self):
        """Test Remotive source can be instantiated and has correct structure."""
        source = RemotiveSource("Remotive")

        assert source.name == "Remotive"
        assert hasattr(source, 'fetch_jobs')

    def test_job_listing_hash_and_equality(self):
        """Test job listing deduplication logic."""
        job1 = JobListing(
            title="Backend Engineer",
            company="TestCorp",
            location="Remote",
            description="Python developer needed",
            apply_url="https://example.com/job1",
            source="RemoteOK"
        )

        job2 = JobListing(
            title="Backend Engineer",
            company="TestCorp",
            location="Remote",
            description="Python developer needed",
            apply_url="https://example.com/job1",
            source="RemoteOK"
        )

        job3 = JobListing(
            title="Backend Engineer",
            company="TestCorp",
            location="Remote",
            description="Python developer needed",
            apply_url="https://example.com/job2",  # Different URL
            source="RemoteOK"
        )

        # job1 and job2 should be equal (same content)
        assert job1 == job2
        assert hash(job1) == hash(job2)

        # job3 should be different (different URL)
        assert job1 != job3
        assert hash(job1) != hash(job3)

    def test_job_listing_deduplication_in_set(self):
        """Test that job listings can be deduplicated using sets."""
        jobs = [
            JobListing(
                title="Backend Engineer",
                company="TestCorp",
                location="Remote",
                description="Python developer",
                apply_url="https://example.com/1",
                source="RemoteOK"
            ),
            JobListing(
                title="Backend Engineer",
                company="TestCorp",
                location="Remote",
                description="Python developer",
                apply_url="https://example.com/1",  # Duplicate
                source="RemoteOK"
            ),
            JobListing(
                title="Frontend Engineer",
                company="TestCorp",
                location="Remote",
                description="React developer",
                apply_url="https://example.com/2",
                source="RemoteOK"
            ),
        ]

        # Should deduplicate to 2 unique jobs
        unique_jobs = list(set(jobs))
        assert len(unique_jobs) == 2
