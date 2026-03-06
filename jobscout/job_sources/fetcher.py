"""Unified job fetcher that uses all sources."""

import logging
import os
from typing import List, Optional
from .base import JobListing
from .company_boards import CompanyBoardsSource
from .rss_feeds import RemoteOKSource, WeWorkRemotelySource, HimalayasSource, JavascriptJobsSource
from .remotive_api import RemotiveSource
from .boolean_search import BooleanSearchSource


logger = logging.getLogger(__name__)


class JobFetcher:
    """Fetches jobs from multiple sources based on search queries."""

    # Default sources if none specified - include all available sources
    DEFAULT_SOURCES = ["company_boards", "remoteok", "weworkremotely", "remotive", "himalayas", "jsjobs"]

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
        self.sources = sources or self.DEFAULT_SOURCES.copy()
        self.location = location.lower()
        self.company_boards_all = company_boards_all
        self.company_boards_specific = company_boards_specific or []

    def fetch_all(
        self,
        search_queries: Optional[List[str]] = None,
        limit_per_source: int = 50,
    ) -> List[JobListing]:
        """
        Fetch jobs from all sources.

        Args:
            search_queries: Optional list of search terms for filtering
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
                unique_count = 0
                for job in jobs:
                    if job.apply_url and job.apply_url not in seen_urls:
                        seen_urls.add(job.apply_url)
                        all_jobs.append(job)
                        unique_count += 1

                logger.info(f"Fetched {unique_count} unique jobs from {source}")

            except Exception as e:
                logger.error(f"Failed to fetch from {source}: {e}")
                continue

        logger.info(f"Total unique jobs fetched: {len(all_jobs)}")
        return all_jobs

    def _fetch_from_source(
        self,
        source: str,
        search_queries: Optional[List[str]],
        limit: int,
    ) -> List[JobListing]:
        """Fetch from a specific source."""
        source_lower = source.lower()

        if source_lower == "company_boards" or source_lower == "company":
            return self._fetch_company_boards(search_queries, limit)

        elif source_lower == "remoteok":
            return self._fetch_remoteok(search_queries, limit)

        elif source_lower == "weworkremotely":
            return self._fetch_weworkremotely(search_queries, limit)

        elif source_lower == "himalayas":
            return self._fetch_himalayas(search_queries, limit)

        elif source_lower == "remotive":
            return self._fetch_remotive(search_queries, limit)

        elif source_lower == "jsjobs":
            return self._fetch_jsjobs(search_queries, limit)

        elif source_lower == "boolean":
            return self._fetch_boolean(search_queries, limit)

        else:
            logger.warning(f"Unknown source: {source}")
            return []

    def _fetch_company_boards(
        self,
        search_queries: Optional[List[str]],
        limit: int,
    ) -> List[JobListing]:
        """Fetch from company boards (Greenhouse/Lever/Ashby)."""
        src = CompanyBoardsSource(
            resume_skills=set(),
            role_keywords=search_queries or ["software engineer"],
            location_preference=self.location,
            max_job_age_days=7,
            companies=self.company_boards_specific if not self.company_boards_all else None,
        )
        return src.fetch_jobs(limit=limit)

    def _fetch_remoteok(
        self,
        search_queries: Optional[List[str]],
        limit: int,
    ) -> List[JobListing]:
        """Fetch from RemoteOK."""
        src = RemoteOKSource("RemoteOK")
        all_jobs = src.fetch_jobs(limit=limit)
        return self._filter_by_search_terms(all_jobs, search_queries)

    def _fetch_weworkremotely(
        self,
        search_queries: Optional[List[str]],
        limit: int,
    ) -> List[JobListing]:
        """Fetch from We Work Remotely."""
        src = WeWorkRemotelySource("We Work Remotely")
        all_jobs = src.fetch_jobs(limit=limit)
        return self._filter_by_search_terms(all_jobs, search_queries)

    def _fetch_himalayas(
        self,
        search_queries: Optional[List[str]],
        limit: int,
    ) -> List[JobListing]:
        """Fetch from Himalayas."""
        src = HimalayasSource("Himalayas")
        all_jobs = src.fetch_jobs(limit=limit)
        return self._filter_by_search_terms(all_jobs, search_queries)

    def _fetch_remotive(
        self,
        search_queries: Optional[List[str]],
        limit: int,
    ) -> List[JobListing]:
        """Fetch from Remotive."""
        src = RemotiveSource("Remotive")
        # Remotive has built-in filtering by category
        # Just return dev jobs
        return src.fetch_jobs(limit=limit)

    def _fetch_jsjobs(
        self,
        search_queries: Optional[List[str]],
        limit: int,
    ) -> List[JobListing]:
        """Fetch from JavaScriptJobs."""
        src = JavascriptJobsSource("JavaScriptJobs")
        all_jobs = src.fetch_jobs(limit=limit)
        return self._filter_by_search_terms(all_jobs, search_queries)

    def _fetch_boolean(
        self,
        search_queries: Optional[List[str]],
        limit: int,
    ) -> List[JobListing]:
        """Fetch from Boolean search (Google/Serper)."""
        serper_key = os.getenv("SERPER_API_KEY")
        if not serper_key:
            logger.info("SERPER_API_KEY not set, skipping boolean search")
            return []

        # Build search parameters from profile
        role_keywords = search_queries or ["software engineer", "developer"]
        location_pref = self.location

        # Infer user skills and seniority from search queries
        user_skills = set()
        for query in (search_queries or []):
            # Extract skill-like terms
            for word in query.split():
                if len(word) > 2:
                    user_skills.add(word.lower())

        src = BooleanSearchSource(
            resume_skills=user_skills,
            role_keywords=role_keywords[:5],
            seniority="unknown",
            location_preference=location_pref,
            max_job_age_days=7,
            serper_api_key=serper_key
        )
        return src.fetch_jobs(limit=limit)

    def _filter_by_search_terms(
        self,
        jobs: List[JobListing],
        search_queries: Optional[List[str]],
    ) -> List[JobListing]:
        """Filter jobs by search terms (simple keyword matching)."""
        if not search_queries:
            return jobs

        filtered = []
        # Create a set of normalized search terms
        search_terms = set(q.lower() for q in search_queries)

        # Also extract individual words from multi-word queries
        search_words = set()
        for term in search_terms:
            search_words.update(term.split())

        for job in jobs:
            job_text = f"{job.title} {job.description}".lower()

            # Job matches if any search term or word is present
            if any(term in job_text for term in search_terms) or \
               any(word in job_text.split() for word in search_words if len(word) > 3):
                filtered.append(job)

        return filtered


def fetch_jobs_for_profile(
    profile,
    location: str = "remote",
    sources: Optional[List[str]] = None,
) -> List[JobListing]:
    """
    Convenience function to fetch jobs based on a CV profile.

    Args:
        profile: CVProfile with search_keywords
        location: Location preference
        sources: List of sources to use

    Returns:
        List of job listings
    """
    fetcher = JobFetcher(
        sources=sources,
        location=location,
        company_boards_all=True,
    )
    return fetcher.fetch_all(
        search_queries=profile.search_keywords,
        limit_per_source=50,
    )
