"""Boolean search via public indexing (Greenhouse, Lever)."""

import logging
import re
from typing import List, Set
from typing import Tuple
from .base import JobSource, JobListing


logger = logging.getLogger(__name__)


class BooleanSearchSource(JobSource):
    """
    Fetch jobs using Boolean search on public indexing.

    Searches for jobs on:
    - Greenhouse (boards.greenhouse.io)
    - Lever (jobs.lever.co)

    Conservative approach: Only fetch URLs returned by search,
    never crawl company pages directly.
    """

    def __init__(self, resume_skills: Set[str], role_keywords: List[str]):
        super().__init__("BooleanSearch")
        self.resume_skills = resume_skills
        self.role_keywords = role_keywords

    def fetch_jobs(self, limit: int = 30) -> List[JobListing]:
        """
        Fetch jobs using Boolean search patterns.

        Hard cap at 30 results per run to avoid overwhelming processing.
        """
        # For MVP, we simulate boolean search results
        # In production, this would integrate with a search API
        # or use a scraping approach that respects robots.txt

        logger.info("Boolean search is simulated in MVP")
        logger.info(f"Skills: {self.resume_skills}")
        logger.info(f"Role keywords: {self.role_keywords}")

        # Build boolean queries (for logging/reference)
        queries = self._build_boolean_queries()
        for query in queries:
            logger.info(f"Boolean query: {query}")

        # Return empty list for MVP - implement search integration as enhancement
        # This would require:
        # 1. Search API access (Google Custom Search, Bing API)
        # 2. Or respectful scraping with rate limiting
        return []

    def _build_boolean_queries(self) -> List[str]:
        """Build Boolean search queries from resume data."""
        queries = []

        # Build skill OR clause
        skill_terms = list(self.resume_skills)[:10]  # Limit to top 10
        if not skill_terms:
            return []

        skill_clause = " OR ".join(f'"{s}"' for s in skill_terms)

        # Build role clause
        role_terms = self.role_keywords[:3] if self.role_keywords else ['"software engineer"']
        role_clause = " OR ".join(role_terms)

        # Greenhouse query
        greenhouse_query = f'({skill_clause}) AND ({role_clause}) site:boards.greenhouse.io'
        queries.append(greenhouse_query)

        # Lever query
        lever_query = f'({skill_clause}) AND ({role_clause}) site:jobs.lever.co'
        queries.append(lever_query)

        return queries

    def _parse_greenhouse_job(self, html: str, url: str) -> JobListing:
        """Parse job details from Greenhouse page."""
        # Simplified parsing - in production would use BeautifulSoup
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'lxml')

        # Extract basic info
        title = soup.find('h1', class_='app-title')
        title = title.text.strip() if title else "Unknown Title"

        company = url.split('/')[2].replace('.boards.greenhouse.io', '')
        company = company.replace('greenhouse', '').title()

        # Extract description
        content_div = soup.find('div', class_='content')
        description = content_div.get_text('\n', strip=True) if content_div else ""

        return JobListing(
            title=title,
            company=company,
            location="Remote",  # Would parse from page
            description=description,
            apply_url=url,
            source="Greenhouse",
            posted_date=None
        )

    def _parse_lever_job(self, html: str, url: str) -> JobListing:
        """Parse job details from Lever page."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'lxml')

        # Extract basic info
        title = soup.find('h2', class_='posting-title')
        title = title.text.strip() if title else "Unknown Title"

        company = url.split('/')[2].replace('.jobs.lever.co', '')
        company = company.replace('lever', '').title()

        # Extract description
        content_div = soup.find('div', class_='posting-section')
        description = content_div.get_text('\n', strip=True) if content_div else ""

        return JobListing(
            title=title,
            company=company,
            location="Remote",  # Would parse from page
            description=description,
            apply_url=url,
            source="Lever",
            posted_date=None
        )
