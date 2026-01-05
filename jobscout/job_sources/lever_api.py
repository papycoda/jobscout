"""Lever job board API source."""

import logging
from datetime import datetime
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from .base import JobSource, JobListing


logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10


class LeverSource(JobSource):
    """Fetch jobs from Lever postings API."""

    def __init__(self, companies: List[str]):
        super().__init__("Lever")
        self.companies = [company.strip() for company in companies if company and company.strip()]

    def fetch_jobs(self, limit: int = 50) -> List[JobListing]:
        if not self.companies:
            logger.info("No Lever companies configured; skipping.")
            return []

        all_jobs: List[JobListing] = []
        for company in self.companies:
            if len(all_jobs) >= limit:
                break
            remaining = limit - len(all_jobs)
            all_jobs.extend(self._fetch_company(company, remaining))

        return all_jobs

    def _fetch_company(self, company: str, limit: int) -> List[JobListing]:
        url = f"https://api.lever.co/v0/postings/{company}?mode=json"
        try:
            response = requests.get(url, timeout=_DEFAULT_TIMEOUT, headers={"User-Agent": "JobScout/1.0"})
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(f"Lever fetch failed for company {company}: {exc}")
            return []

        postings = response.json()
        results: List[JobListing] = []

        for job in postings:
            if len(results) >= limit:
                break

            title = job.get("text") or "Unknown Title"
            location = self._extract_location(job.get("categories", {}))
            apply_url = job.get("hostedUrl") or job.get("applyUrl") or ""
            description_html = job.get("description") or ""
            if not description_html:
                description_html = self._build_description(job.get("lists", []))

            description = self._clean_html(description_html)
            posted_date = self._parse_timestamp(job.get("createdAt"))

            results.append(JobListing(
                title=title,
                company=company.replace("-", " ").title(),
                location=location or "Remote",
                description=description,
                apply_url=apply_url,
                source="Lever",
                posted_date=posted_date
            ))

        return results

    def _build_description(self, lists) -> str:
        if not lists:
            return ""
        sections = []
        for section in lists:
            heading = section.get("text") if isinstance(section, dict) else ""
            items = section.get("content") if isinstance(section, dict) else None
            if heading:
                sections.append(f"<h3>{heading}</h3>")
            if items:
                sections.append("<ul>")
                sections.extend([f"<li>{item}</li>" for item in items])
                sections.append("</ul>")
        return "\n".join(sections)

    def _clean_html(self, html: str) -> str:
        if not html:
            return ""
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text("\n", strip=True)

    def _parse_timestamp(self, value) -> Optional[datetime]:
        if value is None:
            return None
        try:
            return datetime.utcfromtimestamp(int(value) / 1000)
        except (TypeError, ValueError):
            return None

    def _extract_location(self, categories) -> str:
        if not isinstance(categories, dict):
            return ""
        return categories.get("location") or ""
