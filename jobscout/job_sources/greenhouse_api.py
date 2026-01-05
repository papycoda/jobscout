"""Greenhouse job board API source."""

import logging
from datetime import datetime
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from .base import JobSource, JobListing


logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10


class GreenhouseSource(JobSource):
    """Fetch jobs from Greenhouse boards API."""

    def __init__(self, boards: List[str]):
        super().__init__("Greenhouse")
        self.boards = [board.strip() for board in boards if board and board.strip()]

    def fetch_jobs(self, limit: int = 50) -> List[JobListing]:
        if not self.boards:
            logger.info("No Greenhouse boards configured; skipping.")
            return []

        all_jobs: List[JobListing] = []
        for board in self.boards:
            if len(all_jobs) >= limit:
                break
            remaining = limit - len(all_jobs)
            all_jobs.extend(self._fetch_board(board, remaining))

        return all_jobs

    def _fetch_board(self, board: str, limit: int) -> List[JobListing]:
        url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true"
        try:
            response = requests.get(url, timeout=_DEFAULT_TIMEOUT, headers={"User-Agent": "JobScout/1.0"})
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(f"Greenhouse fetch failed for board {board}: {exc}")
            return []

        payload = response.json()
        jobs = payload.get("jobs", [])
        results: List[JobListing] = []

        for job in jobs:
            if len(results) >= limit:
                break

            title = job.get("title") or "Unknown Title"
            location = self._extract_location(job.get("location"))
            apply_url = job.get("absolute_url") or job.get("url") or ""
            description_html = job.get("content") or ""

            if not description_html and job.get("id"):
                description_html = self._fetch_job_detail(board, job.get("id"))

            description = self._clean_html(description_html)
            posted_date = self._parse_iso_date(job.get("updated_at") or job.get("created_at"))

            results.append(JobListing(
                title=title,
                company=board.replace("-", " ").title(),
                location=location or "Remote",
                description=description,
                apply_url=apply_url,
                source="Greenhouse",
                posted_date=posted_date
            ))

        return results

    def _fetch_job_detail(self, board: str, job_id: int) -> str:
        url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}"
        try:
            response = requests.get(url, timeout=_DEFAULT_TIMEOUT, headers={"User-Agent": "JobScout/1.0"})
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(f"Greenhouse detail fetch failed for {board} job {job_id}: {exc}")
            return ""

        payload = response.json()
        return payload.get("content") or ""

    def _clean_html(self, html: str) -> str:
        if not html:
            return ""
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text("\n", strip=True)

    def _parse_iso_date(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None

    def _extract_location(self, location_data) -> str:
        if isinstance(location_data, dict):
            return location_data.get("name") or ""
        if isinstance(location_data, str):
            return location_data
        return ""
