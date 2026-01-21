"""Remotive API job source."""

import requests
import logging
from datetime import datetime
from typing import List
from .base import JobSource, JobListing, strip_html_tags
from .truncation import expand_truncated_jobs


logger = logging.getLogger(__name__)


class RemotiveSource(JobSource):
    """Fetch jobs from Remotive API."""

    REMOTIVE_API = "https://remotive.com/api/remote-jobs"

    def fetch_jobs(self, limit: int = 50) -> List[JobListing]:
        """Fetch jobs from Remotive API."""
        jobs = []
        try:
            response = requests.get(self.REMOTIVE_API, timeout=10)
            response.raise_for_status()

            data = response.json()

            for job_data in data.get('jobs', [])[:limit]:
                try:
                    # Filter to tech roles only
                    category = job_data.get('category', '').lower()
                    if 'software' not in category and 'dev' not in category:
                        continue

                    title = job_data.get('title', '')
                    company = job_data.get('company_name', 'Unknown Company')
                    apply_url = job_data.get('url', '')
                    description = strip_html_tags(job_data.get('description', ''))

                    # Remotive is 100% remote
                    location = "Remote"

                    # Parse date
                    posted_date = None
                    publication_date = job_data.get('publication_date')
                    if publication_date:
                        try:
                            posted_date = datetime.fromisoformat(publication_date.replace('Z', '+00:00'))
                        except (ValueError, AttributeError):
                            pass

                    job = JobListing(
                        title=title,
                        company=company,
                        location=location,
                        description=description,
                        apply_url=apply_url,
                        source="Remotive",
                        posted_date=posted_date
                    )
                    jobs.append(job)

                except Exception as e:
                    logger.warning(f"Failed to parse Remotive job: {e}")
                    continue

        except requests.RequestException as e:
            logger.error(f"Failed to fetch Remotive API: {e}")

        # Expand truncated job descriptions
        jobs = expand_truncated_jobs(jobs)
        return jobs
