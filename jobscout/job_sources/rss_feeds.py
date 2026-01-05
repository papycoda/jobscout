"""RSS feed job sources (RemoteOK, We Work Remotely)."""

import feedparser
import logging
from datetime import datetime
from typing import List
from .base import JobSource, JobListing


logger = logging.getLogger(__name__)


class RemoteOKSource(JobSource):
    """Fetch jobs from RemoteOK via RSS."""

    REMOTEOK_RSS = "https://remoteok.com/feed"

    def fetch_jobs(self, limit: int = 50) -> List[JobListing]:
        """Fetch jobs from RemoteOK."""
        jobs = []
        try:
            feed = feedparser.parse(self.REMOTEOK_RSS)

            for entry in feed.entries[:limit]:
                try:
                    # RemoteOK RSS structure
                    title = entry.get('title', '')

                    # Skip non-dev roles
                    if any(keyword in title.lower() for keyword in ['design', 'marketing', 'sales', 'support']):
                        continue

                    company = entry.get('author', 'Unknown Company')
                    apply_url = entry.get('link', '')
                    description = entry.get('description', '')

                    # Extract location from description (RemoteOK is 100% remote)
                    location = "Remote"

                    # Parse date
                    posted_date = None
                    if 'published' in entry:
                        try:
                            posted_date = datetime(*entry.published_parsed[:6])
                        except (TypeError, ValueError):
                            pass

                    job = JobListing(
                        title=title,
                        company=company,
                        location=location,
                        description=description,
                        apply_url=apply_url,
                        source="RemoteOK",
                        posted_date=posted_date
                    )
                    jobs.append(job)

                except Exception as e:
                    logger.warning(f"Failed to parse RemoteOK entry: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to fetch RemoteOK feed: {e}")

        return jobs


class WeWorkRemotelySource(JobSource):
    """Fetch jobs from We Work Remotely via RSS."""

    WWR_RSS = "https://weworkremotely.com/categories/remote-programming-jobs.rss"

    def fetch_jobs(self, limit: int = 50) -> List[JobListing]:
        """Fetch jobs from We Work Remotely."""
        jobs = []
        try:
            feed = feedparser.parse(self.WWR_RSS)

            for entry in feed.entries[:limit]:
                try:
                    title = entry.get('title', '')
                    company = entry.get('author', 'Unknown Company')
                    apply_url = entry.get('link', '')
                    description = entry.get('description', '')

                    location = "Remote"

                    # Parse date
                    posted_date = None
                    if 'published' in entry:
                        try:
                            posted_date = datetime(*entry.published_parsed[:6])
                        except (TypeError, ValueError):
                            pass

                    job = JobListing(
                        title=title,
                        company=company,
                        location=location,
                        description=description,
                        apply_url=apply_url,
                        source="We Work Remotely",
                        posted_date=posted_date
                    )
                    jobs.append(job)

                except Exception as e:
                    logger.warning(f"Failed to parse WWR entry: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to fetch WWR feed: {e}")

        return jobs
