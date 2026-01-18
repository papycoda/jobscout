"""RSS feed job sources (RemoteOK, We Work Remotely)."""

import feedparser
import logging
from datetime import datetime
from typing import List
from .base import JobSource, JobListing
from .truncation import expand_truncated_jobs


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

                    company = entry.get('author', '')
                    apply_url = entry.get('link', '')
                    description = entry.get('description', '')

                    # Extract company from URL if author field is empty
                    # RemoteOK URLs format: https://remoteok.com/remote-jobs/role-name-company-slug-ID
                    if not company and apply_url:
                        import re
                        # Extract company slug from URL (part before final dash-number)
                        # Example: .../remote-software-engineer-agent-infrastructure-openai-1129438
                        url_parts = apply_url.rstrip('/').split('/')
                        if len(url_parts) > 1:
                            job_slug = url_parts[-1]
                            # Remove the numeric ID at the end
                            slug_parts = job_slug.rsplit('-', 1)
                            if len(slug_parts) == 2 and slug_parts[1].isdigit():
                                # Get everything after "remote-" and before the ID
                                company_slug = slug_parts[0]
                                # Remove common prefixes like "remote-", "senior-", etc.
                                for prefix in ['remote-', 'senior-', 'junior-', 'lead-', 'principal-', 'staff-']:
                                    if company_slug.startswith(prefix):
                                        company_slug = company_slug[len(prefix)::]
                                        break
                                # Remove role keywords that sometimes appear in the slug
                                role_keywords = ['engineer', 'developer', 'manager', 'architect', 'designer', 'qa-', 'devops']
                                for keyword in role_keywords:
                                    # Try to find the company name after these keywords
                                    pattern = f'-{keyword}-'
                                    if pattern in company_slug.lower():
                                        parts = company_slug.lower().split(pattern)
                                        if len(parts) > 1:
                                            company_slug = parts[-1]
                                        break
                                # If we still have multiple parts, try to be smarter about extraction
                                # Keywords that often appear before company names in the slug
                                pre_company_keywords = ['team', 'platform', 'infrastructure', 'product', 'data', 'backend', 'frontend', 'full-stack']
                                for keyword in pre_company_keywords:
                                    pattern = f'-{keyword}-'
                                    if pattern in company_slug.lower():
                                        parts = company_slug.lower().split(pattern)
                                        if len(parts) > 1:
                                            company_slug = parts[-1]
                                            break

                                # Check for company suffixes before doing aggressive trimming
                                company_suffixes = ['inc', 'llc', 'ltd', 'corp', 'corporation', 'co', 'gmbh', 'pty', 'io', 'ai']
                                slug_lower = company_slug.lower()
                                has_suffix = any(suffix in slug_lower.split('-')[-1] for suffix in company_suffixes)

                                # If still multiple parts and no clear company suffix, be aggressive
                                if '-' in company_slug and not has_suffix:
                                    parts = company_slug.split('-')
                                    if len(parts) > 2:
                                        # Be more aggressive - take last 1-2 segments only
                                        company_slug = '-'.join(parts[-1:] if len(parts) <= 3 else parts[-2:])

                                # Clean up and format
                                company = company_slug.replace('-', ' ').strip().title()

                    if not company:
                        company = 'Unknown Company'

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

        # Expand truncated job descriptions
        jobs = expand_truncated_jobs(jobs)
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

        # Expand truncated job descriptions
        jobs = expand_truncated_jobs(jobs)
        return jobs


class HimalayasSource(JobSource):
    """Fetch jobs from Himalayas (remote startup jobs)."""

    HIMALAYAS_RSS = "https://himalayas.app/jobs/rss"

    def fetch_jobs(self, limit: int = 20) -> List[JobListing]:
        """Fetch jobs from Himalayas."""
        jobs = []
        try:
            feed = feedparser.parse(self.HIMALAYAS_RSS)

            for entry in feed.entries[:limit]:
                try:
                    title = entry.get('title', '')

                    company = entry.get('author', '')
                    if not company:
                        # Try to extract company from title
                        parts = title.split(' at ')
                        if len(parts) > 1:
                            company = parts[-1].strip()
                        else:
                            company = 'Unknown Company'

                    apply_url = entry.get('link', '')
                    description = entry.get('description', '')

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
                        location="Remote",
                        description=description,
                        apply_url=apply_url,
                        source="Himalayas",
                        posted_date=posted_date
                    )
                    jobs.append(job)

                except Exception as e:
                    logger.warning(f"Failed to parse Himalayas entry: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to fetch Himalayas feed: {e}")

        return jobs


class JavascriptJobsSource(JobSource):
    """Fetch jobs from JavaScriptJobs RSS feed."""

    JSJOBS_RSS = "https://www.jsjobs.net/feed"

    def fetch_jobs(self, limit: int = 15) -> List[JobListing]:
        """Fetch jobs from JavaScriptJobs."""
        jobs = []
        try:
            feed = feedparser.parse(self.JSJOBS_RSS)

            for entry in feed.entries[:limit]:
                try:
                    title = entry.get('title', '')

                    company = entry.get('author', 'Unknown Company')
                    apply_url = entry.get('link', '')
                    description = entry.get('description', '')

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
                        location="Remote",
                        description=description,
                        apply_url=apply_url,
                        source="JavaScriptJobs",
                        posted_date=posted_date
                    )
                    jobs.append(job)

                except Exception as e:
                    logger.warning(f"Failed to parse JavaScriptJobs entry: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to fetch JavaScriptJobs feed: {e}")

        return jobs
