"""Direct scraping of company job boards (Greenhouse, Lever, Ashby) without API keys.

This source fetches jobs directly from known company job boards. It maintains a registry
of popular tech companies and scrapes their public job posting pages.

No API keys required - just HTTP requests to public endpoints.
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import List, Set, Optional, Dict

import requests
from bs4 import BeautifulSoup

from .base import JobSource, JobListing, strip_html_tags


logger = logging.getLogger(__name__)

# Popular tech companies with public Greenhouse boards
GREENHOUSE_COMPANIES = [
    "airbnb", "stripe", "shopify", "doordash", "instacart", "uber",
    "lyft", "slack", "figma", "notion", "linear", "vercel", "supabase",
    "openai", "anthropic", "scale", "retool", "cron", "raycast",
    "mercury", "brex", "plaid", "coinbase", "rainforest",
    "1password", "okta", "snyk", "hashicorp", "docker",
    "mongodb", "redis", "confluent", "databricks", "snowflake",
    "fivetran", "airbyte", "segment", "mixpanel", "amplitude",
    "webflow", "squarespace", "wix", "shopify", "bigcommerce",
    "zendesk", "intercom", "drift", "gong", "chili-piper",
    "sprout-social", "hootsuite", "buffer", "mailchimp", "sendgrid",
    "twilio", "mailgun", "postmark", "sparkpost",
]

# Popular tech companies with public Lever boards
LEVER_COMPANIES = [
    "netflix", "spotify", "amazon", "adobe", "microsoft", "meta",
    "doordash", "instacart", "robinhood", "coinbase", "coinbase-corporate",
    "better", "affirm", "klarna", "block", "square",
    "quora", "reddit", "pinterest", "twitter", "yext",
    " Dropbox", "box", "egnyte", "zoom", "slack",
    "palantir", "anduril", "scale-ai", "shield-ai", "skydio",
]

# Companies with Ashby HQ boards
ASHBY_COMPANIES = [
    "ramp", "mercury", "raft", "cipher", "stytch", "close",
    "solid-ai", "keyword", "canva", "figma", "notion",
]

_TIMEOUT = 10


class CompanyBoardsSource(JobSource):
    """
    Fetch jobs directly from company job boards (Greenhouse, Lever, Ashby).

    This is a FREE alternative to boolean search APIs. It scrapes known company
    job boards directly using their public endpoints.
    """

    def __init__(
        self,
        resume_skills: Set[str],
        role_keywords: List[str],
        location_preference: str = "remote",
        max_job_age_days: int = 7,
        companies: Optional[List[str]] = None,
    ):
        super().__init__("CompanyBoards")
        self.resume_skills = resume_skills
        self.role_keywords = role_keywords
        self.location_preference = location_preference.lower()
        self.max_job_age_days = max_job_age_days

        # Use provided companies or defaults
        self.companies = companies or []

    def fetch_jobs(self, limit: int = 50) -> List[JobListing]:
        """Fetch jobs from all registered company boards."""
        all_jobs = []
        seen_urls = set()

        # Build company list from defaults + any user-specified companies
        companies_to_fetch = self._get_companies_to_fetch()

        logger.info(f"Fetching from {len(companies_to_fetch)} company boards")

        for company_domain in companies_to_fetch:
            if len(all_jobs) >= limit:
                break

            jobs = self._fetch_company_board(company_domain, seen_urls)
            all_jobs.extend(jobs)
            seen_urls.update(j.apply_url for j in jobs)

            # Rate limiting between companies
            if len(all_jobs) < limit:
                time.sleep(0.5)

        # Sort by posted date (newest first) and limit
        all_jobs.sort(key=lambda j: j.posted_date or datetime.min, reverse=True)
        return all_jobs[:limit]

    def _get_companies_to_fetch(self) -> List[str]:
        """Get list of company domains to fetch, categorizing by platform."""
        companies = []

        # Add user-specified companies
        if self.companies:
            companies.extend(self.companies)

        # Otherwise, use curated lists (limited to avoid overwhelming)
        if not companies:
            # Sample of high-quality tech companies
            companies = GREENHOUSE_COMPANIES[:20] + LEVER_COMPANIES[:10] + ASHBY_COMPANIES[:5]

        return companies

    def _fetch_company_board(self, company: str, seen_urls: Set[str]) -> List[JobListing]:
        """Fetch jobs from a single company's board."""
        # Detect board type and fetch accordingly
        if self._is_greenhouse_company(company):
            return self._fetch_greenhouse_company(company, seen_urls)
        elif self._is_lever_company(company):
            return self._fetch_lever_company(company, seen_urls)
        elif self._is_ashby_company(company):
            return self._fetch_ashby_company(company, seen_urls)
        else:
            # Try to auto-detect
            return self._try_auto_detect(company, seen_urls)

    def _is_greenhouse_company(self, company: str) -> bool:
        """Check if company uses Greenhouse."""
        return company in GREENHOUSE_COMPANIES or '.' in company  # Allow custom domains

    def _is_lever_company(self, company: str) -> bool:
        """Check if company uses Lever."""
        return company in LEVER_COMPANIES

    def _is_ashby_company(self, company: str) -> bool:
        """Check if company uses Ashby HQ."""
        return company in ASHBY_COMPANIES

    def _fetch_greenhouse_company(self, company: str, seen_urls: Set[str]) -> List[JobListing]:
        """Fetch jobs from a Greenhouse board."""
        jobs = []

        # Try JSON API first (faster, more reliable)
        url = f"https://boards.greenhouse.io/{company}/jobs"
        json_url = f"{url}.json"

        try:
            response = requests.get(json_url, timeout=_TIMEOUT, headers={"User-Agent": "JobScout/1.0"})
            if response.status_code == 200:
                data = response.json()
                jobs = self._parse_greenhouse_json(data, company, seen_urls)
            else:
                # Fall back to HTML scraping
                response = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "JobScout/1.0"})
                if response.status_code == 200:
                    jobs = self._parse_greenhouse_html(response.text, company, seen_urls)
        except requests.RequestException as e:
            logger.debug(f"Failed to fetch Greenhouse board for {company}: {e}")

        return jobs

    def _fetch_lever_company(self, company: str, seen_urls: Set[str]) -> List[JobListing]:
        """Fetch jobs from a Lever board."""
        jobs = []

        # Lever has a nice JSON API
        url = f"https://jobs.lever.co/v1/postings/{company}?group=team&mode=json"

        try:
            response = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "JobScout/1.0"})
            if response.status_code == 200:
                data = response.json()
                jobs = self._parse_lever_json(data, company, seen_urls)
        except requests.RequestException as e:
            logger.debug(f"Failed to fetch Lever board for {company}: {e}")

        return jobs

    def _fetch_ashby_company(self, company: str, seen_urls: Set[str]) -> List[JobListing]:
        """Fetch jobs from an Ashby HQ board."""
        jobs = []

        # Ashby has a JSON API
        url = f"https://jobs.ashbyhq.com/api/{company}/jobs"

        try:
            response = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "JobScout/1.0"})
            if response.status_code == 200:
                data = response.json()
                jobs = self._parse_ashby_json(data, company, seen_urls)
        except requests.RequestException as e:
            logger.debug(f"Failed to fetch Ashby board for {company}: {e}")

        return jobs

    def _try_auto_detect(self, company: str, seen_urls: Set[str]) -> List[JobListing]:
        """Try to auto-detect which platform the company uses."""
        # Try Greenhouse first (most common)
        jobs = self._fetch_greenhouse_company(company, seen_urls)
        if jobs:
            return jobs

        # Try Lever
        jobs = self._fetch_lever_company(company, seen_urls)
        if jobs:
            return jobs

        # Try Ashby
        return self._fetch_ashby_company(company, seen_urls)

    def _parse_greenhouse_json(self, data: dict, company: str, seen_urls: Set[str]) -> List[JobListing]:
        """Parse Greenhouse JSON response."""
        jobs = []

        for job in data.get("jobs", []):
            try:
                # Apply filters early
                if not self._is_location_match(job.get("location", {})):
                    continue

                posted_date = self._parse_greenhouse_date(job)
                if not self._is_age_valid(posted_date):
                    continue

                apply_url = job.get("absolute_url")
                if not apply_url or apply_url in seen_urls:
                    continue

                # Check if job matches skills/roles (quick filter)
                title = job.get("title", "")
                if not self._is_relevant_job(title):
                    continue

                job_listing = JobListing(
                    title=title,
                    company=job.get("metadata", [{}])[0].get("name") if job.get("metadata") else company.title(),
                    location=self._format_location(job.get("location", {})),
                    description=strip_html_tags(job.get("content") or job.get("description", "")),
                    apply_url=apply_url,
                    source="Greenhouse",
                    posted_date=posted_date
                )
                jobs.append(job_listing)

            except Exception as e:
                logger.debug(f"Failed to parse Greenhouse job: {e}")
                continue

        return jobs

    def _parse_greenhouse_html(self, html: str, company: str, seen_urls: Set[str]) -> List[JobListing]:
        """Parse Greenhouse HTML response (fallback)."""
        jobs = []
        soup = BeautifulSoup(html, "lxml")

        for job_div in soup.select(".opening"):
            try:
                title = job_div.find("a", class_="opening-title")
                if not title:
                    continue

                title_text = title.text.strip()
                if not self._is_relevant_job(title_text):
                    continue

                apply_url = title.get("href")
                if not apply_url:
                    continue

                if apply_url.startswith("/"):
                    apply_url = f"https://boards.greenhouse.io{apply_url}"

                if apply_url in seen_urls:
                    continue

                location = job_div.find("span", class_="location")
                location_text = location.text.strip() if location else "Unknown"

                if not self._is_location_match(location_text):
                    continue

                # Get job details page
                posted_date = self._fetch_greenhouse_job_date(apply_url)
                if not self._is_age_valid(posted_date):
                    continue

                jobs.append(JobListing(
                    title=title_text,
                    company=company.title(),
                    location=location_text,
                    description="",
                    apply_url=apply_url,
                    source="Greenhouse",
                    posted_date=posted_date
                ))

            except Exception as e:
                logger.debug(f"Failed to parse Greenhouse HTML job: {e}")
                continue

        return jobs

    def _parse_lever_json(self, data: list, company: str, seen_urls: Set[str]) -> List[JobListing]:
        """Parse Lever JSON response."""
        jobs = []

        for posting in data:
            try:
                if posting.get("workflowState", "") == "closed":
                    continue

                if not self._is_location_match(posting.get("categories", {}).get("location", "")):
                    continue

                posted_date = self._parse_lever_date(posting)
                if not self._is_age_valid(posted_date):
                    continue

                apply_url = posting.get("hostedUrl")
                if not apply_url or apply_url in seen_urls:
                    continue

                title = posting.get("title")
                if not title or not self._is_relevant_job(title):
                    continue

                # Additional description may require fetching the job page
                description = strip_html_tags(posting.get("description", "") or posting.get("content", ""))

                jobs.append(JobListing(
                    title=title,
                    company=posting.get("company", company.title()),
                    location=self._format_location(posting.get("categories", {})),
                    description=description,
                    apply_url=apply_url,
                    source="Lever",
                    posted_date=posted_date
                ))

            except Exception as e:
                logger.debug(f"Failed to parse Lever job: {e}")
                continue

        return jobs

    def _parse_ashby_json(self, data: dict, company: str, seen_urls: Set[str]) -> List[JobListing]:
        """Parse Ashby HQ JSON response."""
        jobs = []

        for job in data.get("jobs", []):
            try:
                if job.get("status") != "ACTIVE":
                    continue

                location_compound = job.get("locationCompound", "")
                if not self._is_location_match(location_compound):
                    continue

                posted_date = self._parse_iso_date(job.get("publishedDate"))
                if not self._is_age_valid(posted_date):
                    continue

                apply_url = job.get("canonicalUrl")
                if not apply_url or apply_url in seen_urls:
                    continue

                title = job.get("title")
                if not title or not self._is_relevant_job(title):
                    continue

                description = strip_html_tags(job.get("descriptionPlain") or job.get("description", ""))

                jobs.append(JobListing(
                    title=title,
                    company=job.get("companyName", company.title()),
                    location=location_compound,
                    description=description,
                    apply_url=apply_url,
                    source="Ashby HQ",
                    posted_date=posted_date
                ))

            except Exception as e:
                logger.debug(f"Failed to parse Ashby job: {e}")
                continue

        return jobs

    def _fetch_greenhouse_job_date(self, job_url: str) -> Optional[datetime]:
        """Fetch posted date from Greenhouse job detail page."""
        try:
            response = requests.get(job_url, timeout=5, headers={"User-Agent": "JobScout/1.0"})
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")

                # Greenhouse sometimes puts date in meta tags
                meta_date = soup.find("meta", property="article:published_time")
                if meta_date and meta_date.get("content"):
                    return self._parse_iso_date(meta_date["content"])

                # Or in the posting details
                details_div = soup.find("div", class_="posting-details")
                if details_div:
                    # Look for date patterns
                    text = details_div.get_text()
                    date_match = re.search(r"(\w+ \d+, \d{4})", text)
                    if date_match:
                        return self._parse_us_date(date_match.group(1))
        except Exception:
            pass
        return None

    def _is_relevant_job(self, title: str) -> bool:
        """Quick filter: check if job title matches user's skills/roles."""
        title_lower = title.lower()

        # Exclude non-tech roles
        exclude_keywords = [
            "design", "marketing", "sales", "support", "customer success",
            "account executive", "recruiter", "hr", "people", "finance",
            "legal", "operations", "product manager", "data entry"
        ]

        for keyword in exclude_keywords:
            if keyword in title_lower:
                return False

        # Check if title contains any of our skills or role keywords
        for skill in self.resume_skills:
            if skill.lower() in title_lower:
                return True

        for role in self.role_keywords[:3]:
            if role.lower() in title_lower:
                return True

        # Accept generic engineer/developer roles
        generic_accept = ["engineer", "developer", "programmer", "technical"]
        return any(keyword in title_lower for keyword in generic_accept)

    def _is_location_match(self, location) -> bool:
        """Check if job location matches user preference."""
        if not location:
            return False

        # Handle dict format (from Greenhouse/Lever APIs)
        if isinstance(location, dict):
            location_str = " ".join(str(v) for v in location.values() if v).lower()
        else:
            location_str = str(location).lower()

        # User wants remote - accept remote or any location
        if self.location_preference == "remote":
            return (
                "remote" in location_str or
                "anywhere" in location_str or
                "distributed" in location_str
            )

        # User wants hybrid - accept remote, hybrid, or local
        if self.location_preference == "hybrid":
            return (
                "remote" in location_str or
                "hybrid" in location_str or
                "anywhere" in location_str
            )

        # User wants onsite - exclude remote
        if self.location_preference == "onsite":
            return "remote" not in location_str and "anywhere" not in location_str

        # User doesn't care
        return True

    def _is_age_valid(self, posted_date: Optional[datetime]) -> bool:
        """Check if job is within age limit.

        IMPORTANT: Jobs without dates are REJECTED to avoid old jobs passing through.
        """
        if not posted_date:
            # If we can't determine the date, reject the job
            # This prevents old jobs from slipping through
            return False

        # Ensure we're comparing timezone-aware datetimes
        now = datetime.now(timezone.utc)
        if posted_date.tzinfo is None:
            # Assume posted_date is UTC if no timezone
            posted_date = posted_date.replace(tzinfo=timezone.utc)

        age = now - posted_date
        return age.days <= self.max_job_age_days

    def _format_location(self, location) -> str:
        """Format location dict/object to string."""
        if isinstance(location, dict):
            parts = [str(v) for v in location.values() if v]
            return ", ".join(parts) if parts else "Remote"
        return str(location) if location else "Remote"

    def _parse_greenhouse_date(self, job: dict) -> Optional[datetime]:
        """Parse date from Greenhouse job object."""
        # Greenhouse sometimes has updateAtAt or createdAt
        for date_key in ["updated_at", "createdAt", "updatedAt"]:
            if date_key in job:
                return self._parse_iso_date(job[date_key])
        return None

    def _parse_lever_date(self, posting: dict) -> Optional[datetime]:
        """Parse date from Lever posting object."""
        # Lever uses createdAt
        return self._parse_iso_date(posting.get("createdAt"))

    def _parse_iso_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO date string."""
        if not date_str:
            return None

        try:
            # Handle various ISO formats
            date_str = date_str.replace("Z", "+00:00")
            if "." in date_str:
                # Split on the decimal point and take the timezone part
                parts = date_str.split(".")
                if len(parts) == 2:
                    # Check if the second part has timezone info (last 6 chars like +00:00)
                    timezone_part = parts[1][-6:] if len(parts[1]) >= 6 and parts[1][-3] == ":" else ""
                    date_str = parts[0] + timezone_part

            return datetime.fromisoformat(date_str)
        except (ValueError, AttributeError):
            return None

    def _parse_us_date(self, date_str: str) -> Optional[datetime]:
        """Parse US format date like 'December 25, 2024'."""
        try:
            from datetime import datetime
            return datetime.strptime(date_str, "%B %d, %Y")
        except ValueError:
            return None


def discover_company_board(company_name: str) -> Optional[str]:
    """
    Discover which job board platform a company uses.

    Returns the URL of the company's job board, or None if not found.
    """
    # Try Greenhouse
    try:
        url = f"https://boards.greenhouse.io/{company_name}/jobs"
        response = requests.head(url, timeout=5, headers={"User-Agent": "JobScout/1.0"})
        if response.status_code == 200:
            return url
    except requests.RequestException:
        pass

    # Try Lever
    try:
        url = f"https://jobs.lever.co/{company_name}"
        response = requests.head(url, timeout=5, headers={"User-Agent": "JobScout/1.0"})
        if response.status_code == 200:
            return url
    except requests.RequestException:
        pass

    # Try Ashby
    try:
        url = f"https://jobs.ashbyhq.com/{company_name}"
        response = requests.head(url, timeout=5, headers={"User-Agent": "JobScout/1.0"})
        if response.status_code == 200:
            return url
    except requests.RequestException:
        pass

    return None
