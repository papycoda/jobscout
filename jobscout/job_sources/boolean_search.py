"""Boolean search via Serper (Google) for Greenhouse, Lever, Breezy HR, and Ashby HQ postings."""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import List, Set, Optional, Dict

import requests
from bs4 import BeautifulSoup

from .base import JobSource, JobListing


logger = logging.getLogger(__name__)

_SERPER_URL = "https://google.serper.dev/search"
_DEFAULT_TIMEOUT = 10
_GREENHOUSE_DOMAIN = "boards.greenhouse.io"
_LEVER_DOMAIN = "jobs.lever.co"
_BREEZYHR_DOMAIN = "jobs.breezy.hr"
_ASHBYHQ_DOMAIN = "jobs.ashbyhq.com"
_GREENHOUSE_JOB_RE = re.compile(r"https?://boards\.greenhouse\.io/[^/]+/jobs/\d+")
_LEVER_JOB_RE = re.compile(r"https?://jobs\.lever\.co/[^/]+/[^/?#]+")
_BREEZYHR_JOB_RE = re.compile(r"https?://jobs\.breezy\.hr/[^/]+/[^/?#]+")
_ASHBYHQ_JOB_RE = re.compile(r"https?://jobs\.ashbyhq\.com/[^/]+/[^/?#]+")


class BooleanSearchSource(JobSource):
    """
    Fetch jobs using Boolean search on public indexing.

    Searches for jobs on:
    - Greenhouse (boards.greenhouse.io)
    - Lever (jobs.lever.co)
    - Breezy HR (jobs.breezy.hr)
    - Ashby HQ (jobs.ashbyhq.com)

    Conservative approach: Only fetch URLs returned by search,
    never crawl company pages directly.
    """

    def __init__(
        self,
        resume_skills: Set[str],
        role_keywords: List[str],
        seniority: str,
        location_preference: str,
        max_job_age_days: int,
        serper_api_key: Optional[str],
        now: Optional[datetime] = None
    ):
        super().__init__("BooleanSearch")
        self.resume_skills = resume_skills
        self.role_keywords = role_keywords
        self.seniority = (seniority or "unknown").lower()
        self.location_preference = (location_preference or "any").lower()
        self.max_job_age_days = max_job_age_days
        self.serper_api_key = serper_api_key
        self._now_override = now

    def fetch_jobs(self, limit: int = 30) -> List[JobListing]:
        """
        Fetch jobs using Boolean search patterns.

        Hard cap at 30 results per run to avoid overwhelming processing.
        """
        if not self.serper_api_key:
            logger.info("Serper API key not configured; skipping boolean search.")
            return []

        logger.info(f"Skills: {self.resume_skills}")
        logger.info(f"Role keywords: {self.role_keywords}")

        # Build boolean queries (for logging/reference)
        queries = self._build_boolean_queries()
        if not queries:
            logger.info("No boolean queries generated; skipping.")
            return []
        for query in queries:
            logger.info(f"Boolean query: {query}")

        per_query_limit = max(1, limit // len(queries))
        results: List[JobListing] = []
        seen_links = set()

        for query in queries:
            for item in self._serper_search(query, per_query_limit):
                link = self._normalize_link(item.get("link"))
                if not link or link in seen_links:
                    continue
                if not self._is_supported_link(link):
                    continue
                seen_links.add(link)

                job = self._fetch_and_parse_job(link, item)
                if job:
                    results.append(job)

                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

        return results

    def _build_boolean_queries(self) -> List[str]:
        """
        Build Boolean search queries from resume data.

        Creates a single query searching all supported domains at once,
        like: site:(boards.greenhouse.io OR jobs.lever.co OR jobs.breezy.hr OR jobs.ashbyhq.com)
        """
        # Build skill OR clause
        skill_terms = sorted(self.resume_skills)[:8]  # Limit to top 8
        if not skill_terms:
            return []

        skill_clause = " OR ".join(self._quote_term(s) for s in skill_terms)

        # Build role clause
        role_terms = self.role_keywords[:3] if self.role_keywords else ["software engineer"]
        role_clause = " OR ".join(self._quote_term(term) for term in role_terms)

        location_clause = self._build_location_clause()
        exclude_clause = self._build_exclude_clause()
        after_clause = self._build_after_clause()

        # Build single query with all domains
        # Format: site:(domain1 OR domain2 OR domain3)
        all_domains = [_GREENHOUSE_DOMAIN, _LEVER_DOMAIN, _BREEZYHR_DOMAIN, _ASHBYHQ_DOMAIN]
        domain_clause = "site:(" + " OR ".join(all_domains) + ")"

        parts = [
            f"({role_clause})",
            f"({skill_clause})",
            domain_clause
        ]
        if location_clause:
            parts.append(location_clause)
        if after_clause:
            parts.append(after_clause)
        if exclude_clause:
            parts.append(exclude_clause)

        return [" ".join(parts)]  # Return single query

    def _serper_search(self, query: str, limit: int) -> List[Dict]:
        payload = {
            "q": query,
            "num": limit
        }
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
            "User-Agent": "JobScout/1.0"
        }
        try:
            response = requests.post(_SERPER_URL, json=payload, headers=headers, timeout=_DEFAULT_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(f"Serper search failed: {exc}")
            return []

        data = response.json()
        return data.get("organic", [])

    def _normalize_link(self, link: Optional[str]) -> Optional[str]:
        if not link:
            return None
        clean = link.split("#")[0]
        clean = clean.split("?")[0]
        return clean

    def _is_supported_link(self, link: str) -> bool:
        if _GREENHOUSE_DOMAIN in link:
            return bool(_GREENHOUSE_JOB_RE.search(link))
        if _LEVER_DOMAIN in link:
            return bool(_LEVER_JOB_RE.search(link))
        if _BREEZYHR_DOMAIN in link:
            return bool(_BREEZYHR_JOB_RE.search(link))
        if _ASHBYHQ_DOMAIN in link:
            return bool(_ASHBYHQ_JOB_RE.search(link))
        return False

    def _fetch_and_parse_job(self, link: str, item: Dict) -> Optional[JobListing]:
        html = self._fetch_page(link)
        if not html:
            return None

        result_date = self._parse_result_date(item.get("date") or item.get("publishedDate"))

        if _GREENHOUSE_DOMAIN in link:
            job = self._parse_greenhouse_job(html, link, result_date, item)
        elif _LEVER_DOMAIN in link:
            job = self._parse_lever_job(html, link, result_date, item)
        elif _BREEZYHR_DOMAIN in link:
            job = self._parse_breezy_job(html, link, result_date, item)
        elif _ASHBYHQ_DOMAIN in link:
            job = self._parse_ashby_job(html, link, result_date, item)
        else:
            return None

        if job and not job.posted_date:
            job.posted_date = result_date or self._now()

        return job

    def _fetch_page(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=_DEFAULT_TIMEOUT, headers={"User-Agent": "JobScout/1.0"})
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(f"Failed to fetch job page {url}: {exc}")
            return ""
        return response.text

    def _parse_greenhouse_job(
        self,
        html: str,
        url: str,
        posted_date: Optional[datetime],
        search_item: Optional[Dict]
    ) -> JobListing:
        """Parse job details from Greenhouse page."""
        soup = BeautifulSoup(html, "lxml")

        # Extract basic info
        title = soup.find("h1", class_="app-title")
        title = title.text.strip() if title else "Unknown Title"

        company = self._extract_company_name(soup, url, search_item)

        location = self._extract_greenhouse_location(soup)

        # Extract description
        content_div = soup.find("div", class_="content")
        description = content_div.get_text("\n", strip=True) if content_div else ""

        return JobListing(
            title=title,
            company=company,
            location=location,
            description=description,
            apply_url=url,
            source="Greenhouse",
            posted_date=posted_date
        )

    def _parse_lever_job(
        self,
        html: str,
        url: str,
        posted_date: Optional[datetime],
        search_item: Optional[Dict]
    ) -> JobListing:
        """Parse job details from Lever page."""
        soup = BeautifulSoup(html, "lxml")

        # Extract basic info
        title = soup.find("h2", class_="posting-title")
        title = title.text.strip() if title else "Unknown Title"

        company = self._extract_company_name(soup, url, search_item)

        location = self._extract_lever_location(soup)

        # Extract description
        content_div = soup.find("div", class_="posting-section")
        description = content_div.get_text("\n", strip=True) if content_div else ""

        return JobListing(
            title=title,
            company=company,
            location=location,
            description=description,
            apply_url=url,
            source="Lever",
            posted_date=posted_date
        )

    def _parse_breezy_job(
        self,
        html: str,
        url: str,
        posted_date: Optional[datetime],
        search_item: Optional[Dict]
    ) -> JobListing:
        """Parse job details from Breezy HR page."""
        soup = BeautifulSoup(html, "lxml")

        # Extract basic info - Breezy HR uses different classes
        title = soup.find("h1", class_="position-title")
        if not title:
            title = soup.find("h1")
        title = title.text.strip() if title else "Unknown Title"

        company = self._extract_company_name(soup, url, search_item)

        # Extract location - Breezy HR specific selectors
        location = ""
        location_el = soup.find("span", class_="location")
        if not location_el:
            location_el = soup.find("div", class_="location")
        if location_el:
            location = location_el.get_text(strip=True)
        if not location:
            location = self._default_location()

        # Extract description
        content_div = soup.find("div", class_="position-description")
        if not content_div:
            content_div = soup.find("div", class_="description")
        description = content_div.get_text("\n", strip=True) if content_div else ""

        return JobListing(
            title=title,
            company=company,
            location=location,
            description=description,
            apply_url=url,
            source="Breezy HR",
            posted_date=posted_date
        )

    def _parse_ashby_job(
        self,
        html: str,
        url: str,
        posted_date: Optional[datetime],
        search_item: Optional[Dict]
    ) -> JobListing:
        """Parse job details from Ashby HQ page."""
        soup = BeautifulSoup(html, "lxml")

        # Extract basic info - Ashby HQ uses different classes
        title = soup.find("h1")
        if title:
            title = title.text.strip()
        if not title:
            title = "Unknown Title"

        company = self._extract_company_name(soup, url, search_item)

        # Extract location - Ashby HQ specific selectors
        location = ""
        location_el = soup.find("div", class_="location")
        if not location_el:
            # Try to find location in other common patterns
            location_el = soup.find(string=re.compile(r"remote|location|office", re.I))
        if location_el and location_el.parent:
            location = location_el.parent.get_text(strip=True)
        if not location:
            location = self._default_location()

        # Extract description
        content_div = soup.find("div", class_="description")
        if not content_div:
            content_div = soup.find("div", class_="job-description")
        if not content_div:
            content_div = soup.find("article")
        description = content_div.get_text("\n", strip=True) if content_div else ""

        return JobListing(
            title=title,
            company=company,
            location=location,
            description=description,
            apply_url=url,
            source="Ashby HQ",
            posted_date=posted_date
        )

    def _extract_company_name(
        self,
        soup: BeautifulSoup,
        url: str,
        search_item: Optional[Dict]
    ) -> str:
        company = self._company_from_structured_data(soup)
        if not company:
            company = self._company_from_meta(soup)
        if not company:
            company = self._company_from_page(soup)
        if not company and search_item:
            company = self._company_from_search_item(search_item)
        if not company:
            company = self._company_from_url(url)
        return company or "Unknown Company"

    def _company_from_structured_data(self, soup: BeautifulSoup) -> str:
        for script in soup.find_all("script", type="application/ld+json"):
            payload = script.string or script.get_text(strip=True)
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except (TypeError, ValueError):
                continue
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                org = item.get("hiringOrganization") or item.get("organization")
                if isinstance(org, dict):
                    name = org.get("name")
                    if name:
                        return name.strip()
        return ""

    def _company_from_meta(self, soup: BeautifulSoup) -> str:
        meta = soup.find("meta", property="og:site_name") or soup.find("meta", attrs={"name": "application-name"})
        if meta and meta.get("content"):
            return meta["content"].strip()
        return ""

    def _company_from_page(self, soup: BeautifulSoup) -> str:
        selectors = [
            ".company-name",
            ".posting-company",
            ".posting-headline .company",
            ".app-title + .company",
            ".app-company",
        ]
        for selector in selectors:
            element = soup.select_one(selector)
            if element and element.get_text(strip=True):
                return element.get_text(strip=True)

        title_tag = soup.title.get_text(strip=True) if soup.title else ""
        return self._company_from_title(title_tag)

    def _company_from_search_item(self, item: Dict) -> str:
        title = item.get("title") or ""
        return self._company_from_title(title)

    def _company_from_title(self, title: str) -> str:
        if not title:
            return ""
        # Remove platform suffixes like " - Greenhouse", " | Lever", " - Breezy HR", " - Ashby HQ", etc.
        cleaned = re.sub(r"\s*[-|]\s*(Greenhouse|Lever|Breezy(\s+HR)?|Ashby(\s+HQ)?)(\s+Jobs|\s+Careers)?$", "", title, flags=re.I).strip()

        at_match = re.split(r"\s+at\s+", cleaned, flags=re.I)
        if len(at_match) > 1:
            candidate = at_match[-1].strip()
            if candidate:
                return candidate

        if ":" in cleaned:
            candidate = cleaned.split(":", 1)[0].strip()
            if 1 <= len(candidate.split()) <= 5:
                return candidate

        if " - " in cleaned:
            left, right = cleaned.split(" - ", 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                if len(left.split()) <= len(right.split()):
                    return left
                if re.search(r"\b(inc|llc|ltd|corp|gmbh|plc|co)\b", right, re.I):
                    return right

        return ""

    def _company_from_url(self, url: str) -> str:
        company_match = re.search(r"boards\.greenhouse\.io/([^/]+)/", url)
        if not company_match:
            company_match = re.search(r"jobs\.lever\.co/([^/]+)/", url)
        if not company_match:
            company_match = re.search(r"jobs\.breezy\.hr/([^/]+)/", url)
        if not company_match:
            company_match = re.search(r"jobs\.ashbyhq\.com/([^/]+)/", url)
        if not company_match:
            return ""
        company = company_match.group(1)
        company = company.replace("greenhouse", "").replace("lever", "").replace("breezy", "").replace("ashby", "")
        return company.replace("-", " ").strip().title()

    def _extract_greenhouse_location(self, soup: BeautifulSoup) -> str:
        location = ""
        location_el = soup.find("div", class_="location") or soup.find("div", class_="app-location")
        if location_el:
            location = location_el.get_text(strip=True)
        return location or self._default_location()

    def _extract_lever_location(self, soup: BeautifulSoup) -> str:
        location = ""
        categories = soup.find("div", class_="posting-categories")
        if categories:
            location_el = categories.find("span", class_="sort-by-time posting-category")
            if not location_el:
                location_el = categories.find("span", class_="posting-category")
            if location_el:
                location = location_el.get_text(strip=True)
        return location or self._default_location()

    def _default_location(self) -> str:
        if self.location_preference in {"remote", "hybrid"}:
            return "Remote"
        if self.location_preference == "onsite":
            return "Onsite"
        return "Unknown"

    def _build_location_clause(self) -> str:
        if self.location_preference == "remote":
            return '"remote"'
        if self.location_preference == "hybrid":
            return '("hybrid" OR "remote")'
        if self.location_preference == "onsite":
            return '-"remote" -"hybrid"'
        return ""

    def _build_exclude_clause(self) -> str:
        exclude = []
        if self.seniority == "junior":
            exclude = ["senior", "staff", "principal", "lead", "manager"]
        elif self.seniority == "mid":
            exclude = ["staff", "principal", "lead"]
        elif self.seniority == "senior":
            exclude = ["staff", "principal"]
        return " ".join(f'-"{term}"' for term in exclude)

    def _build_after_clause(self) -> str:
        if self.max_job_age_days < 1:
            return ""
        cutoff = self._now().date() - timedelta(days=self.max_job_age_days)
        return f"after:{cutoff.isoformat()}"

    def _quote_term(self, term: str) -> str:
        cleaned = term.strip().strip('"')
        return f'"{cleaned}"'

    def _parse_result_date(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None

        value = value.strip()
        now = self._now()

        if value.lower() == "yesterday":
            return now - timedelta(days=1)

        ago_match = re.match(r"(\d+)\s+(day|hour|minute)s?\s+ago", value.lower())
        if ago_match:
            amount = int(ago_match.group(1))
            unit = ago_match.group(2)
            if unit == "day":
                return now - timedelta(days=amount)
            if unit == "hour":
                return now - timedelta(hours=amount)
            if unit == "minute":
                return now - timedelta(minutes=amount)

        for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d %b %Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

        return None

    def _now(self) -> datetime:
        return self._now_override or datetime.utcnow()
