"""Fetch full job description when RSS description is truncated."""

import logging
import re
import requests
from typing import Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Headers for web requests
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Site-specific selectors for full job descriptions
_DESCRIPTION_SELECTORS = {
    "remoteok.com": [
        "div.description",
        "div.job_description",
        "div[id^=job]",
        "article",
        "main",
    ],
    "weworkremotely.com": [
        "div.listing-container",
        "div.description",
        "section.listing-description",
        "article",
    ],
    "remotive.com": [
        "div.job-description",
        "div.description",
        "section.job",
        "article",
    ],
    "boards.greenhouse.io": [
        "div.content",
        "div.job-description",
        "div[id='content']",
        "main",
    ],
    "jobs.lever.co": [
        "div.posting-section",
        "div.posting-description",
        "section.description",
        "main",
    ],
}


def is_truncated(description: str, min_length: int = 500) -> bool:
    """
    Check if a job description appears to be truncated.

    Signs of truncation:
    - Very short (< min_length chars)
    - Ends with "..." or "&hellip;" or similar
    - Ends mid-sentence (no proper ending punctuation)
    - Contains "click to view more" or similar phrases

    Args:
        description: The job description text
        min_length: Minimum expected length for a full description

    Returns:
        True if description appears truncated
    """
    if not description:
        return True

    # Check length
    if len(description) < min_length:
        return True

    # Check for truncation markers
    truncated_endings = ["...", "&hellip;", "&#8230;", "…", " more", "read more", "click to", "view full"]
    description_lower = description.lower().strip()
    for ending in truncated_endings:
        if description_lower.endswith(ending):
            return True

    # Check for obvious truncation markers in content
    truncation_phrases = [
        "click to apply",
        "view full job",
        "read full description",
        "see full details",
        "apply now to view",
    ]
    for phrase in truncation_phrases:
        if phrase in description_lower:
            return True

    # Check if ends mid-sentence (no proper ending)
    # but only for descriptions that are long enough to matter
    if len(description) > min_length:
        last_char = description.strip()[-1]
        if last_char not in '.!?\'")]' and not last_char.isdigit():
            # Look for other signs this might be truncated
            # Check if there are section headers without content near the end
            lines = description_lower.split('\n')
            for line in lines[-3:]:  # Check last 3 lines
                line = line.strip()
                if any(header in line for header in [
                    'requirements:', 'qualifications:', 'what you\'ll bring:',
                    'you will bring:', 'you have:', 'responsibilities:',
                    'benefits:', 'perks:'
                ]):
                    # Section header found near end, likely truncated
                    return True

    return False


def fetch_full_description(url: str, current_description: str = "") -> Optional[str]:
    """
    Fetch the full job description from the job posting URL.

    Args:
        url: The job posting URL
        current_description: Current (possibly truncated) description

    Returns:
        Full description if successfully fetched, otherwise None
    """
    if not url:
        return None

    # Determine domain for selector selection
    domain = _extract_domain(url)
    selectors = _DESCRIPTION_SELECTORS.get(domain, ["article", "main", "div.job-description", "div.description"])

    try:
        response = requests.get(url, headers=_HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        # Try each selector
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                # Get the first matching element
                element = elements[0]
                text = element.get_text(separator='\n', strip=True)

                if text and len(text) > len(current_description):
                    logger.debug(f"Fetched full description from {url} using selector '{selector}': {len(text)} chars")
                    return text

        # Fallback: try to find any div with substantial text
        for div in soup.find_all('div'):
            text = div.get_text(separator='\n', strip=True)
            if text and len(text) > 1000:  # Substantial content
                # Check if it looks like job description
                text_lower = text.lower()
                if any(keyword in text_lower for keyword in [
                    'requirements', 'qualifications', 'responsibilities',
                    'skills', 'experience', 'you will', 'you have'
                ]):
                    logger.debug(f"Fetched full description from {url} using fallback div: {len(text)} chars")
                    return text

        logger.warning(f"Could not extract full description from {url}")
        return None

    except requests.RequestException as e:
        logger.warning(f"Failed to fetch full description from {url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error parsing HTML from {url}: {e}")
        return None


def _extract_domain(url: str) -> str:
    """Extract domain from URL for selector lookup."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def expand_truncated_jobs(jobs: list) -> list:
    """
    Expand truncated job descriptions by fetching full page content.

    Args:
        jobs: List of JobListing objects

    Returns:
        List of JobListing objects with expanded descriptions
    """
    from .base import JobListing

    expanded_jobs = []
    expanded_count = 0

    for job in jobs:
        description = job.description

        if is_truncated(description):
            logger.debug(f"Job '{job.title}' at {job.company} appears truncated, fetching full description")
            full_desc = fetch_full_description(job.apply_url, description)

            if full_desc:
                # Create new JobListing with expanded description
                expanded_job = JobListing(
                    title=job.title,
                    company=job.company,
                    location=job.location,
                    description=full_desc,
                    apply_url=job.apply_url,
                    source=job.source,
                    posted_date=job.posted_date
                )
                expanded_jobs.append(expanded_job)
                expanded_count += 1
            else:
                # Keep original if fetch failed
                expanded_jobs.append(job)
        else:
            expanded_jobs.append(job)

    if expanded_count > 0:
        logger.info(f"Expanded {expanded_count}/{len(jobs)} truncated job descriptions")

    return expanded_jobs
