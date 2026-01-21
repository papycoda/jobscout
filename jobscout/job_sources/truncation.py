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
# Ordered from most specific to most general
_DESCRIPTION_SELECTORS = {
    "remoteok.com": [
        "#job",
        ".description",
        "div.description",
        "div.job_description",
        "div[id^=job]",
        "article",
        "main",
    ],
    "weworkremotely.com": [
        ".listing",
        ".listing-container",
        "div.listing-container",
        ".description",
        "div.description",
        "section.listing-description",
        "article",
    ],
    "remotive.com": [
        ".job-description",
        "div.job-description",
        ".description",
        "div.description",
        "section.job",
        "article",
    ],
    "himalayas.app": [
        "[class*='description']",
        "[class*='job']",
        "article",
        "main",
    ],
    "boards.greenhouse.io": [
        ".content",
        "div.content",
        ".job-description",
        "div.job-description",
        "#content",
        "main",
    ],
    "jobs.lever.co": [
        ".posting-section",
        "div.posting-section",
        ".posting-description",
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
    selectors = _DESCRIPTION_SELECTORS.get(domain, [
        "#job",
        "article",
        "main",
        "div.job-description",
        "div.description",
        ".description",
        ".job-description",
        "section",
    ])

    try:
        response = requests.get(url, headers=_HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        best_text = None
        best_length = len(current_description) if current_description else 0

        # Try each selector
        for selector in selectors:
            try:
                elements = soup.select(selector)
                if elements:
                    # Get the first matching element
                    element = elements[0]
                    text = _clean_extracted_text(element)

                    if text and len(text) > best_length:
                        best_text = text
                        best_length = len(text)
                        logger.debug(f"Fetched description from {url} using '{selector}': {len(text)} chars")
            except Exception:
                continue

        # Enhanced fallback: try to find the largest text block that looks like a job description
        if not best_text or best_length < 500:
            best_text = _fallback_find_description(soup, current_description, best_text or "")
            if best_text:
                best_length = len(best_text)

        if best_text and best_length > len(current_description):
            return best_text

        logger.warning(f"Could not extract better description from {url} (current: {len(current_description)}, best: {best_length})")
        return None

    except requests.RequestException as e:
        logger.warning(f"Failed to fetch full description from {url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error parsing HTML from {url}: {e}")
        return None


def _clean_extracted_text(element) -> str:
    """Clean and extract text from a BeautifulSoup element."""
    # Remove script and style elements
    for script in element(["script", "style", "nav", "header", "footer", "aside"]):
        script.decompose()

    text = element.get_text(separator='\n', strip=True)

    # Clean up extra whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)


def _fallback_find_description(soup, current_description: str, current_best: str) -> Optional[str]:
    """
    Fallback method to find job description by analyzing all text blocks.

    Looks for the largest text block that contains job-related keywords.
    """
    job_keywords = [
        'requirements', 'qualifications', 'responsibilities',
        'skills', 'experience', 'you will', 'you have',
        'what you', 'about the', 'we are looking',
        'role', 'position', 'team', 'company'
    ]

    candidates = []

    # Check all common container types
    for tag in ['div', 'section', 'article', 'main']:
        for element in soup.find_all(tag, limit=50):  # Limit to avoid excessive processing
            text = _clean_extracted_text(element)

            # Skip if too short or too long (likely not the job description)
            if len(text) < 200 or len(text) > 20000:
                continue

            # Check for job-related keywords
            text_lower = text.lower()
            keyword_count = sum(1 for kw in job_keywords if kw in text_lower)

            # Skip if no job keywords found
            if keyword_count == 0:
                continue

            candidates.append((text, keyword_count, len(text)))

    # Sort by keyword count (desc), then by length (desc)
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

    if candidates:
        best = candidates[0][0]
        logger.debug(f"Fallback found description with {candidates[0][1]} keywords: {len(best)} chars")
        return best

    return current_best if len(current_best) > len(current_description) else None


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

    This function is more aggressive than just checking for truncation -
    it will attempt to fetch full descriptions for any job that has:
    - A clearly truncated description (ends with ..., section header at end, etc.)
    - A short description (< 1000 chars) that might be insufficient for proper matching

    Args:
        jobs: List of JobListing objects

    Returns:
        List of JobListing objects with expanded descriptions
    """
    from .base import JobListing

    expanded_jobs = []
    expanded_count = 0
    attempted_count = 0

    for job in jobs:
        description = job.description
        should_fetch = False

        # Check if clearly truncated (using lower threshold)
        if is_truncated(description, min_length=300):
            should_fetch = True
            logger.debug(f"Job '{job.title}' at {job.company} appears truncated")
        # Also try to fetch if description is short but might be just an RSS summary
        # (Under 1000 chars is likely insufficient for good skill matching)
        elif len(description) < 1000:
            should_fetch = True
            logger.debug(f"Job '{job.title}' at {job.company} has short description ({len(description)} chars), attempting expansion")

        if should_fetch:
            attempted_count += 1
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
                logger.info(f"Expanded '{job.title}': {len(description)} → {len(full_desc)} chars")
            else:
                # Keep original if fetch failed
                expanded_jobs.append(job)
        else:
            expanded_jobs.append(job)

    if attempted_count > 0:
        logger.info(f"Expanded {expanded_count}/{attempted_count} job descriptions (out of {len(jobs)} total)")

    return expanded_jobs
