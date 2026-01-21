"""Base class for job sources."""

import re
import html
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


def strip_html_tags(html_content: str) -> str:
    """
    Strip HTML tags from a string and return clean plain text.

    Args:
        html_content: String containing HTML markup

    Returns:
        Clean plain text with HTML tags removed and entities decoded
    """
    if not html_content:
        return ""

    # Remove script and style tags with their content
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.IGNORECASE | re.DOTALL)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', html_content)

    # Decode HTML entities
    text = html.unescape(text)

    # Replace common HTML entities that unescape might miss
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&#x27;', "'")

    # Clean up whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)


@dataclass
class JobListing:
    """Unified job listing representation."""
    title: str
    company: str
    location: str  # e.g., "Remote", "San Francisco, CA", "Hybrid"
    description: str
    apply_url: str
    source: str  # e.g., "RemoteOK", "We Work Remotely", "Greenhouse"
    posted_date: Optional[datetime] = None
    salary: Optional[str] = None

    def __hash__(self):
        """Hash for deduplication."""
        import hashlib
        content = f"{self.title}{self.company}{self.location}{self.apply_url}"
        return int(hashlib.sha256(content.encode()).hexdigest(), 16)

    def __eq__(self, other):
        """Equality check for deduplication."""
        if not isinstance(other, JobListing):
            return False
        return self.__hash__() == other.__hash__()


class JobSource(ABC):
    """Abstract base class for job sources."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fetch_jobs(self, limit: int = 50) -> List[JobListing]:
        """Fetch jobs from this source."""
        pass
