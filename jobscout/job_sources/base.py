"""Base class for job sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


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
