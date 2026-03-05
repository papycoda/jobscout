"""Job sources for agent-based JobScout."""

from .base import JobListing, JobSource, strip_html_tags
from .fetcher import JobFetcher, fetch_jobs_for_profile
from .rss_feeds import RemoteOKSource, WeWorkRemotelySource, HimalayasSource
from .remotive_api import RemotiveSource
from .company_boards import CompanyBoardsSource

# Default sources constant
DEFAULT_SOURCES = ["company_boards", "remoteok", "weworkremotely", "remotive"]

__all__ = [
    "JobListing",
    "JobSource",
    "strip_html_tags",
    "JobFetcher",
    "fetch_jobs_for_profile",
    "DEFAULT_SOURCES",
    "RemoteOKSource",
    "WeWorkRemotelySource",
    "HimalayasSource",
    "RemotiveSource",
    "CompanyBoardsSource",
]
