"""Hard exclusion filters for job listings."""

import logging
from datetime import datetime, timedelta
from typing import List
from .job_parser import ParsedJob


logger = logging.getLogger(__name__)


class HardExclusionFilters:
    """
    Apply hard exclusion filters before scoring.

    If a job fails ANY filter, it's immediately discarded.
    """

    def __init__(self, config):
        """Initialize filters with configuration."""
        self.config = config
        self.excluded_count = 0

    def apply_filters(self, jobs: List[ParsedJob]) -> List[ParsedJob]:
        """
        Apply all hard exclusion filters.

        Returns only jobs that pass ALL filters.
        """
        filtered_jobs = []
        self.excluded_count = 0

        for job in jobs:
            exclusion_reason = self._check_exclusion(job)

            if exclusion_reason:
                self.excluded_count += 1
                logger.debug(f"Excluded {job.title} at {job.company}: {exclusion_reason}")
            else:
                filtered_jobs.append(job)

        logger.info(f"Filtered to {len(filtered_jobs)} jobs from {len(jobs)} (excluded {self.excluded_count})")
        return filtered_jobs

    def _check_exclusion(self, job: ParsedJob) -> str:
        """
        Check if job should be excluded.

        Returns reason string if excluded, empty string if passes.
        """
        # Filter 1: Apply URL must be present
        if not job.apply_url or not job.apply_url.startswith(('http://', 'https://')):
            return "Missing or invalid apply URL"

        # Filter 2: Location preference
        location_reason = self._check_location_preference(job)
        if location_reason:
            return location_reason

        # Filter 3: Job age
        age_reason = self._check_job_age(job)
        if age_reason:
            return age_reason

        # Filter 4: Posting date freshness
        freshness_reason = self._check_posting_date_freshness(job)
        if freshness_reason:
            return freshness_reason

        # Filter 5: Content quality
        quality_reason = self._check_content_quality(job)
        if quality_reason:
            return quality_reason

        # Filter 6: Boolean search page validation
        validation_reason = self._check_boolean_search_validation(job)
        if validation_reason:
            return validation_reason

        # Passed all filters
        return ""

    def _check_location_preference(self, job: ParsedJob) -> str:
        """Check if job matches user's location preference."""
        pref = self.config.job_preferences.location_preference.lower()

        if pref == "any":
            return ""

        job_location = job.location.lower()

        # User wants remote only
        if pref == "remote":
            if "remote" not in job_location and "anywhere" not in job_location:
                return f"Location mismatch: user wants remote, job is {job.location}"

        # User wants hybrid
        elif pref == "hybrid":
            if "hybrid" not in job_location and "remote" not in job_location:
                return f"Location mismatch: user wants hybrid/remote, job is {job.location}"

        # User wants onsite
        elif pref == "onsite":
            if "remote" in job_location or "anywhere" in job_location:
                return f"Location mismatch: user wants onsite, job is {job.location}"

        return ""

    def _check_job_age(self, job: ParsedJob) -> str:
        """Check if job is too old."""
        if not job.posted_date:
            # If date is missing, we'll check freshness next
            return ""

        try:
            posted_date = datetime.fromisoformat(job.posted_date)
            max_age_days = self.config.job_preferences.max_job_age_days

            # Calculate age
            age = datetime.now(posted_date.tzinfo) - posted_date

            if age.days > max_age_days:
                return f"Job too old: {age.days} days (max {max_age_days})"

        except (ValueError, AttributeError) as e:
            logger.debug(f"Could not parse posted date for {job.title}: {e}")
            return ""

        return ""

    def _check_posting_date_freshness(self, job: ParsedJob) -> str:
        """
        Check posting date freshness.

        If date is missing AND source is not guaranteed fresh, exclude.
        """
        if job.posted_date:
            return ""

        # Sources that are guaranteed fresh (RSS feeds, APIs)
        guaranteed_fresh = ["RemoteOK", "We Work Remotely", "Remotive"]

        if job.source in guaranteed_fresh:
            return ""

        # If we don't know the date and source isn't guaranteed fresh, exclude
        return "Posting date missing and source not guaranteed fresh"

    def _check_content_quality(self, job: ParsedJob) -> str:
        """Check if job content is high-quality (not spammy or vague)."""
        description = job.description.lower()

        # Minimum description length
        if len(job.description) < 200:
            return "Description too short (likely low quality)"

        # Check for spammy indicators
        spam_keywords = [
            'urgent', 'immediate start', 'work from home', 'easy money',
            'no experience', 'entry level everyone', 'apply now',
            'click here', 'unlimited earning', 'be your own boss'
        ]

        spam_count = sum(1 for keyword in spam_keywords if keyword in description)
        if spam_count >= 3:
            return f"Too many spam indicators ({spam_count} keywords)"

        # Check for vague/low-signal content
        # If description has very few concrete technical terms
        tech_keywords = [
            'python', 'javascript', 'java', 'react', 'django', 'aws', 'docker',
            'sql', 'api', 'frontend', 'backend', 'database', 'cloud'
        ]

        tech_count = sum(1 for keyword in tech_keywords if keyword in description)
        if tech_count == 0:
            return "No technical keywords found (low signal)"

        return ""

    def _check_boolean_search_validation(self, job: ParsedJob) -> str:
        """
        Validate Boolean-sourced job pages.

        For MVP, this is a placeholder. In production, would:
        1. Fetch the page
        2. Verify it shows an active, legitimate job posting
        3. Check for "this job has been filled" messages
        """
        # Boolean search returns empty in MVP, so this won't trigger
        if job.source in ["Greenhouse", "Lever"]:
            # In production, would validate page
            pass

        return ""
