"""Hard exclusion filters for job listings."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Set
from .job_parser import ParsedJob


logger = logging.getLogger(__name__)


class HardExclusionFilters:
    """
    Apply hard exclusion filters before scoring.

    If a job fails ANY filter, it's immediately discarded.
    """

    def __init__(self, config, user_roles: Optional[Set[str]] = None):
        """Initialize filters with configuration."""
        self.config = config
        self.excluded_count = 0
        # Optional override so stateless flows don't re-parse the resume for every job.
        self.user_roles_override = user_roles
        self._cached_user_roles: Optional[Set[str]] = None

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

        # Filter 6: Role mismatch (NEW!)
        role_mismatch_reason = self._check_role_mismatch(job)
        if role_mismatch_reason:
            return role_mismatch_reason

        # Filter 7: Boolean search page validation
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

        If date is missing AND source is not guaranteed fresh, apply small penalty
        instead of excluding. Jobs with unknown dates will be penalized during scoring
        rather than filtered out entirely.
        """
        if job.posted_date:
            return ""

        # Sources that are guaranteed fresh (RSS feeds, APIs)
        guaranteed_fresh = ["RemoteOK", "We Work Remotely", "Remotive"]

        if job.source in guaranteed_fresh:
            return ""

        # Don't exclude jobs with missing dates - let them through with a note
        # The scoring system will apply a small penalty for unknown dates
        logger.debug(f"Job {job.title} at {job.company} has unknown posting date from {job.source} - allowing with potential penalty")
        return ""

    def _check_content_quality(self, job: ParsedJob) -> str:
        """Check if job content is high-quality (not spammy or vague)."""
        description = job.description.lower()

        # Check for severely truncated descriptions
        truncation_reason = self._check_truncation(job)
        if truncation_reason:
            return truncation_reason

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

    def _check_truncation(self, job: ParsedJob) -> str:
        """
        Check if job description is severely truncated or incomplete.

        Filters out jobs where the description is cut off mid-section,
        which indicates missing critical requirements information.
        """
        import re

        description = job.description

        # Check for severely truncated descriptions (too short to be useful)
        if len(description) < 300:
            return f"Description too short and incomplete ({len(description)} chars)"

        # Check for incomplete section headers - this is the strongest signal
        # Only flag if section header is near the END (within ~150 chars) meaning no content follows
        incomplete_patterns = [
            r'what you\'?ll bring\s*:\s*$',
            r'what you bring\s*:\s*$',
            r'you will bring\s*:\s*$',
            r'you bring\s*:\s*$',
            r'requirements\s*:\s*$',
            r'qualifications\s*:\s*$',
            r'you have\s*:\s*$',
            r'you should have\s*:\s*$',
            r'benefits\s*:\s*$',
            r'about the role\s*:\s*$',
        ]

        description_lower = description.lower()
        for pattern in incomplete_patterns:
            # Find all matches of this pattern
            for match in re.finditer(pattern, description_lower, re.MULTILINE):
                # Only flag if this header is in the last ~150 chars of the description
                # (meaning no meaningful content follows)
                if match.start() > len(description_lower) - 150:
                    return "Description truncated (section header at end without content)"

        # Check if description ends mid-sentence or mid-word (strong truncation signal)
        stripped = description.strip()
        if len(stripped) > 200:  # Only check for longer descriptions
            last_char = stripped[-1] if stripped else ''
            # If it doesn't end with proper sentence terminator
            if last_char not in '.!?\'")])':
                # Check if it's not a reasonable endpoint like a digit or parenthesis
                if not last_char.isdigit() and last_char not in ')]}%':
                    # Additional check: see if there are incomplete patterns
                    # Look for section headers without content
                    lines = stripped.split('\n')
                    for line in lines[-5:]:  # Check last 5 lines
                        line_lower = line.strip().lower()
                        if any(header in line_lower for header in [
                            'what you\'ll bring', 'what you bring', 'requirements',
                            'qualifications', 'you will bring', 'you bring'
                        ]):
                            # If this line exists but description ends shortly after, it's truncated
                            return "Description truncated (section header without content)"

        # Check for obvious truncation markers at end
        truncation_markers = ['...', '&nbsp;', '•', '◆', '▪', '→']
        if any(stripped.endswith(marker) for marker in truncation_markers):
            return "Description ends with truncation marker"

        return ""

    def _check_role_mismatch(self, job: ParsedJob) -> str:
        """
        Check for role mismatches between user's resume and job requirements.

        Blocks hard mismatches:
        - Backend profile → NO frontend-only roles
        - Frontend profile → NO backend-only roles
        - Allows fullstack for both
        """
        if not job.job_roles or not job.job_roles:
            return ""

        # Import here to avoid circular dependency
        from .resume_parser import ResumeParser

        # Get user's role intent from their resume role keywords
        # We need to parse the resume to get role_keywords unless supplied directly
        if self.user_roles_override is not None:
            user_roles = self.user_roles_override
        else:
            if self._cached_user_roles is not None:
                user_roles = self._cached_user_roles
            else:
                resume_parser = ResumeParser()
                try:
                    user_resume = resume_parser.parse(self.config.resume_path)
                    user_roles = self._infer_user_roles(user_resume.role_keywords, user_resume.skills)
                    self._cached_user_roles = user_roles
                except Exception as e:
                    logger.debug(f"Could not extract user role intent: {e}")
                    return ""

        # If user role intent is unknown, allow all jobs
        if 'unknown' in user_roles or not user_roles:
            return ""

        # Define role compatibility rules
        # Backend roles: backend, fullstack, devops, data
        # Frontend roles: frontend, fullstack, mobile
        # Fullstack roles: all roles

        job_role_set = job.job_roles

        # Backend profile blocking rules
        if 'backend' in user_roles and 'frontend' not in user_roles:
            # Backend-only profile: block frontend-only jobs
            if job_role_set == {'frontend'}:
                return f"Role mismatch: backend profile, frontend-only job ({job.title})"

        # Frontend profile blocking rules
        if 'frontend' in user_roles and 'backend' not in user_roles:
            # Frontend-only profile: block backend-only jobs
            if job_role_set == {'backend'}:
                return f"Role mismatch: frontend profile, backend-only job ({job.title})"

        # Fullstack is always allowed for any profile
        # DevOps, Data, Mobile are allowed for both backend and frontend profiles
        return ""

    def _infer_user_roles(self, role_keywords: List[str], skills: set) -> set:
        """
        Infer user's role intent from resume role keywords and skills.

        Returns set of roles: 'backend', 'frontend', 'fullstack', 'devops', etc.
        """
        user_roles = set()

        # Check explicit role keywords
        for keyword in role_keywords:
            keyword_lower = keyword.lower()
            if 'backend' in keyword_lower:
                user_roles.add('backend')
            if 'frontend' in keyword_lower:
                user_roles.add('frontend')
            if 'fullstack' in keyword_lower or 'full-stack' in keyword_lower or 'full stack' in keyword_lower:
                user_roles.add('fullstack')
            if 'devops' in keyword_lower or 'sre' in keyword_lower:
                user_roles.add('devops')
            if 'data' in keyword_lower or 'machine learning' in keyword_lower or 'ml' in keyword_lower:
                user_roles.add('data')

        # If no explicit role keywords, infer from skills
        if not user_roles:
            backend_indicators = {'python', 'django', 'fastapi', 'flask', 'java', 'go', 'ruby', 'php', 'rust'}
            frontend_indicators = {'react', 'vue', 'angular', 'javascript', 'typescript', 'css', 'html'}

            backend_count = sum(1 for skill in skills if skill in backend_indicators)
            frontend_count = sum(1 for skill in skills if skill in frontend_indicators)

            # Use 3:1 ratio for clear backend/frontend bias
            if backend_count >= 3 and backend_count > frontend_count * 3:
                user_roles.add('backend')
            elif frontend_count >= 3 and frontend_count > backend_count * 3:
                user_roles.add('frontend')
            elif backend_count >= 1 and frontend_count >= 1:
                user_roles.add('fullstack')

        return user_roles if user_roles else {'unknown'}

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
