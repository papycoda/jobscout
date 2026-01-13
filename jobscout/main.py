"""JobScout main orchestration."""

import logging
from typing import List
from .config import JobScoutConfig
from .resume_parser import ResumeParser
from .job_sources.base import JobListing
from .job_sources.rss_feeds import RemoteOKSource, WeWorkRemotelySource
from .job_sources.remotive_api import RemotiveSource
from .job_sources.boolean_search import BooleanSearchSource
from .job_sources.greenhouse_api import GreenhouseSource
from .job_sources.lever_api import LeverSource
from .job_parser import JobParser
from .filters import HardExclusionFilters
from .scoring import JobScorer
from .emailer import EmailDelivery
from .role_recommender import RoleRecommender


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


logger = logging.getLogger(__name__)


class JobScout:
    """Main JobScout orchestration."""

    def __init__(self, config: JobScoutConfig):
        """Initialize JobScout with configuration."""
        self.config = config

        # Initialize components
        self.resume_parser = ResumeParser()
        self.job_parser = JobParser(config)  # Pass config for LLM support
        self.emailer = EmailDelivery(config)

        # Parse resume
        logger.info(f"Parsing resume from {config.resume_path}")
        self.resume = self.resume_parser.parse(config.resume_path)
        logger.info(f"Extracted {len(self.resume.skills)} skills from resume")
        logger.info(f"Skills: {', '.join(sorted(self.resume.skills))}")
        logger.info(f"Seniority: {self.resume.seniority}, Years: {self.resume.years_experience}")

        # Build preferred stack
        preferred_stack = set(config.job_preferences.preferred_tech_stack)
        logger.info(f"Preferred stack: {preferred_stack}")

        # Share user context with the job parser so LLM fallback has real signals
        self.job_parser._user_seniority = self.resume.seniority
        self.job_parser._user_years_experience = self.resume.years_experience

        # Recommend role keywords (fallback to resume-extracted keywords)
        self.role_keywords_for_search = list(self.resume.role_keywords) or ["software engineer"]

        if self.config.use_llm_parser and self.config.openai_api_key:
            try:
                advisor = RoleRecommender(
                    api_key=self.config.openai_api_key,
                    model=self.config.openai_model,
                )
                recommended_roles = advisor.recommend_roles(self.resume)
                if recommended_roles:
                    self.role_keywords_for_search = recommended_roles
                    logger.info(f"Using AI-recommended role keywords: {self.role_keywords_for_search}")
            except Exception as e:
                logger.warning(f"Role recommendation failed; using resume keywords: {e}")

        # Derive coarse role intent for filtering (backend/frontend/etc.)
        user_role_categories = self._infer_user_roles(self.role_keywords_for_search, self.resume.skills)

        # Initialize filters with derived user roles to avoid re-parsing the resume
        self.filters = HardExclusionFilters(config, user_roles=user_role_categories)

        # Keep around preferred stack for scoring
        self.preferred_stack = preferred_stack

        # Initialize scorer
        self.scorer = JobScorer(self.resume, config, preferred_stack)

    def run_search(self) -> bool:
        """
        Run complete job search pipeline.

        Returns True if successful, False otherwise.
        """
        try:
            # Step 1: Fetch jobs from all sources
            logger.info("Step 1: Fetching jobs from all sources...")
            jobs = self._fetch_jobs()
            logger.info(f"Fetched {len(jobs)} total jobs")

            # Step 2: Parse job descriptions
            logger.info("Step 2: Parsing job descriptions...")
            parsed_jobs = self._parse_jobs(jobs)
            logger.info(f"Parsed {len(parsed_jobs)} jobs")

            # Step 3: Apply hard exclusion filters
            logger.info("Step 3: Applying hard exclusion filters...")
            filtered_jobs = self.filters.apply_filters(parsed_jobs)

            if not filtered_jobs:
                logger.info("No jobs passed filters. Exiting.")
                return True

            # Step 4: Score jobs
            logger.info("Step 4: Scoring jobs...")
            scored_jobs = self.scorer.score_jobs(filtered_jobs)

            if not scored_jobs:
                logger.info("No jobs scored above threshold. Exiting.")
                return True

            # Step 5: Deduplicate within run
            logger.info("Step 5: Deduplicating jobs...")
            deduped_jobs = self._deduplicate_jobs(scored_jobs)
            logger.info(f"After deduplication: {len(deduped_jobs)} jobs")

            if not deduped_jobs:
                logger.info("No jobs after deduplication. Exiting.")
                return True

            # Step 6: Send email digest
            logger.info("Step 6: Sending email digest...")
            success = self.emailer.send_digest(deduped_jobs)

            if success:
                logger.info("JobScout run completed successfully")
            else:
                logger.error("Failed to send email digest")

            return success

        except Exception as e:
            logger.error(f"JobScout run failed: {e}", exc_info=True)
            return False

    def _fetch_jobs(self) -> List[JobListing]:
        """Fetch jobs from all enabled sources."""
        all_jobs = []

        # Default sources if none specified
        boards = self.config.job_preferences.job_boards
        if not boards:
            boards = ["remoteok", "weworkremotely", "remotive"]
            if self.config.job_preferences.greenhouse_boards:
                boards.append("greenhouse")
            if self.config.job_preferences.lever_companies:
                boards.append("lever")
            if self.config.serper_api_key:
                boards.append("boolean")

        # Fetch from each source
        for board in boards:
            try:
                jobs = self._fetch_from_source(board)
                all_jobs.extend(jobs)
                logger.info(f"Fetched {len(jobs)} jobs from {board}")
            except Exception as e:
                logger.error(f"Failed to fetch from {board}: {e}")
                continue

        return all_jobs

    def _fetch_from_source(self, board: str) -> List[JobListing]:
        """Fetch jobs from specific source."""
        board_lower = board.lower()

        if board_lower == "remoteok":
            source = RemoteOKSource("RemoteOK")
            return source.fetch_jobs(limit=50)

        elif board_lower == "weworkremotely":
            source = WeWorkRemotelySource("We Work Remotely")
            return source.fetch_jobs(limit=50)

        elif board_lower == "remotive":
            source = RemotiveSource("Remotive")
            return source.fetch_jobs(limit=50)
        elif board_lower == "greenhouse":
            source = GreenhouseSource(self.config.job_preferences.greenhouse_boards)
            return source.fetch_jobs(limit=50)
        elif board_lower == "lever":
            source = LeverSource(self.config.job_preferences.lever_companies)
            return source.fetch_jobs(limit=50)

        elif board_lower == "boolean":
            source = BooleanSearchSource(
                resume_skills=self.resume.skills,
                role_keywords=self.role_keywords_for_search,
                seniority=self.resume.seniority,
                location_preference=self.config.job_preferences.location_preference,
                max_job_age_days=self.config.job_preferences.max_job_age_days,
                serper_api_key=self.config.serper_api_key
            )
            return source.fetch_jobs(limit=30)

        else:
            logger.warning(f"Unknown job board: {board}")
            return []

    def _parse_jobs(self, jobs: List[JobListing]) -> List:
        """Parse job descriptions."""
        try:
            return self.job_parser.parse_batch(jobs, user_skills=self.resume.skills)
        except Exception as e:
            logger.warning(f"Batch parsing failed, falling back to single-job parsing: {e}")

        parsed_jobs = []
        for job in jobs:
            try:
                parsed = self.job_parser.parse(job, user_skills=self.resume.skills)
                parsed_jobs.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse job {job.title} at {job.company}: {e}")
                continue

        return parsed_jobs

    def _deduplicate_jobs(self, scored_jobs) -> List:
        """Deduplicate jobs within a single run."""
        seen_urls = set()
        unique_jobs = []

        for scored_job in scored_jobs:
            # Use apply_url as unique identifier instead of hashing the object
            job_url = scored_job.job.apply_url
            if job_url not in seen_urls:
                seen_urls.add(job_url)
                unique_jobs.append(scored_job)

        return unique_jobs

    @staticmethod
    def _infer_user_roles(role_keywords: List[str], skills: set) -> set:
        """Infer coarse role categories from role keywords and skills."""
        user_roles = set()

        for keyword in role_keywords or []:
            kw = keyword.lower()
            if 'backend' in kw:
                user_roles.add('backend')
            if 'frontend' in kw or 'front-end' in kw:
                user_roles.add('frontend')
            if 'fullstack' in kw or 'full-stack' in kw or 'full stack' in kw:
                user_roles.add('fullstack')
            if 'devops' in kw or 'sre' in kw or 'site reliability' in kw:
                user_roles.add('devops')
            if 'data' in kw or 'machine learning' in kw or 'ml ' in kw or kw.endswith(' ml'):
                user_roles.add('data')
            if 'mobile' in kw or 'ios' in kw or 'android' in kw:
                user_roles.add('mobile')

        if user_roles:
            return user_roles

        backend_indicators = {'python', 'django', 'fastapi', 'flask', 'java', 'go', 'ruby', 'php', 'rust'}
        frontend_indicators = {'react', 'vue', 'angular', 'javascript', 'typescript', 'css', 'html'}

        backend_count = sum(1 for skill in skills if skill in backend_indicators)
        frontend_count = sum(1 for skill in skills if skill in frontend_indicators)

        if backend_count >= 3 and backend_count > frontend_count * 3:
            user_roles.add('backend')
        elif frontend_count >= 3 and frontend_count > backend_count * 3:
            user_roles.add('frontend')
        elif backend_count >= 1 and frontend_count >= 1:
            user_roles.add('fullstack')

        return user_roles if user_roles else {'unknown'}


def run_jobscout(config_path: str = "config.yaml"):
    """Run JobScout with given config file."""
    # Load config
    config = JobScoutConfig.from_yaml(config_path)
    errors = config.validate()

    if errors:
        logger.error("Configuration errors:")
        for error in errors:
            logger.error(f"  - {error}")
        return False

    # Run JobScout
    jobscout = JobScout(config)
    return jobscout.run_search()


if __name__ == "__main__":
    import sys

    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_jobscout(config_file)
