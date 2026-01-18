"""Adapter to wrap JobScout and capture JSON outputs."""

import sys
import logging
import re
import html
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime

# Add parent directory to path to import jobscout
sys.path.insert(0, str(Path(__file__).parent.parent))

from jobscout.config import JobScoutConfig
from jobscout.resume_parser import ResumeParser, ParsedResume
from jobscout.job_sources.base import JobListing
from jobscout.job_sources.rss_feeds import RemoteOKSource, WeWorkRemotelySource, HimalayasSource, JavascriptJobsSource
from jobscout.job_sources.remotive_api import RemotiveSource
from jobscout.job_sources.boolean_search import BooleanSearchSource
from jobscout.job_sources.greenhouse_api import GreenhouseSource
from jobscout.job_sources.lever_api import LeverSource
from jobscout.job_parser import JobParser, ParsedJob
from jobscout.filters import HardExclusionFilters
from jobscout.scoring import JobScorer, ScoredJob
from jobscout.emailer import EmailDelivery


logger = logging.getLogger(__name__)


def _strip_html_tags(html_content: str) -> str:
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

    # Replace common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&#x27;', "'")
    text = text.replace('&#x2F;', '/')

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def _summarize_reasons(reasons: List[str]) -> tuple[str, str]:
    """Create a short summary and full detail string for filter reasons."""
    if not reasons:
        return "Filtered out", ""

    detail = "; ".join(reasons)

    summary_map = {
        "score below threshold": "Below score threshold",
        "must-have coverage too low": "Must-have coverage low",
        "no must-have skills and score below fallback": "Low score (no must-haves)",
        "location mismatch": "Location mismatch",
        "job too old": "Job too old",
        "description too short": "Description too short",
        "posting date missing": "Posting date missing",
        "missing or invalid apply url": "Missing apply URL",
        "role mismatch": "Role mismatch",
    }

    for reason in reasons:
        reason_lower = reason.lower()
        for prefix, summary in summary_map.items():
            if reason_lower.startswith(prefix):
                return summary, detail

    return reasons[0], detail


def _build_scored_jobs_from_json(jobs_json: List[Dict]) -> List[ScoredJob]:
    scored_jobs = []

    for job_dict in jobs_json:
        # Create minimal ParsedJob
        from jobscout.job_parser import ParsedJob
        from jobscout.scoring import ScoredJob

        parsed_job = ParsedJob(
            title=job_dict["title"],
            company=job_dict["company"],
            location=job_dict["location"],
            description=job_dict["snippet"],
            apply_url=job_dict["apply_url"],
            source=job_dict["source"],
            must_have_skills=set(job_dict["must_have"]["matched"] + job_dict["must_have"]["missing"]),
            nice_to_have_skills=set(),
            posted_date=job_dict.get("posted_at")
        )

        scored = ScoredJob(
            job=parsed_job,
            score=job_dict["score_total"],
            is_apply_ready=True,
            must_have_coverage=job_dict["breakdown"]["must_have_coverage"],
            stack_overlap=job_dict["breakdown"]["stack_overlap"],
            seniority_alignment=job_dict["breakdown"]["seniority_alignment"],
            missing_must_haves=set(job_dict["must_have"]["missing"]),
            matching_skills=set(job_dict["must_have"]["matched"])
        )

        scored_jobs.append(scored)

    return scored_jobs


def _resume_from_profile(profile: Dict) -> ParsedResume:
    years_value = profile.get("years_experience", 0.0) or 0.0
    return ParsedResume(
        raw_text=profile.get("raw_text", ""),
        skills=set(profile.get("skills", [])),
        tools=set(profile.get("tools", [])),
        seniority=profile.get("seniority", "unknown"),
        years_experience=float(years_value),
        role_keywords=profile.get("role_keywords", [])
    )


def send_email_digest_from_jobs(
    jobs_json: List[Dict],
    config: JobScoutConfig,
    outbox_mode: bool = False
) -> Dict:
    """
    Send email digest without parsing the resume.

    Returns: (digest_id, mode)
    """
    if not jobs_json:
        raise ValueError("No jobs to send")

    scored_jobs = _build_scored_jobs_from_json(jobs_json)
    emailer = EmailDelivery(config)

    if outbox_mode:
        original_smtp_config = (
            config.email.smtp_host,
            config.email.smtp_username,
            config.email.smtp_password
        )
        config.email.smtp_host = None
        config.email.smtp_username = None
        config.email.smtp_password = None

    success = emailer.send_digest(scored_jobs)

    if outbox_mode:
        config.email.smtp_host, config.email.smtp_username, config.email.smtp_password = original_smtp_config

    if not success:
        raise Exception("Failed to send email digest")

    digest_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    return {
        "digest_id": digest_id,
        "mode": "outbox" if outbox_mode else "smtp"
    }


class JobScoutAdapter:
    """
    Adapter that wraps JobScout and captures JSON outputs.

    This extends the original JobScout to return structured data
    instead of just sending emails.
    """

    def __init__(self, config: JobScoutConfig, resume_profile: Optional[Dict] = None):
        """Initialize adapter with configuration."""
        self.config = config

        # Initialize components
        self.resume_parser = ResumeParser()
        self.job_parser = JobParser(config)  # Pass config for LLM support
        self.filters = HardExclusionFilters(config)
        self.emailer = EmailDelivery(config)

        # Parse resume (or use stored profile)
        if resume_profile:
            logger.info("Using stored resume profile for job search")
            self.resume = _resume_from_profile(resume_profile)
        else:
            logger.info(f"Parsing resume from {config.resume_path}")
            self.resume = self.resume_parser.parse(config.resume_path)
        logger.info(f"Extracted {len(self.resume.skills)} skills from resume")

        # Build preferred stack
        preferred_stack = set(config.job_preferences.preferred_tech_stack)

        # Initialize scorer
        self.scorer = JobScorer(self.resume, config, preferred_stack)

    def run_and_capture(self) -> Dict:
        """
        Run JobScout and capture all outputs as JSON.

        Returns: Dict with jobs, filtered_jobs, and metadata
        """
        start_time = datetime.now()

        try:
            # Step 1: Fetch jobs
            logger.info("Fetching jobs from all sources...")
            jobs = self._fetch_jobs()
            logger.info(f"Fetched {len(jobs)} total jobs")

            # Step 2: Parse job descriptions
            logger.info("Parsing job descriptions...")
            parsed_jobs = self._parse_jobs(jobs)
            logger.info(f"Parsed {len(parsed_jobs)} jobs")

            # Step 3: Apply hard filters
            logger.info("Applying hard exclusion filters...")
            filtered = self._apply_filters_and_capture(parsed_jobs)
            passing_jobs = filtered["passing"]
            hard_filtered_jobs = filtered["filtered"]

            if not passing_jobs:
                logger.info("No jobs passed filters")
                return self._empty_result(hard_filtered_jobs, start_time)

            # Step 4: Score jobs
            logger.info("Scoring jobs...")
            scored = self._score_and_capture(passing_jobs)
            matching_jobs = scored["matching"]
            score_filtered_jobs = scored["filtered"]

            # Combine filtered jobs
            all_filtered = hard_filtered_jobs + score_filtered_jobs

            # Step 5: Deduplicate
            logger.info("Deduplicating jobs...")
            deduped_jobs = self._deduplicate_jobs(matching_jobs)
            logger.info(f"After deduplication: {len(deduped_jobs)} jobs")

            if not deduped_jobs:
                logger.info("No jobs after deduplication")
                return self._empty_result(all_filtered, start_time)

            # Serialize to JSON
            jobs_json = [self._scored_job_to_dict(j) for j in deduped_jobs]
            filtered_json = [self._filtered_job_to_dict(j) for j in all_filtered]

            end_time = datetime.now()

            # Metadata
            metadata = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "status": "completed",
                "total_fetched": len(jobs),
                "total_parsed": len(parsed_jobs),
                "hard_filtered": len(hard_filtered_jobs),
                "score_filtered": len(score_filtered_jobs),
                "matching": len(deduped_jobs),
                "resume_skills": list(self.resume.skills),
                "resume_seniority": self.resume.seniority
            }

            return {
                "jobs": jobs_json,
                "filtered_jobs": filtered_json,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"JobScout run failed: {e}", exc_info=True)
            end_time = datetime.now()

            return {
                "jobs": [],
                "filtered_jobs": [],
                "metadata": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": (end_time - start_time).total_seconds(),
                    "status": "error",
                    "error": str(e)
                }
            }

    def _empty_result(self, filtered_jobs: List, start_time: datetime) -> Dict:
        """Return empty result with metadata."""
        end_time = datetime.now()

        return {
            "jobs": [],
            "filtered_jobs": [self._filtered_job_to_dict(j) for j in filtered_jobs],
            "metadata": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "status": "completed",
                "matching": 0
            }
        }

    def _fetch_jobs(self) -> List[JobListing]:
        """Fetch jobs from all sources."""
        all_jobs = []

        boards = self.config.job_preferences.job_boards
        if not boards:
            boards = ["remoteok", "weworkremotely", "remotive", "himalayas", "jsjobs", "greenhouse", "lever"]
            if self.config.serper_api_key:
                boards.append("boolean")

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
        """Fetch from specific source."""
        board_lower = board.lower()

        if board_lower == "remoteok":
            source = RemoteOKSource("RemoteOK")
            return source.fetch_jobs(limit=50)
        elif board_lower == "weworkremotely":
            source = WeWorkRemotelySource("We Work Remotely")
            return source.fetch_jobs(limit=50)
        elif board_lower == "himalayas":
            source = HimalayasSource("Himalayas")
            return source.fetch_jobs(limit=20)
        elif board_lower == "jsjobs":
            source = JavascriptJobsSource("JavaScriptJobs")
            return source.fetch_jobs(limit=15)
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
                role_keywords=self.resume.role_keywords,
                seniority=self.resume.seniority,
                location_preference=self.config.job_preferences.location_preference,
                max_job_age_days=self.config.job_preferences.max_job_age_days,
                serper_api_key=self.config.serper_api_key
            )
            return source.fetch_jobs(limit=30)
        else:
            logger.warning(f"Unknown job board: {board}")
            return []

    def _parse_jobs(self, jobs: List[JobListing]) -> List[ParsedJob]:
        """
        Parse job descriptions using smart hybrid approach.

        Uses fast regex for most jobs, with selective LLM enhancement for promising jobs
        that have poor regex extraction.
        """
        try:
            # Use batch parsing for smart LLM fallback with user skills
            parsed = self.job_parser.parse_batch(
                jobs,
                user_skills=self.resume.skills
            )
            logger.info(f"Parsed {len(parsed)} jobs using smart hybrid approach")
            return parsed
        except Exception as e:
            logger.warning(f"Batch parsing failed, falling back to single-job parsing: {e}")
            # Fallback to single-job parsing
            parsed = []
            for job in jobs:
                try:
                    parsed_job = self.job_parser.parse(job, user_skills=self.resume.skills)
                    parsed.append(parsed_job)
                except Exception as e2:
                    logger.warning(f"Failed to parse job: {e2}")
                    continue
            return parsed

    def _apply_filters_and_capture(self, jobs: List[ParsedJob]) -> Dict:
        """Apply hard filters and capture passing/filtered jobs."""
        passing = []
        filtered = []

        for job in jobs:
            exclusion_reason = self._check_exclusion(job)

            if exclusion_reason:
                filtered.append({
                    "job": job,
                    "reasons": [exclusion_reason],
                    "score": None
                })
            else:
                passing.append(job)

        return {
            "passing": passing,
            "filtered": filtered
        }

    def _check_exclusion(self, job: ParsedJob) -> str:
        """Check if job should be hard-filtered."""
        # Missing apply URL
        if not job.apply_url or not job.apply_url.startswith(('http://', 'https://')):
            return "Missing or invalid apply URL"

        # Location preference - smarter matching
        pref = self.config.job_preferences.location_preference.lower()
        if pref != "any":
            job_location = job.location.lower()

            # Check if the location field OR description contains remote indicators
            combined_text = f"{job.location} {job.description}".lower()
            has_remote = any(indicator in combined_text for indicator in
                          ['remote', 'anywhere', 'global', 'distributed', 'home-based'])

            if pref == "remote":
                # Accept if explicitly says remote/anywhere OR if location is vague/empty
                if not has_remote and job_location and not any(
                    vague in job_location for vague in ['remote', 'anywhere', 'global', 'n/a', '-']
                ):
                    return f"Location mismatch: wants remote, job is {job.location}"
            elif pref == "hybrid":
                if not (has_remote or 'hybrid' in combined_text):
                    return f"Location mismatch: wants hybrid/remote, job is {job.location}"
            elif pref == "onsite":
                if has_remote:
                    return f"Location mismatch: wants onsite, job is remote"

        # Job age
        if job.posted_date:
            try:
                from datetime import datetime
                posted = datetime.fromisoformat(job.posted_date)
                max_age = self.config.job_preferences.max_job_age_days
                age_days = (datetime.now(posted.tzinfo) - posted).days

                if age_days > max_age:
                    return f"Job too old: {age_days} days (max {max_age})"
            except Exception:
                pass

        # Content quality - more lenient
        # Allow shorter descriptions if job has meaningful content
        if len(job.description) < 50:
            return "Description too short"

        # Passed all filters
        return ""

    def _score_and_capture(self, jobs: List[ParsedJob]) -> Dict:
        """Score jobs and capture matching/filtered."""
        matching = []
        filtered = []

        for job in jobs:
            # Use the shared JobScorer logic (languages/frameworks + role-aware)
            scored = self.scorer._score_job(job)

            if scored.is_apply_ready:
                matching.append(scored)
            else:
                # Build reasons
                reasons = []

                if scored.score < self.config.min_score_threshold:
                    reasons.append(f"Score below threshold ({scored.score:.0f}% < {self.config.min_score_threshold}%)")

                # Only check coverage if must-have skills exist
                if job.must_have_skills and scored.must_have_coverage < 0.6:
                    reasons.append(f"Must-have coverage too low ({scored.must_have_coverage:.0%} < 60%)")

                # Flag explicit role mismatches when both sides are known
                job_roles = self.scorer._extract_roles(
                    job.title + " " + job.description,
                    job.must_have_skills | job.nice_to_have_skills
                )
                if job_roles and self.scorer.candidate_roles and not (job_roles & self.scorer.candidate_roles):
                    reasons.append("Role mismatch (job role vs resume role)")

                # Note: we no longer use fallback_min_score for filtering
                # Jobs without must-haves are evaluated using the same threshold as normal jobs

                filtered.append({
                    "job": job,
                    "scored": scored,
                    "reasons": reasons
                })

        return {
            "matching": matching,
            "filtered": filtered
        }

    def _deduplicate_jobs(self, scored_jobs: List[ScoredJob]) -> List[ScoredJob]:
        """Deduplicate jobs."""
        seen = set()
        unique = []

        for scored in scored_jobs:
            # Use apply_url as hashable identifier instead of hashing the object
            job_hash = hash(scored.job.apply_url)
            if job_hash not in seen:
                seen.add(job_hash)
                unique.append(scored)

        return unique

    def _scored_job_to_dict(self, scored: ScoredJob) -> Dict:
        """Convert ScoredJob to dict."""
        job = scored.job

        # Generate stable ID from apply_url
        import hashlib
        job_id = hashlib.sha256(job.apply_url.encode()).hexdigest()[:16]

        # Seniority explanation
        if scored.seniority_alignment >= 0.9:
            sen_expl = "Strong match"
        elif scored.seniority_alignment >= 0.7:
            sen_expl = "Good match"
        elif scored.seniority_alignment >= 0.5:
            sen_expl = "Possible mismatch"
        else:
            sen_expl = "Poor match"

        # Clean HTML from description for snippet
        clean_description = _strip_html_tags(job.description)
        snippet = clean_description[:500] + "..." if len(clean_description) > 500 else clean_description

        job_skills = job.must_have_skills | job.nice_to_have_skills
        candidate_skills = self.scorer.all_candidate_skills
        stack_matched = sorted(job_skills & candidate_skills)
        stack_missing = sorted(job_skills - candidate_skills)
        must_have_matched = sorted(scored.matching_skills)
        must_have_missing = sorted(scored.missing_must_haves)
        if not job.must_have_skills:
            must_have_matched = stack_matched
            must_have_missing = stack_missing

        return {
            "id": job_id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "posted_at": job.posted_date,
            "apply_url": job.apply_url,
            "source": job.source,
            "snippet": snippet,
            "score_total": round(scored.score, 1),
            "breakdown": {
                "must_have_coverage": round(scored.must_have_coverage, 2),
                "stack_overlap": round(scored.stack_overlap, 2),
                "seniority_alignment": round(scored.seniority_alignment, 2)
            },
            "must_have": {
                "matched": must_have_matched,
                "missing": must_have_missing
            },
            "stack": {
                "matched": stack_matched,
                "missing": stack_missing
            },
            "seniority": {
                "expected": job.seniority_level,
                "found": self.resume.seniority,
                "explanation": sen_expl
            }
        }

    def _filtered_job_to_dict(self, filtered_item: Dict) -> Dict:
        """Convert filtered job to dict."""
        if "scored" in filtered_item:
            # Scored and filtered
            scored = filtered_item["scored"]
            job = scored.job
            score = scored.score
        else:
            # Hard filtered
            job = filtered_item["job"]
            score = None

        import hashlib
        job_id = hashlib.sha256(job.apply_url.encode()).hexdigest()[:16]

        # Clean HTML from description for snippet
        clean_description = _strip_html_tags(job.description)
        snippet = clean_description[:300] + "..." if len(clean_description) > 300 else clean_description
        summary, detail = _summarize_reasons(filtered_item.get("reasons", []))

        return {
            "id": job_id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "apply_url": job.apply_url,
            "source": job.source,
            "snippet": snippet,
            "score_total": round(score, 1) if score is not None else None,
            "reasons": filtered_item["reasons"],
            "reason_summary": summary,
            "reason_detail": detail
        }

    def send_email_digest(self, jobs_json: List[Dict], outbox_mode: bool = False) -> Dict:
        """
        Send email digest using existing EmailDelivery.

        Returns: (digest_id, mode)
        """
        return send_email_digest_from_jobs(jobs_json, self.config, outbox_mode=outbox_mode)
