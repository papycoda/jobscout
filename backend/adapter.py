"""Adapter to wrap JobScout and capture JSON outputs."""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime

# Add parent directory to path to import jobscout
sys.path.insert(0, str(Path(__file__).parent.parent))

from jobscout.config import JobScoutConfig
from jobscout.resume_parser import ResumeParser
from jobscout.job_sources.base import JobListing
from jobscout.job_sources.rss_feeds import RemoteOKSource, WeWorkRemotelySource
from jobscout.job_sources.remotive_api import RemotiveSource
from jobscout.job_sources.boolean_search import BooleanSearchSource
from jobscout.job_parser import JobParser, ParsedJob
from jobscout.filters import HardExclusionFilters
from jobscout.scoring import JobScorer, ScoredJob
from jobscout.emailer import EmailDelivery


logger = logging.getLogger(__name__)


class JobScoutAdapter:
    """
    Adapter that wraps JobScout and captures JSON outputs.

    This extends the original JobScout to return structured data
    instead of just sending emails.
    """

    def __init__(self, config: JobScoutConfig):
        """Initialize adapter with configuration."""
        self.config = config

        # Initialize components
        self.resume_parser = ResumeParser()
        self.job_parser = JobParser()
        self.filters = HardExclusionFilters(config)
        self.emailer = EmailDelivery(config)

        # Parse resume
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
                "status": "success",
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
                "status": "success",
                "matching": 0
            }
        }

    def _fetch_jobs(self) -> List[JobListing]:
        """Fetch jobs from all sources."""
        all_jobs = []

        boards = self.config.job_preferences.job_boards
        if not boards:
            boards = ["remoteok", "weworkremotely", "remotive"]

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
        elif board_lower == "remotive":
            source = RemotiveSource("Remotive")
            return source.fetch_jobs(limit=50)
        else:
            logger.warning(f"Unknown job board: {board}")
            return []

    def _parse_jobs(self, jobs: List[JobListing]) -> List[ParsedJob]:
        """Parse job descriptions."""
        parsed = []

        for job in jobs:
            try:
                parsed_job = self.job_parser.parse(job)
                parsed.append(parsed_job)
            except Exception as e:
                logger.warning(f"Failed to parse job: {e}")
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

        # Location preference
        pref = self.config.job_preferences.location_preference.lower()
        if pref != "any":
            job_location = job.location.lower()

            if pref == "remote" and "remote" not in job_location and "anywhere" not in job_location:
                return f"Location mismatch: wants remote, job is {job.location}"
            elif pref == "hybrid" and "hybrid" not in job_location and "remote" not in job_location:
                return f"Location mismatch: wants hybrid/remote, job is {job.location}"
            elif pref == "onsite" and ("remote" in job_location or "anywhere" in job_location):
                return f"Location mismatch: wants onsite, job is {job.location}"

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

        # Content quality
        if len(job.description) < 200:
            return "Description too short"

        # Passed all filters
        return ""

    def _score_and_capture(self, jobs: List[ParsedJob]) -> Dict:
        """Score jobs and capture matching/filtered."""
        matching = []
        filtered = []

        for job in jobs:
            scored = self._score_job(job)

            if scored.is_apply_ready:
                matching.append(scored)
            else:
                # Build reasons
                reasons = []

                if scored.score < self.config.min_score_threshold:
                    reasons.append(f"Score below threshold ({scored.score:.0f}% < {self.config.min_score_threshold}%)")

                if scored.must_have_coverage < 0.6:
                    reasons.append(f"Must-have coverage too low ({scored.must_have_coverage:.0%} < 60%)")

                if not job.must_have_skills and scored.score < self.config.fallback_min_score:
                    reasons.append(f"No must-have skills and score below fallback ({scored.score:.0f}% < {self.config.fallback_min_score}%)")

                filtered.append({
                    "job": job,
                    "scored": scored,
                    "reasons": reasons
                })

        return {
            "matching": matching,
            "filtered": filtered
        }

    def _score_job(self, job: ParsedJob) -> ScoredJob:
        """Score a single job (simplified from JobScorer)."""
        # Calculate coverage
        coverage, missing, matching = self._calculate_coverage(job)

        # Calculate overlap
        overlap = self._calculate_overlap(job)

        # Calculate seniority
        seniority = self._calculate_seniority(job)

        # Weighted score
        score = (coverage * 60) + (overlap * 25) + (seniority * 15)

        # Empty must-have penalty
        if not job.must_have_skills:
            coverage = 0.5
            score -= 10

        score = max(0, min(100, score))

        # Check if apply-ready
        is_ready = self._is_apply_ready(score, coverage, job)

        return ScoredJob(
            job=job,
            score=score,
            is_apply_ready=is_ready,
            must_have_coverage=coverage,
            stack_overlap=overlap,
            seniority_alignment=seniority,
            missing_must_haves=missing,
            matching_skills=matching
        )

    def _calculate_coverage(self, job: ParsedJob):
        """Calculate must-have coverage."""
        must_haves = job.must_have_skills

        if not must_haves:
            return 0.0, set(), set()

        all_skills = self.resume.skills.copy()
        all_skills.update(self.config.job_preferences.preferred_tech_stack)

        matching = set()
        missing = set()

        for skill in must_haves:
            if skill in all_skills:
                matching.add(skill)
            else:
                missing.add(skill)

        coverage = len(matching) / len(must_haves)
        return coverage, missing, matching

    def _calculate_overlap(self, job: ParsedJob) -> float:
        """Calculate stack overlap."""
        job_skills = job.must_have_skills | job.nice_to_have_skills

        if not job_skills:
            return 0.5

        all_skills = self.resume.skills.copy()
        all_skills.update(self.config.job_preferences.preferred_tech_stack)

        overlap = job_skills & all_skills
        return len(overlap) / len(job_skills)

    def _calculate_seniority(self, job: ParsedJob) -> float:
        """Calculate seniority alignment."""
        if job.seniority_level == "unknown" or self.resume.seniority == "unknown":
            return 0.7

        if job.seniority_level == self.resume.seniority:
            return 1.0

        # Simple alignment logic
        if job.seniority_level == "senior" and self.resume.seniority in ["junior", "mid"]:
            if job.min_years_experience and job.min_years_experience >= 8:
                return 0.3 if self.resume.years_experience < 5 else 0.5
            return 0.6

        if job.seniority_level == "junior" and self.resume.seniority == "senior":
            return 0.6

        if job.seniority_level == "mid" and self.resume.seniority == "junior":
            return 0.5

        if self.resume.seniority == "senior":
            return 0.9

        return 0.7

    def _is_apply_ready(self, score: float, coverage: float, job: ParsedJob) -> bool:
        """Check if job is apply-ready."""
        if not job.must_have_skills:
            return score >= self.config.fallback_min_score

        return score >= self.config.min_score_threshold and coverage >= 0.6

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

        return {
            "id": job_id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "posted_at": job.posted_date,
            "apply_url": job.apply_url,
            "source": job.source,
            "snippet": job.description[:500] + "..." if len(job.description) > 500 else job.description,
            "score_total": round(scored.score, 1),
            "breakdown": {
                "must_have_coverage": round(scored.must_have_coverage, 2),
                "stack_overlap": round(scored.stack_overlap, 2),
                "seniority_alignment": round(scored.seniority_alignment, 2)
            },
            "must_have": {
                "matched": sorted(scored.matching_skills),
                "missing": sorted(scored.missing_must_haves)
            },
            "stack": {
                "matched": sorted(scored.matching_skills),
                "missing": sorted(job.must_have_skills | job.nice_to_have_skills - scored.matching_skills)
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

        return {
            "id": job_id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "source": job.source,
            "snippet": job.description[:300] + "..." if len(job.description) > 300 else job.description,
            "score_total": round(score, 1) if score is not None else None,
            "reasons": filtered_item["reasons"]
        }

    def send_email_digest(self, jobs_json: List[Dict], outbox_mode: bool = False) -> Dict:
        """
        Send email digest using existing EmailDelivery.

        Returns: (digest_id, mode)
        """
        if not jobs_json:
            raise ValueError("No jobs to send")

        # Reconstruct ScoredJob objects from JSON
        # This is a bit hacky but necessary to reuse EmailDelivery
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

        # Override outbox mode if requested
        if outbox_mode:
            original_smtp_config = (
                self.config.email.smtp_host,
                self.config.email.smtp_username,
                self.config.email.smtp_password
            )
            self.config.email.smtp_host = None
            self.config.email.smtp_username = None
            self.config.email.smtp_password = None

        # Send digest
        success = self.emailer.send_digest(scored_jobs)

        # Restore SMTP config if we overrode it
        if outbox_mode:
            self.config.email.smtp_host, self.config.email.smtp_username, self.config.email.smtp_password = original_smtp_config

        if not success:
            raise Exception("Failed to send email digest")

        # Generate digest ID
        digest_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        return {
            "digest_id": digest_id,
            "mode": "outbox" if outbox_mode else "smtp"
        }
