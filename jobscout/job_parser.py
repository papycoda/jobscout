"""Parse job descriptions to extract requirements."""

import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Set, List, Optional, TYPE_CHECKING
from .resume_parser import SKILL_DICT

if TYPE_CHECKING:
    from .config import JobScoutConfig


logger = logging.getLogger(__name__)


@dataclass
class ParsedJob:
    """Parsed job requirements."""
    title: str
    company: str
    location: str
    description: str
    apply_url: str
    source: str

    # Extracted requirements
    must_have_skills: Set[str]
    nice_to_have_skills: Set[str]

    # Seniority requirements
    min_years_experience: Optional[float] = None
    seniority_level: str = "unknown"  # junior, mid, senior, unknown

    # Role categories (for filtering)
    job_roles: Set[str] = field(default_factory=set)
    role_keywords: Set[str] = field(default_factory=set)

    # Metadata
    posted_date: Optional[str] = None


class JobParser:
    """Parse job descriptions to extract structured requirements."""

    def __init__(self, config: Optional["JobScoutConfig"] = None):
        """
        Initialize parser with skill patterns and optional LLM support.

        Args:
            config: Optional config for LLM parser settings
        """
        self._build_patterns()
        self._build_section_patterns()
        self.llm_parser = None
        self.use_llm = False

        if config and config.use_llm_parser and config.openai_api_key:
            try:
                from .llm_parser import LLMJobParser
                self.llm_parser = LLMJobParser(
                    api_key=config.openai_api_key,
                    model=config.openai_model,
                    fallback_parser=self
                )
                self.use_llm = True
                logger.info(f"LLM parser enabled with model: {config.openai_model}")
            except ImportError as e:
                logger.warning(f"Failed to initialize LLM parser: {e}. Using regex parser.")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM parser: {e}. Using regex parser.")

    def _build_patterns(self):
        """Build regex patterns for skill matching."""
        self.patterns = {}
        for canonical, variants in SKILL_DICT.items():
            pattern = r'\b(' + '|'.join(re.escape(v) for v in variants) + r')\b'
            self.patterns[canonical] = re.compile(pattern, re.IGNORECASE)

    def _build_section_patterns(self):
        """Build patterns to identify requirement sections."""
        # Patterns that indicate must-have requirements
        self.must_have_patterns = [
            re.compile(r'(requirements|qualifications|what you\'ll bring|you have)', re.IGNORECASE),
            re.compile(r'(must have|required|essential)', re.IGNORECASE),
        ]

        # Patterns that indicate nice-to-have requirements
        self.nice_to_have_patterns = [
            re.compile(r'(nice to have|bonus|preferred|plus|desirable)', re.IGNORECASE),
        ]

        # Patterns that indicate experience requirements
        self.experience_patterns = [
            re.compile(r'(\d+)\+?\s*years?\s*(of)?\s*experience', re.IGNORECASE),
            re.compile(r'(\d+)\+?\s*years?\s*(of)?\s*(professional|work)', re.IGNORECASE),
        ]

        self.primary_requirement_headers = [
            "requirements",
            "qualifications",
            "what you'll bring",
            "you have",
            "must have",
            "required",
            "essential",
        ]
        self.secondary_requirement_headers = [
            "responsibilities",
            "about the role",
        ]
        self.nice_to_have_headers = [
            "nice to have",
            "bonus",
            "preferred",
            "plus",
            "desirable",
        ]

        stop_headers = (
            self.primary_requirement_headers
            + self.secondary_requirement_headers
            + self.nice_to_have_headers
        )
        stop_pattern = r'\n\s*(' + '|'.join(re.escape(h) for h in stop_headers) + r')\s*\n'
        self.section_stop_pattern = re.compile(stop_pattern, re.IGNORECASE)

    def parse(self, job, user_skills: Optional[Set[str]] = None) -> ParsedJob:
        """
        Parse job listing to extract requirements.

        Uses smart hybrid approach:
        1. Always try regex first (fast)
        2. If LLM enabled AND regex finds < 2 must-haves AND job looks promising, use LLM
        3. Otherwise use regex results

        Args:
            job: JobListing to parse
            user_skills: Optional set of user's skills to determine if job is promising
        """
        # Always try regex first (fast, <1ms)
        regex_result = self._parse_with_regex(job)

        # Smart LLM fallback: only if LLM enabled AND regex found little AND job looks promising
        if self.use_llm and self.llm_parser and user_skills is not None:
            # Check if regex found enough must-haves
            if len(regex_result.must_have_skills) < 2:
                # Check if job looks promising (has some matching skills)
                all_job_skills = regex_result.must_have_skills | regex_result.nice_to_have_skills
                matching_skills = all_job_skills & user_skills

                # If job has at least 2 matching skills, it's worth LLM enhancement
                if len(matching_skills) >= 2:
                    try:
                        logger.debug(f"Job '{job.title}' at {job.company} has poor regex extraction ({len(regex_result.must_have_skills)} must-haves) but {len(matching_skills)} matching skills - trying LLM")
                        job_metadata = {
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "apply_url": job.apply_url,
                            "source": job.source,
                        }
                        # Pass user context to LLM parser
                        llm_result = self.llm_parser.parse(
                            job.description,
                            job_metadata,
                            user_skills=user_skills,
                            user_seniority=getattr(self, '_user_seniority', 'unknown'),
                            user_years_experience=getattr(self, '_user_years_experience', 0.0)
                        )
                        # Mark as LLM-enhanced for debugging
                        llm_result._parsing_method = "llm_fallback"
                        return llm_result
                    except Exception as e:
                        logger.warning(f"LLM parsing failed for {job.title} at {job.company}: {e}. Using regex results.")

        # Mark as regex-parsed for debugging
        regex_result._parsing_method = "regex"
        return regex_result

    def parse_batch(self, jobs: List, user_skills: Optional[Set[str]] = None) -> List[ParsedJob]:
        """
        Parse multiple jobs with smart LLM fallback in parallel.

        This is faster than calling parse() multiple times because LLM calls are parallelized.

        Args:
            jobs: List of JobListing objects to parse
            user_skills: Optional set of user's skills

        Returns:
            List of ParsedJob objects
        """
        if not self.use_llm or not self.llm_parser or user_skills is None:
            # No LLM or no user skills - parse all with regex
            return [self.parse(job, user_skills) for job in jobs]

        # First pass: regex parsing for all jobs (fast)
        regex_results = []
        jobs_needing_llm = []

        for job in jobs:
            regex_result = self._parse_with_regex(job)
            regex_results.append((job, regex_result))

            # Check if this job needs LLM enhancement
            if len(regex_result.must_have_skills) < 2:
                all_job_skills = regex_result.must_have_skills | regex_result.nice_to_have_skills
                matching_skills = all_job_skills & user_skills

                if len(matching_skills) >= 2:
                    jobs_needing_llm.append((job, regex_result))

        # If no jobs need LLM, return regex results
        if not jobs_needing_llm:
            results = [r for _, r in regex_results]
            for r in results:
                r._parsing_method = "regex"
            return results

        # Second pass: LLM parsing for promising jobs in parallel
        logger.info(f"Smart hybrid: {len(jobs)} total jobs, {len(jobs_needing_llm)} need LLM enhancement ({len(jobs_needing_llm)/len(jobs)*100:.1f}%)")

        def parse_job_llm(job_tuple):
            job, regex_result = job_tuple
            try:
                job_metadata = {
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "apply_url": job.apply_url,
                    "source": job.source,
                }
                # Pass user context to LLM parser
                llm_result = self.llm_parser.parse(
                    job.description,
                    job_metadata,
                    user_skills=user_skills,
                    user_seniority=getattr(self, '_user_seniority', 'unknown'),
                    user_years_experience=getattr(self, '_user_years_experience', 0.0)
                )
                llm_result._parsing_method = "llm_fallback"
                return job, llm_result
            except Exception as e:
                logger.warning(f"LLM parsing failed for {job.title} at {job.company}: {e}. Using regex results.")
                regex_result._parsing_method = "regex_fallback"
                return job, regex_result

        # Run LLM parsing in parallel using threads
        llm_results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all LLM parsing jobs
            future_to_job = {executor.submit(parse_job_llm, jt): jt for jt in jobs_needing_llm}

            # Collect results as they complete
            for future in as_completed(future_to_job):
                try:
                    result = future.result()
                    llm_results.append(result)
                except Exception as e:
                    logger.warning(f"LLM parsing failed: {e}")

        # Merge results
        llm_lookup = {job: result for job, result in llm_results}
        final_results = []

        for job, regex_result in regex_results:
            if job in llm_lookup:
                final_results.append(llm_lookup[job])
            else:
                regex_result._parsing_method = "regex"
                final_results.append(regex_result)

        return final_results

    def _is_description_truncated(self, description: str) -> bool:
        """
        Detect if job description is truncated or incomplete.

        Returns True if description shows signs of being cut off.
        """
        if not description or len(description) < 500:
            return True

        description_lower = description.lower()

        # Check for incomplete section headers
        incomplete_patterns = [
            r'(what you\'ll bring|what you bring|you will bring|you bring)\s*:\s*$',
            r'(requirements|qualifications)\s*:\s*$',
            r'(you have|you should have)\s*:\s*$',
            r'benefits\s*:\s*$',
            r'about\s+the\s+role\s*:\s*$',
        ]

        for pattern in incomplete_patterns:
            if re.search(pattern, description_lower, re.MULTILINE | re.IGNORECASE):
                return True

        # Check if description ends mid-sentence or mid-word
        stripped = description.strip()
        if stripped and stripped[-1] not in ".!?'\"]":
            # Check if it's not a reasonable endpoint
            if len(stripped) > 100 and not stripped[-1].isdigit():
                return True

        # Check for common truncation markers
        truncation_markers = [
            '...',
            '&nbsp;',
            '•',
            '◆',
            '▪'
        ]
        if any(stripped.endswith(marker) for marker in truncation_markers):
            return True

        return False

    def _extract_skills_from_title(self, title: str) -> Set[str]:
        """
        Extract skills from job title as fallback for truncated descriptions.

        This catches cases where the title contains key tech requirements
        but the description is cut off before listing them.
        """
        found_skills = set()
        title_lower = title.lower()

        # Check all skills in our dictionary against the title
        for canonical, variants in SKILL_DICT.items():
            # For title matching, we need to be more strict
            # Use word boundaries to avoid partial matches
            for variant in variants:
                # Create a pattern that matches the skill as a whole word
                pattern = r'\b' + re.escape(variant) + r'\b'
                if re.search(pattern, title_lower, re.IGNORECASE):
                    found_skills.add(canonical)
                    break

        return found_skills

    def _extract_job_roles(self, job) -> Set[str]:
        """
        Extract role categories from job title and description.

        Returns set of role categories: 'backend', 'frontend', 'fullstack',
        'devops', 'data', 'mobile', 'unknown'
        """
        title = job.title.lower()
        description = job.description.lower()
        combined = title + " " + description

        roles = set()

        # Backend indicators
        backend_keywords = ['backend', 'back-end', 'server', 'api', 'infrastructure']
        backend_skills = {'python', 'java', 'go', 'ruby', 'php', 'rust', 'scala', 'kubernetes', 'docker', 'aws', 'terraform'}

        # Frontend indicators
        frontend_keywords = ['frontend', 'front-end', 'ui', 'ux', 'client-side']
        frontend_skills = {'react', 'vue', 'angular', 'svelte', 'javascript', 'typescript', 'css', 'html'}

        # Fullstack indicators
        fullstack_keywords = ['fullstack', 'full-stack', 'full stack', 'full stack engineer', 'fullstack developer']

        # DevOps indicators
        devops_keywords = ['devops', 'sre', 'site reliability', 'platform', 'infrastructure']

        # Data indicators
        data_keywords = ['data scientist', 'data engineer', 'machine learning', 'ml engineer', 'analytics']

        # Mobile indicators
        mobile_keywords = ['mobile', 'ios', 'android', 'native']

        # Check for explicit role keywords in title
        if any(kw in title for kw in backend_keywords):
            roles.add('backend')
        if any(kw in title for kw in frontend_keywords):
            roles.add('frontend')
        if any(kw in title for kw in fullstack_keywords):
            roles.add('fullstack')
        if any(kw in title for kw in devops_keywords):
            roles.add('devops')
        if any(kw in title for kw in data_keywords):
            roles.add('data')
        if any(kw in title for kw in mobile_keywords):
            roles.add('mobile')

        # Check skill patterns for role inference
        # Count backend vs frontend skills mentioned
        backend_mentioned = sum(1 for skill in backend_skills if skill in combined)
        frontend_mentioned = sum(1 for skill in frontend_skills if skill in combined)

        # If no explicit role keywords but clear skill bias, infer role
        if not roles and backend_mentioned >= 2 and frontend_mentioned == 0:
            roles.add('backend')
        if not roles and frontend_mentioned >= 2 and backend_mentioned == 0:
            roles.add('frontend')
        if not roles and backend_mentioned >= 1 and frontend_mentioned >= 1:
            roles.add('fullstack')

        # If still unknown, check for common patterns
        if not roles:
            if 'software engineer' in title or 'software developer' in title:
                roles.add('fullstack')  # Default to fullstack for generic roles

        return roles if roles else {'unknown'}

    def _parse_with_regex(self, job) -> ParsedJob:
        """
        Parse job using regex-based extraction (fast).

        This is always tried first, regardless of LLM settings.
        """
        # Fallback to regex-based parsing
        description = job.description.lower()

        # Find requirements section
        requirements_text = self._extract_requirements_section(job.description)

        # Extract skills from requirements section
        must_have = self._extract_skills(requirements_text)
        nice_to_have = self._extract_nice_to_have(job.description)

        # Check if description is truncated and supplement with title-based extraction
        is_truncated = self._is_description_truncated(job.description)
        if is_truncated:
            logger.debug(f"Job '{job.title}' appears to have truncated description, extracting from title")
            title_skills = self._extract_skills_from_title(job.title)

            # Add title-extracted skills to must-have since description is incomplete
            # This catches cases like "Senior Backend PHP Engineer" where PHP is only in title
            if title_skills:
                must_have.update(title_skills)
                logger.debug(f"Added {len(title_skills)} skills from title: {title_skills}")

        # Extract seniority requirements
        years, seniority = self._extract_seniority_requirements(job.description)

        # Extract job roles for filtering
        job_roles = self._extract_job_roles(job)

        return ParsedJob(
            title=job.title,
            company=job.company,
            location=job.location,
            description=job.description,
            apply_url=job.apply_url,
            source=job.source,
            must_have_skills=must_have,
            nice_to_have_skills=nice_to_have,
            min_years_experience=years,
            seniority_level=seniority,
            job_roles=job_roles,
            posted_date=job.posted_date.isoformat() if job.posted_date else None
        )

    def _extract_requirements_section(self, full_description: str) -> str:
        """Extract the requirements/qualifications section from description."""
        section = self._extract_section_by_headers(full_description, self.primary_requirement_headers)
        if section:
            return section

        section = self._extract_section_by_headers(full_description, self.secondary_requirement_headers)
        if section:
            return section

        return full_description

    def _extract_skills(self, text: str) -> Set[str]:
        """Extract must-have skills from text."""
        found_skills = set()

        for canonical, pattern in self.patterns.items():
            if pattern.search(text):
                found_skills.add(canonical)

        return found_skills

    def _extract_nice_to_have(self, full_description: str) -> Set[str]:
        """Extract nice-to-have skills from description."""
        found_skills = set()

        section = self._extract_section_by_headers(full_description, self.nice_to_have_headers)
        if section:
            found_skills.update(self._extract_skills(section))

        return found_skills

    def _extract_section_by_headers(self, full_description: str, headers: List[str]) -> str:
        """Extract the first section that follows any header in the list."""
        if not headers:
            return ""

        header_pattern = '|'.join(re.escape(h) for h in headers)
        sections = re.split(
            rf'\n\s*({header_pattern})\s*\n',
            full_description,
            flags=re.IGNORECASE
        )

        if len(sections) < 3:
            return ""

        for i in range(1, len(sections), 2):
            content = sections[i + 1] if i + 1 < len(sections) else ""
            if content:
                return self._truncate_section(content)

        return ""

    def _truncate_section(self, section_text: str) -> str:
        """Trim a section at the next header boundary."""
        parts = self.section_stop_pattern.split(section_text)
        if parts:
            return parts[0].strip()
        return section_text.strip()

    def _extract_seniority_requirements(self, description: str) -> tuple[Optional[float], str]:
        """Extract experience and seniority requirements."""
        # Extract years of experience
        years = None
        for pattern in self.experience_patterns:
            matches = pattern.findall(description)
            for match in matches:
                try:
                    years_val = float(match[0])
                    if years is None or years_val > years:
                        years = years_val
                except (ValueError, IndexError):
                    continue

        # Infer seniority level from title and description
        description_lower = description.lower()

        # Check for explicit seniority keywords
        if any(keyword in description_lower for keyword in ['senior', 'sr.', 'sr ', 'lead', 'principal', 'staff']):
            seniority = "senior"
        elif any(keyword in description_lower for keyword in ['junior', 'jr.', 'jr ', 'entry level', 'associate']):
            seniority = "junior"
        elif any(keyword in description_lower for keyword in ['mid-level', 'midlevel', 'intermediate']):
            seniority = "mid"
        else:
            seniority = "unknown"

        # Use years as additional signal
        if years:
            if years >= 7 and seniority == "unknown":
                seniority = "senior"
            elif years >= 3 and seniority == "unknown":
                seniority = "mid"
            elif years < 3 and seniority == "unknown":
                seniority = "junior"

        return years, seniority
