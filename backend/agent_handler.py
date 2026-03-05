"""Agent-based search handler for backend API."""

import logging
import os
from typing import List, Optional, Set
from datetime import datetime

from jobscout.agent import JobScoutAgent
from jobscout.agent_models import CVProfile, JobEvaluation, AgentConfig
from jobscout.job_sources.base import JobListing
from jobscout.job_sources.fetcher import JobFetcher
from jobscout.emailer_agent import AgentEmailer


logger = logging.getLogger(__name__)


def run_agent_search(
    profile_data: dict,
    preferences: dict,
    send_digest: bool = False,
    to_email: Optional[str] = None,
) -> dict:
    """
    Run agent-based job search.

    Args:
        profile_data: Candidate profile from frontend (skills, seniority, etc.)
        preferences: Search preferences (location, job_boards, etc.)
        send_digest: Whether to send email digest
        to_email: Email address for digest

    Returns:
        Dict with matched_jobs, filtered_jobs, stats, email_status
    """
    import time
    start_time = time.time()

    # Step 1: Build CV profile from request data
    profile = _build_cv_profile(profile_data)

    # Step 2: Fetch jobs
    fetcher = JobFetcher(
        sources=preferences.get("job_boards"),
        location=preferences.get("location_preference", "remote"),
        company_boards_all=True,
    )

    search_queries = profile.search_keywords or _generate_fallback_search_terms(profile)
    logger.info(f"Search queries: {search_queries[:5]}...")

    jobs = fetcher.fetch_all(search_queries=search_queries, limit_per_source=50)
    logger.info(f"Fetched {len(jobs)} total jobs")

    if not jobs:
        return {
            "matched_jobs": [],
            "filtered_jobs": [],
            "stats": {
                "fetched": 0,
                "parsed": 0,
                "scored": 0,
                "matched": 0,
                "filtered": 0,
                "top_filter_reasons": [],
            },
        }

    # Step 3: Initialize agent
    agent = JobScoutAgent(AgentConfig(
        llm_provider="openai",
        max_results=preferences.get("max_results", 7),
    ))

    # Set the profile directly on the agent to skip re-analysis
    agent.cv_profile = profile

    # Step 4: Evaluate jobs
    evaluations = agent.evaluate_jobs_batch(jobs, profile)
    logger.info(f"Evaluated {len(evaluations)} jobs")

    # Step 5: Split matches and filtered
    matches = [e for e in evaluations if e.is_match]
    filtered = [e for e in evaluations if not e.is_match]

    # Sort matches by score
    matches.sort(key=lambda e: e.match_score, reverse=True)

    # Limit matches
    max_results = preferences.get("max_results", 7)
    matches = matches[:max_results]

    # Calculate stats
    filter_reason_counts = _count_filter_reasons(filtered)

    elapsed = time.time() - start_time
    logger.info(f"Search complete: {len(matches)} matches, {len(filtered)} filtered in {elapsed:.1f}s")

    # Step 6: Send email if requested
    email_status = None
    if send_digest and matches:
        email_status = _send_email_digest(
            matches,
            profile.name,
            len(evaluations),
            to_email,
        )

    return {
        "matched_jobs": [_format_match_for_api(e) for e in matches],
        "filtered_jobs": [_format_filtered_for_api(e) for e in filtered[:50]],
        "stats": {
            "fetched": len(jobs),
            "parsed": len(jobs),
            "scored": len(evaluations),
            "matched": len(matches),
            "filtered": len(filtered),
            "top_filter_reasons": _format_filter_reasons(filter_reason_counts),
        },
        "email": email_status,
    }


def _build_cv_profile(profile_data: dict) -> CVProfile:
    """Build CVProfile from frontend profile data."""
    # Extract skills
    skills = set(profile_data.get("skills", []))

    # Categorize skills (simple heuristic)
    languages, frameworks, databases, infrastructure = _categorize_skills(skills)

    # Build role keywords
    role_focus = profile_data.get("role_focus", []) or []
    keywords = profile_data.get("keywords", []) or []
    role_keywords = list({*role_focus, *keywords})

    # Generate search keywords from role and skills
    search_keywords = _generate_search_keywords(role_keywords, skills, profile_data.get("seniority", "unknown"))

    return CVProfile(
        name=profile_data.get("name") or "",
        role_primary=_infer_primary_role(role_keywords, skills),
        seniority=profile_data.get("seniority", "unknown"),
        skills=skills,
        languages=languages,
        frameworks=frameworks,
        databases=databases,
        infrastructure=infrastructure,
        years_experience=float(profile_data.get("years_experience", 0)),
        search_keywords=search_keywords,
        preferred_locations=[profile_data.get("location_preference", "remote")],
    )


def _categorize_skills(skills: Set[str]) -> tuple[Set[str], Set[str], Set[str], Set[str]]:
    """Categorize skills into languages, frameworks, databases, infrastructure."""
    skills_lower = {s.lower() for s in skills}

    # Language sets (normalized)
    language_keywords = {
        'python', 'javascript', 'typescript', 'java', 'go', 'golang', 'c#', 'c++',
        'ruby', 'php', 'rust', 'swift', 'kotlin', 'scala', 'dart', 'r', 'sql',
    }

    # Framework sets
    framework_keywords = {
        'django', 'fastapi', 'flask', 'spring', 'express', 'nest', 'nestjs',
        'react', 'vue', 'angular', 'svelte', 'next', 'nextjs', 'nuxt', 'remix',
        'rails', 'laravel', 'symfony', 'angularjs', 'jquery',
    }

    # Database sets
    database_keywords = {
        'postgresql', 'postgres', 'mysql', 'sqlite', 'mongodb', 'redis',
        'elasticsearch', 'dynamodb', 'cassandra', 'neo4j', 'influx', 'timescale',
    }

    # Infrastructure sets
    infra_keywords = {
        'docker', 'kubernetes', 'k8s', 'aws', 'gcp', 'azure', 'terraform',
        'ansible', 'jenkins', 'gitlab', 'github', 'ci/cd', 'cicd', 'linux',
        'nginx', 'apache', 'vercel', 'netlify', 'heroku', 'lambda', 'ec2',
    }

    languages = set()
    frameworks = set()
    databases = set()
    infrastructure = set()

    for skill in skills_lower:
        if any(lang in skill or skill in lang for lang in language_keywords):
            languages.add(skill)
        elif any(fw in skill or skill in fw for fw in framework_keywords):
            frameworks.add(skill)
        elif any(db in skill or skill in db for db in database_keywords):
            databases.add(skill)
        elif any(inf in skill or skill in inf for inf in infra_keywords):
            infrastructure.add(skill)
        else:
            # Uncategorized - add to all for now, agent will figure it out
            frameworks.add(skill)

    return languages, frameworks, databases, infrastructure


def _infer_primary_role(role_keywords: List[str], skills: Set[str]) -> str:
    """Infer primary role from keywords and skills."""
    role_text = ' '.join(role_keywords).lower()
    skills_text = ' '.join(skills).lower()

    combined = role_text + ' ' + skills_text

    # Check for role indicators
    if 'backend' in combined or 'back-end' in combined or 'server side' in combined:
        return 'backend'
    if 'frontend' in combined or 'front-end' in combined or 'front end' in combined:
        return 'frontend'
    if 'fullstack' in combined or 'full-stack' in combined or 'full stack' in combined:
        return 'fullstack'
    if 'devops' in combined or 'sre' in combined or 'site reliability' in combined:
        return 'devops'
    if 'data' in combined or 'machine learning' in combined or 'ml engineer' in combined:
        return 'data'
    if 'mobile' in combined or 'ios' in combined or 'android' in combined:
        return 'mobile'

    # Default to backend if python/go/java etc. are present
    backend_langs = {'python', 'go', 'golang', 'java', 'ruby', 'php', 'rust'}
    if any(lang in skills_text for lang in backend_langs):
        return 'backend'

    return 'unknown'


def _generate_search_keywords(
    role_keywords: List[str],
    skills: Set[str],
    seniority: str,
) -> List[str]:
    """Generate diverse search keywords."""
    keywords = set()

    # Add role keywords
    keywords.update(k.lower() for k in role_keywords)

    # Add skill-based keywords
    top_skills = list(skills)[:10]
    for skill in top_skills:
        keywords.add(f"{skill} developer")
        keywords.add(f"{skill} engineer")

    # Add seniority-prefixed terms
    if seniority in ['senior', 'lead', 'principal']:
        for role in list(role_keywords)[:3]:
            keywords.add(f"senior {role}")

    # Ensure we have enough keywords
    if not keywords:
        keywords = ['software engineer', 'developer', 'backend engineer', 'fullstack developer']

    return list(keywords)[:15]


def _generate_fallback_search_terms(profile: CVProfile) -> List[str]:
    """Generate fallback search terms if none provided."""
    terms = []

    # Role-based
    if profile.role_primary and profile.role_primary != 'unknown':
        terms.append(f"{profile.role_primary} engineer")
        terms.append(f"{profile.role_primary} developer")

    # Skill-based (top 5)
    for skill in list(profile.skills)[:5]:
        terms.append(f"{skill} developer")

    # Add generic terms
    terms.extend(['software engineer', 'developer', 'engineer'])

    return terms[:10]


def _format_match_for_api(evaluation: JobEvaluation) -> dict:
    """Format a matched job for API response."""
    return {
        "id": _hash_url(evaluation.url),
        "title": evaluation.job_title,
        "company": evaluation.company,
        "location": evaluation.location,
        "posted_at": evaluation.posted_date,
        "url": evaluation.url,
        "source": evaluation.source,
        "description": "",  # Not included in evaluation
        "score_total": round(evaluation.match_score, 1),
        "breakdown": {
            "must_have_coverage": evaluation.match_score / 100,  # Approximate
            "stack_overlap": evaluation.match_score / 100,
            "seniority_alignment": 1.0 if evaluation.seniority_aligned else 0.5,
        },
        "must_have": {
            "matched": evaluation.required_skills_matched,
            "missing": evaluation.required_skills_missing,
        },
        "stack": {
            "matched": evaluation.required_skills_matched,
            "missing": evaluation.required_skills_missing,
        },
        "seniority": {
            "expected": "",  # Not captured in evaluation
            "found": "",  # Not captured in evaluation
            "explanation": "Aligned" if evaluation.seniority_aligned else "May not align",
        },
        "match_explanation": evaluation.why_match or evaluation.summary,
    }


def _format_filtered_for_api(evaluation: JobEvaluation) -> dict:
    """Format a filtered job for API response."""
    reasons = []

    if not evaluation.role_aligned:
        reasons.append(f"Role mismatch (you: backend, job: {evaluation.role_primary or 'unknown'})")
    if evaluation.required_skills_missing:
        reasons.append(f"Missing required skills: {', '.join(evaluation.required_skills_missing[:3])}")
    if not evaluation.location_compatible:
        reasons.append("Location mismatch")
    if not evaluation.is_match:
        reasons.append(evaluation.summary or "Below match threshold")

    return {
        "id": _hash_url(evaluation.url),
        "title": evaluation.job_title,
        "company": evaluation.company,
        "location": evaluation.location,
        "url": evaluation.url,
        "score_total": round(evaluation.match_score, 1),
        "reasons": reasons[:3],
    }


def _hash_url(url: str) -> str:
    """Generate a short hash from URL for job ID."""
    import hashlib
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _count_filter_reasons(filtered: List[JobEvaluation]) -> dict:
    """Count reasons for filtering."""
    counts = {}

    for job in filtered:
        if not job.role_aligned:
            counts["Role mismatch"] = counts.get("Role mismatch", 0) + 1
        if job.required_skills_missing:
            counts["Missing skills"] = counts.get("Missing skills", 0) + 1
        if not job.location_compatible:
            counts["Location mismatch"] = counts.get("Location mismatch", 0) + 1
        if job.match_score < 60:
            counts["Low match score"] = counts.get("Low match score", 0) + 1

    return counts


def _format_filter_reasons(counts: dict) -> List[dict]:
    """Format filter reasons for API response."""
    return [
        {"reason": reason, "count": count}
        for reason, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]


def _send_email_digest(
    matches: List[JobEvaluation],
    candidate_name: str,
    total_evaluated: int,
    to_email: str,
) -> dict:
    """Send email digest with matched jobs."""
    smtp_config = {
        "smtp_host": os.getenv("SMTP_HOST"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "smtp_user": os.getenv("SMTP_USER"),
        "smtp_password": os.getenv("SMTP_PASS"),
        "smtp_from": os.getenv("SMTP_FROM", f"JobScout <{os.getenv('SMTP_USER', 'noreply@example.com')}>"),
        "to_address": to_email,
    }

    emailer = AgentEmailer(smtp_config=smtp_config)
    return emailer.send_digest(matches, candidate_name, total_evaluated)
