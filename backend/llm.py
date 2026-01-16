"""LLM integration for resume and job parsing in stateless mode."""

import os
import json
import logging
from typing import Optional, Dict, List, Any, Iterable

logger = logging.getLogger(__name__)


# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate and set model (fallback to safe default if invalid)
_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
_VALID_MODELS = [
    # Latest OpenAI models
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.1-codex-max",
    # Legacy models for backward compatibility
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]
if _MODEL not in _VALID_MODELS:
    logger.warning(f"Invalid OPENAI_MODEL '{_MODEL}', falling back to 'gpt-5-mini'. Valid models: {_VALID_MODELS}")
    _MODEL = "gpt-5-mini"

DEFAULT_MODEL = _MODEL


def _get_text_content(message_content: Any) -> str:
    """Extract plain text from OpenAI message content (str or list blocks)."""
    if message_content is None:
        return ""

    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, Iterable):
        parts = []
        for block in message_content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_val = block.get("text")
                    if isinstance(text_val, dict):
                        parts.append(text_val.get("value", ""))
                    elif isinstance(text_val, str):
                        parts.append(text_val)
            elif hasattr(block, "text"):
                text_val = getattr(block, "text", "")
                if isinstance(text_val, str):
                    parts.append(text_val)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join([p for p in parts if p])

    return str(message_content)


def _call_chat_completion(client, model: str, messages: List[Dict[str, str]], token_limit: int, **kwargs):
    """
    Call OpenAI chat completion handling API parameter differences.

    Prefers max_completion_tokens (newer models); falls back to max_tokens when not supported.
    """
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=token_limit,
            **kwargs,
        )
    except Exception as e:
        msg = str(e).lower()
        # Some legacy models require max_tokens instead
        if "max_completion_tokens" in msg and "unsupported" in msg:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=token_limit,
                **kwargs,
            )
        # If the reverse happens (max_tokens unsupported), retry with max_completion_tokens
        if "max_tokens" in msg and "unsupported" in msg:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=token_limit,
                **kwargs,
            )
        raise


def extract_profile(resume_text: str) -> Dict[str, Any]:
    """
    Extract structured profile from resume text using LLM.

    Args:
        resume_text: Raw text extracted from resume (PDF/DOCX/TXT)

    Returns:
        Dict with:
        {
            "skills": [str],          # normalized lowercase skills
            "seniority": str,         # "junior|mid|senior|staff|unknown"
            "role_focus": [str],      # e.g., ["backend", "frontend"]
            "years_experience": float|null,
            "keywords": [str]         # additional role keywords
        }

    Falls back to keyword extraction if OPENAI_API_KEY missing or LLM fails.
    """
    if not OPENAI_API_KEY:
        logger.info("OPENAI_API_KEY not set, using keyword extraction for profile")
        return _extract_profile_keywords(resume_text)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        system_prompt = """You are an expert resume analyzer. Extract a structured developer profile from resume text.

Extract:
1. Technical skills (languages, frameworks, tools, databases)
2. Seniority level (junior/mid/senior/staff)
3. Primary role category: ONE of [backend, frontend, fullstack, devops, data, mobile, qa, security]
4. Years of experience (explicit or inferred)
5. Specializations: specific domains (AI/ML, security, gaming, fintech, etc.)

Rules:
- Normalize skills to lowercase
- Group related skills (react/typescript/javascript are separate)
- If seniority is unclear, use "unknown"
- Only infer years if explicitly stated or clear from career progression
- Primary role: Pick the ONE best-fit category based on the candidate's main experience
- Specializations: Domain expertise like "ai/ml", "security", "fintech", "gaming", "embedded systems"
- DO NOT put technical skills in specializations

Return ONLY valid JSON. No markdown, no explanations."""

        user_prompt = f"""Analyze this resume and extract a developer profile:

{resume_text[:8000]}

Return JSON in this exact format:
{{
    "skills": ["python", "javascript", "react", "sql", ...],
    "seniority": "junior|mid|senior|staff|unknown",
    "role_focus": "backend|frontend|fullstack|devops|data|mobile|qa|security",
    "years_experience": <number or null>,
    "specializations": ["ai/ml", "fintech", "security", ...]
}}"""

        response = _call_chat_completion(
            client,
            DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            token_limit=2000,  # Increased from 1000 to avoid cutoff
        )

        # Debug: Log response metadata
        logger.debug(f"LLM response status: {response.choices[0].finish_reason}")
        logger.debug(f"LLM usage: {response.usage}")

        content = _get_text_content(response.choices[0].message.content)

        # Log the raw response for debugging
        if not content or not content.strip():
            logger.error(f"LLM returned empty response. Model: {DEFAULT_MODEL}, Finish reason: {response.choices[0].finish_reason}")
            raise ValueError("Empty LLM response")

        logger.debug(f"LLM raw response (first 200 chars): {content[:200]}")

        # Remove markdown code blocks if present
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
                elif content.startswith("javascript"):
                    content = content[10:]
                content = content.strip()

        # Try to extract JSON if there's still markdown
        if not content.strip().startswith("{"):
            # Look for JSON object in the response
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            else:
                logger.warning(f"Could not find valid JSON in response: {content[:200]}")
                raise ValueError("No JSON found in LLM response")

        result = json.loads(content)

        # Validate and normalize
        profile = _normalize_profile(result)
        logger.info(f"LLM profile extraction successful: {len(profile['skills'])} skills, seniority={profile['seniority']}")
        return profile

    except Exception as e:
        logger.warning(f"LLM profile extraction failed: {e}. Falling back to keyword extraction.")
        return _extract_profile_keywords(resume_text)


def _normalize_profile(raw_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize LLM-extracted profile."""
    # Normalize skills
    skills = raw_profile.get("skills", [])
    skills = [s.lower().strip() for s in skills if s and isinstance(s, str)]
    skills = list(set(skills))  # dedupe

    # Normalize seniority
    seniority_map = {
        "junior": "junior",
        "mid": "mid",
        "mid-level": "mid",
        "midlevel": "mid",
        "intermediate": "mid",
        "senior": "senior",
        "sr": "senior",
        "lead": "senior",
        "principal": "senior",
        "staff": "staff",
        "unknown": "unknown"
    }
    raw_seniority = raw_profile.get("seniority", "unknown").lower().strip()
    seniority = seniority_map.get(raw_seniority, "unknown")

    # Normalize role focus - handle both string and array formats
    valid_roles = {"backend", "frontend", "fullstack", "devops", "data", "mobile", "qa", "security"}

    role_focus_map = {
        "backend": "backend",
        "back-end": "backend",
        "back end": "backend",
        "frontend": "frontend",
        "front-end": "frontend",
        "front end": "frontend",
        "fullstack": "fullstack",
        "full-stack": "fullstack",
        "full stack": "fullstack",
        "devops": "devops",
        "dev-ops": "devops",
        "sre": "devops",
        "site reliability": "devops",
        "data": "data",
        "data science": "data",
        "machine learning": "data",
        "ml": "data",
        "ai": "data",
        "mobile": "mobile",
        "ios": "mobile",
        "android": "mobile",
        "qa": "qa",
        "quality assurance": "qa",
        "testing": "qa",
        "security": "security"
    }

    raw_role_input = raw_profile.get("role_focus", [])
    role_focus = []

    # Handle both string and array input
    if isinstance(raw_role_input, str):
        raw_roles = [raw_role_input]
    else:
        raw_roles = raw_role_input if isinstance(raw_role_input, list) else []

    for role in raw_roles:
        if isinstance(role, str):
            role_lower = role.lower().strip()
            normalized = role_focus_map.get(role_lower)
            if normalized and normalized in valid_roles:
                if normalized not in role_focus:
                    role_focus.append(normalized)

    # If no valid role found, default to unknown/empty
    if not role_focus:
        role_focus = []

    # Years experience
    years_experience = raw_profile.get("years_experience")
    if years_experience is not None:
        try:
            years_experience = float(years_experience)
            if years_experience < 0 or years_experience > 50:
                years_experience = None  # Invalid range
        except (ValueError, TypeError):
            years_experience = None

    # Handle both "specializations" (new) and "keywords" (old/compat)
    # Combine them for output as "keywords" for backwards compatibility
    keywords = []
    if raw_profile.get("specializations"):
        keywords.extend(raw_profile.get("specializations", []))
    if raw_profile.get("keywords"):
        keywords.extend(raw_profile.get("keywords", []))
    keywords = [k.strip() for k in keywords if k and isinstance(k, str)]
    keywords = list(set(keywords))  # dedupe

    return {
        "skills": skills,
        "seniority": seniority,
        "role_focus": role_focus,
        "years_experience": years_experience,
        "keywords": keywords
    }


def _extract_profile_keywords(resume_text: str) -> Dict[str, Any]:
    """
    Fallback keyword-based profile extraction.

    Uses existing ResumeParser logic as fallback.
    """
    from jobscout.resume_parser import ResumeParser

    parser = ResumeParser()
    parsed = parser._extract_from_text(resume_text)

    # Convert ParsedResume to profile format
    skills = sorted([s.lower() for s in parsed.skills])

    # Extract role focus from role keywords
    role_focus = []
    role_keywords_lower = [k.lower() for k in parsed.role_keywords]
    for keyword in role_keywords_lower:
        if "backend" in keyword or "back-end" in keyword:
            role_focus.append("backend")
        elif "frontend" in keyword or "front-end" in keyword:
            role_focus.append("frontend")
        elif "full" in keyword and "stack" in keyword:
            role_focus.append("fullstack")
        elif "devops" in keyword:
            role_focus.append("devops")
        elif "data" in keyword and ("engineer" in keyword or "scientist" in keyword):
            role_focus.append("data")
        elif "mobile" in keyword or "ios" in keyword or "android" in keyword:
            role_focus.append("mobile")

    role_focus = list(set(role_focus))

    return {
        "skills": skills,
        "seniority": parsed.seniority,
        "role_focus": role_focus,
        "years_experience": parsed.years_experience if parsed.years_experience > 0 else None,
        "keywords": parsed.role_keywords
    }


def extract_job_requirements(job_description: str, job_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Extract structured job requirements using LLM.

    Args:
        job_description: Full job description text
        job_metadata: Optional dict with title, company, location, etc.

    Returns:
        Dict with:
        {
            "must_haves": [str],
            "nice_to_haves": [str],
            "seniority": "junior|mid|senior|staff|null",
            "remote_eligible": true|false|null,
            "constraints": [str],  # e.g., "US only", "visa required"
            "years_experience": number|null
        }

    Falls back to keyword extraction if OPENAI_API_KEY missing or LLM fails.
    """
    if not OPENAI_API_KEY:
        logger.info("OPENAI_API_KEY not set, using keyword extraction for job requirements")
        return _extract_job_keywords(job_description)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        system_prompt = """You are an expert job description analyzer. Extract structured requirements from job postings.

Extract:
1. Must-have skills (explicitly required)
2. Nice-to-have skills (preferred but optional)
3. Seniority level (junior/mid/senior/staff)
4. Remote eligibility (can this role be done remotely?)
5. Constraints (location restrictions, visa requirements, etc.)
6. Years of experience required

Rules:
- Be specific about technologies (e.g., "postgresql" not "databases")
- Include versions if specified (e.g., "python 3.8+")
- Group related skills separately (react, typescript, javascript are different)
- If seniority is unclear, use null
- remote_eligible: true if explicitly remote/hybrid, false if onsite-only, null if unclear
- Constraints: location restrictions, visa requirements, timezone limits, etc.
- Only extract years if explicitly stated

Return ONLY valid JSON. No markdown, no explanations."""

        metadata_str = ""
        if job_metadata:
            metadata_str = f"\nJob Title: {job_metadata.get('title', '')}\nCompany: {job_metadata.get('company', '')}\nLocation: {job_metadata.get('location', '')}"

        user_prompt = f"""Analyze this job posting:{metadata_str}

{job_description[:8000]}

Return JSON in this exact format:
{{
    "must_haves": ["python", "react", "sql", ...],
    "nice_to_haves": ["docker", "kubernetes", ...],
    "seniority": "junior|mid|senior|staff|null",
    "remote_eligible": true|false|null,
    "constraints": ["US only", "visa required", ...],
    "years_experience": <number or null>
}}"""

        response = _call_chat_completion(
            client,
            DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            token_limit=2000,  # Increased from 1000 to avoid cutoff
        )

        # Debug: Log response metadata
        logger.debug(f"LLM response status: {response.choices[0].finish_reason}")
        logger.debug(f"LLM usage: {response.usage}")

        content = _get_text_content(response.choices[0].message.content)

        # Log the raw response for debugging
        if not content or not content.strip():
            logger.error(f"LLM returned empty response. Model: {DEFAULT_MODEL}, Finish reason: {response.choices[0].finish_reason}")
            raise ValueError("Empty LLM response")

        logger.debug(f"LLM raw response (first 200 chars): {content[:200]}")

        # Remove markdown code blocks if present
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
                elif content.startswith("javascript"):
                    content = content[10:]
                content = content.strip()

        # Try to extract JSON if there's still markdown
        if not content.strip().startswith("{"):
            # Look for JSON object in the response
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            else:
                logger.warning(f"Could not find valid JSON in response: {content[:200]}")
                raise ValueError("No JSON found in LLM response")

        result = json.loads(content)

        # Validate and normalize
        requirements = _normalize_job_requirements(result)
        logger.info(f"LLM job requirements extraction successful: {len(requirements['must_haves'])} must-haves")
        return requirements

    except Exception as e:
        logger.warning(f"LLM job requirements extraction failed: {e}. Falling back to keyword extraction.")
        return _extract_job_keywords(job_description)


def _normalize_job_requirements(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize LLM-extracted job requirements."""
    # Normalize skills
    must_haves = [s.lower().strip() for s in raw.get("must_haves", []) if s and isinstance(s, str)]
    must_haves = list(set(must_haves))  # dedupe

    nice_to_haves = [s.lower().strip() for s in raw.get("nice_to_haves", []) if s and isinstance(s, str)]
    nice_to_haves = list(set(nice_to_haves))

    # Normalize seniority
    seniority_map = {
        "junior": "junior",
        "mid": "mid",
        "mid-level": "mid",
        "midlevel": "mid",
        "intermediate": "mid",
        "senior": "senior",
        "sr": "senior",
        "lead": "senior",
        "principal": "senior",
        "staff": "staff"
    }
    raw_seniority = raw.get("seniority", "").lower().strip() if raw.get("seniority") else ""
    seniority = seniority_map.get(raw_seniority, None) if raw_seniority else None

    # Remote eligibility
    remote_eligible = raw.get("remote_eligible")
    if remote_eligible is None:
        remote_eligible = None

    # Constraints
    constraints = raw.get("constraints", [])
    constraints = [c.strip() for c in constraints if c and isinstance(c, str)]

    # Years experience
    years_experience = raw.get("years_experience")
    if years_experience is not None:
        try:
            years_experience = float(years_experience)
            if years_experience < 0 or years_experience > 50:
                years_experience = None
        except (ValueError, TypeError):
            years_experience = None

    return {
        "must_haves": must_haves,
        "nice_to_haves": nice_to_haves,
        "seniority": seniority,
        "remote_eligible": remote_eligible,
        "constraints": constraints,
        "years_experience": years_experience
    }


def _extract_job_keywords(job_description: str) -> Dict[str, Any]:
    """
    Fallback keyword-based job requirement extraction.

    Uses existing JobParser logic as fallback.
    """
    from jobscout.job_parser import JobParser
    from jobscout.job_sources.base import JobListing

    parser = JobParser()
    mock_job = JobListing(
        title="Unknown",
        company="Unknown",
        location="Unknown",
        description=job_description,
        apply_url="",
        source="Fallback"
    )

    parsed = parser.parse(mock_job)

    return {
        "must_haves": list(parsed.must_have_skills),
        "nice_to_haves": list(parsed.nice_to_have_skills),
        "seniority": parsed.seniority_level if parsed.seniority_level != "unknown" else None,
        "remote_eligible": None,  # Can't determine from keywords
        "constraints": [],
        "years_experience": parsed.min_years_experience
    }


def generate_match_explanation(
    job_requirements: Dict[str, Any],
    user_profile: Dict[str, Any],
    score_breakdown: Dict[str, float]
) -> str:
    """
    Generate a short "why matched" explanation for a job.

    Args:
        job_requirements: Extracted job requirements
        user_profile: User's profile from resume
        score_breakdown: Score breakdown with must_have_coverage, stack_overlap, seniority_alignment

    Returns:
        Short explanation string like:
        "Strong match: 4/5 must-have skills (python, react, sql), seniority aligned, remote-eligible"
    """
    try:
        if not OPENAI_API_KEY:
            return _generate_match_explanation_simple(job_requirements, user_profile, score_breakdown)

        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Build match details
        must_have_skills = job_requirements.get("must_haves", [])
        user_skills = user_profile.get("skills", [])
        matched_must_haves = [s for s in must_have_skills if s in user_skills]
        missing_must_haves = [s for s in must_have_skills if s not in user_skills]

        job_seniority = job_requirements.get("seniority", "unknown")
        user_seniority = user_profile.get("seniority", "unknown")

        remote_eligible = job_requirements.get("remote_eligible")

        prompt = f"""Generate a one-sentence explanation for why this job matches the candidate.

Job Requirements:
- Must-have skills: {must_have_skills[:10]}
- Seniority: {job_seniority}
- Remote: {remote_eligible}

Candidate Profile:
- Skills: {user_skills[:20]}
- Seniority: {user_seniority}

Match Details:
- Matched must-haves: {matched_must_haves}
- Missing must-haves: {missing_must_haves}
- Must-have coverage: {score_breakdown.get('must_have_coverage', 0):.0%}
- Stack overlap: {score_breakdown.get('stack_overlap', 0):.0%}
- Seniority alignment: {score_breakdown.get('seniority_alignment', 0):.0%}

Generate ONE sentence explaining why this job matches (or doesn't). Focus on:
- Skill overlap (specific skills)
- Seniority alignment
- Any concerns (missing must-haves, seniority mismatch)

Keep it concise and specific. Return just the sentence, no quotes."""

        response = _call_chat_completion(
            client,
            DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            token_limit=150,
        )

        explanation = _get_text_content(response.choices[0].message.content).strip().strip('"\'')
        return explanation

    except Exception as e:
        logger.warning(f"LLM match explanation failed: {e}. Using simple explanation.")
        return _generate_match_explanation_simple(job_requirements, user_profile, score_breakdown)


def _generate_match_explanation_simple(
    job_requirements: Dict[str, Any],
    user_profile: Dict[str, Any],
    score_breakdown: Dict[str, float]
) -> str:
    """Simple rule-based match explanation."""
    must_have_skills = job_requirements.get("must_haves", [])
    user_skills = user_profile.get("skills", [])
    matched = [s for s in must_have_skills if s in user_skills]
    missing = [s for s in must_have_skills if s not in user_skills]

    parts = []

    # Skill match
    if must_have_skills:
        match_pct = len(matched) / len(must_have_skills) if must_have_skills else 0
        if match_pct >= 0.8:
            parts.append(f"Excellent skill match ({len(matched)}/{len(must_have_skills)} must-haves)")
        elif match_pct >= 0.6:
            parts.append(f"Good skill match ({len(matched)}/{len(must_have_skills)} must-haves)")
        else:
            parts.append(f"Partial skill match ({len(matched)}/{len(must_have_skills)} must-haves)")

    # Seniority
    job_seniority = job_requirements.get("seniority")
    user_seniority = user_profile.get("seniority")
    if job_seniority and user_seniority:
        if job_seniority == user_seniority:
            parts.append("seniority aligned")
        elif (user_seniority == "senior" and job_seniority in ["junior", "mid"]) or \
             (user_seniority == "mid" and job_seniority == "junior"):
            parts.append("seniority level acceptable")
        else:
            parts.append("seniority mismatch")

    # Remote
    remote_eligible = job_requirements.get("remote_eligible")
    if remote_eligible is True:
        parts.append("remote-eligible")
    elif remote_eligible is False:
        parts.append("onsite role")

    return ", ".join(parts) if parts else "No strong match indicators"
