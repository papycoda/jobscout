"""Agent-based resume analysis for backend API."""

import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)


async def analyze_resume(
    file_content: bytes,
    filename: str,
) -> dict:
    """
    Analyze uploaded resume using agent.

    Args:
        file_content: Resume file content
        filename: Name of the file

    Returns:
        Dict with profile data matching frontend expectations
    """
    # Extract text from file
    resume_text = _extract_text(file_content, filename)

    # Use agent to analyze
    from jobscout.agent import JobScoutAgent
    from jobscout.agent_models import AgentConfig

    agent = JobScoutAgent(AgentConfig(llm_provider="openai"))

    try:
        profile = agent.analyze_cv(resume_text)

        # Build warnings
        warnings = []
        if not profile.skills:
            warnings.append("No skills detected - resume may need better formatting")

        return {
            "profile": {
                "skills": sorted(profile.skills),
                "seniority": profile.seniority,
                "role_focus": [profile.role_primary] if profile.role_primary else [],
                "years_experience": profile.years_experience,
                "keywords": profile.search_keywords,
            },
            "extracted_skills": sorted(profile.skills),
            "warnings": warnings,
        }

    except Exception as e:
        logger.error(f"Resume analysis failed: {e}")
        # Fallback to keyword extraction
        return _fallback_analysis(resume_text)


def _extract_text(content: bytes, filename: str) -> str:
    """Extract text from PDF, DOCX, or TXT file."""
    suffix = Path(filename).suffix.lower()

    if suffix == '.txt':
        return content.decode('utf-8', errors='ignore')

    elif suffix == '.pdf':
        import io
        import pdfplumber
        text_chunks = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_chunks.append(page_text)
        return "\n".join(text_chunks)

    elif suffix == '.docx':
        import io
        import docx
        doc = docx.Document(io.BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _fallback_analysis(resume_text: str) -> dict:
    """Fallback analysis using simple keyword extraction."""
    # Simple keyword extraction
    skills = set()
    skill_keywords = [
        'python', 'javascript', 'typescript', 'java', 'go', 'rust', 'ruby',
        'django', 'fastapi', 'flask', 'react', 'vue', 'angular', 'node',
        'postgresql', 'mysql', 'mongodb', 'redis', 'docker', 'kubernetes',
        'aws', 'gcp', 'azure', 'git', 'linux', 'sql', 'nosql',
    ]

    text_lower = resume_text.lower()
    for keyword in skill_keywords:
        if keyword in text_lower:
            skills.add(keyword)

    return {
        "profile": {
            "skills": sorted(skills),
            "seniority": "unknown",
            "role_focus": [],
            "years_experience": 0,
            "keywords": [],
        },
        "extracted_skills": sorted(skills),
        "warnings": ["LLM analysis failed, using keyword extraction"],
    }
