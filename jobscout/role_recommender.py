"""LLM-powered role recommendations based on a candidate's resume."""

import json
import logging
from typing import List

from .llm_providers import MultiProviderLLMClient, DEFAULT_MODEL
from .resume_parser import ParsedResume


logger = logging.getLogger(__name__)


class RoleRecommender:
    """Recommend role keywords to drive Boolean search and filtering."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.model_id = model
        self.client = MultiProviderLLMClient(model_id=model, api_key=api_key)

    def recommend_roles(self, resume: ParsedResume, max_roles: int = 5) -> List[str]:
        """
        Recommend concise role keywords for search queries based on resume fit.

        Returns lowercase titles like "backend engineer", "fullstack engineer",
        "devops engineer", etc.
        """
        if not resume:
            return []

        # Keep context light to control token usage
        resume_excerpt = (resume.raw_text or "")[:2000]
        skills_list = sorted(resume.skills) if resume.skills else []

        prompt = f"""You are a tech career coach. Recommend up to {max_roles} concise job search role keywords for this candidate.

Profile:
- Seniority: {resume.seniority}
- Years of Experience: {resume.years_experience}
- Skills: {skills_list}

Resume excerpt (may be truncated):
{resume_excerpt}

Guidelines:
- Use short, realistic role phrases (2-4 words) like "backend engineer", "frontend engineer", "fullstack engineer", "data engineer", "devops engineer", "mobile engineer".
- Align with the candidate's skills and seniority.
- Avoid vague terms ("software professional") and avoid duplicates.
- Lowercase everything.

Return JSON only:
{{
  "role_keywords": ["role1", "role2", "..."]
}}"""

        try:
            content = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": "You recommend precise job roles for search queries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,  # This model only supports temperature=1.0
                max_tokens=500,
            )

            # Strip possible markdown fences
            if content.startswith("```"):
                content = content.split("```", 2)[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            roles = [
                r.strip().lower()
                for r in data.get("role_keywords", [])
                if isinstance(r, str) and r.strip()
            ]
            return roles[:max_roles]

        except Exception as exc:
            logger.warning(f"Role recommendation failed: {exc}")
            return []
