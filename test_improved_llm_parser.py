#!/usr/bin/env python3
"""Test script for improved LLM job parser."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add to path
sys.path.insert(0, os.path.dirname(__file__))

from jobscout.llm_parser import LLMJobParser

def test_improved_llm_parser():
    """Test the improved LLM parser with user context."""

    # Sample job description (from our boolean search results)
    job_description = """
We're looking for a Senior Python Engineer to join our growing team.

Requirements:
- 5+ years of professional Python experience
- Strong experience with Django and FastAPI
- Proficient in PostgreSQL database design and optimization
- Experience with Docker and Kubernetes for containerization
- Knowledge of AWS services (EC2, S3, RDS)
- Familiarity with Redis for caching
- Experience with CI/CD pipelines (GitHub Actions)

Nice to have:
- Experience with GraphQL
- Knowledge of Celery for async tasks
- Familiarity with monitoring tools (Prometheus, Grafana)

Benefits:
- Competitive salary
- Remote work
- Health insurance
"""

    job_metadata = {
        "title": "Senior Python Engineer",
        "company": "TechCorp",
        "location": "Remote",
        "source": "Boolean Search"
    }

    # User profile (Python backend developer)
    user_skills = {
        "python", "django", "fastapi", "postgresql", "docker",
        "kubernetes", "redis", "github_actions", "aws"
    }
    user_seniority = "senior"
    user_years_experience = 8.0

    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        return False

    print("🧪 Testing Improved LLM Parser")
    print("=" * 60)
    print(f"Job: {job_metadata['title']} at {job_metadata['company']}")
    print(f"User Skills: {sorted(user_skills)}")
    print(f"User Seniority: {user_seniority}")
    print(f"User Years Experience: {user_years_experience}")
    print("=" * 60)

    # Create LLM parser
    try:
        parser = LLMJobParser(api_key=api_key)
    except Exception as e:
        print(f"❌ Failed to initialize LLM parser: {e}")
        return False

    # Parse job with user context
    print("\n🔍 Parsing job with user context...")
    print("-" * 60)

    try:
        parsed_job = parser.parse(
            job_description=job_description,
            job_metadata=job_metadata,
            user_skills=user_skills,
            user_seniority=user_seniority,
            user_years_experience=user_years_experience
        )

        print("✅ Parsing successful!")
        print("\n📊 Results:")
        print(f"  Must-Have Skills ({len(parsed_job.must_have_skills)}):")
        for skill in sorted(parsed_job.must_have_skills):
            match = "✓" if skill in user_skills else "✗"
            print(f"    {match} {skill}")

        print(f"\n  Nice-to-Have Skills ({len(parsed_job.nice_to_have_skills)}):")
        for skill in sorted(parsed_job.nice_to_have_skills):
            match = "✓" if skill in user_skills else "✗"
            print(f"    {match} {skill}")

        print(f"\n  Seniority Level: {parsed_job.seniority_level}")
        print(f"  Min Years Experience: {parsed_job.min_years_experience}")

        # Calculate match statistics
        must_have_matches = parsed_job.must_have_skills & user_skills
        all_matches = (parsed_job.must_have_skills | parsed_job.nice_to_have_skills) & user_skills

        print(f"\n  📈 Match Statistics:")
        print(f"    Must-Have Matches: {len(must_have_matches)}/{len(parsed_job.must_have_skills)}")
        print(f"    Total Skill Matches: {len(all_matches)}")

        return True

    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_llm_parser()
    sys.exit(0 if success else 1)
