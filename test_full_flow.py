#!/usr/bin/env python3
"""Test full flow with the new changes."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add to path
sys.path.insert(0, os.path.dirname(__file__))

from jobscout.job_sources.boolean_search import BooleanSearchSource
from jobscout.job_parser import JobParser
from jobscout.config import JobScoutConfig
from jobscout.resume_parser import ParsedResume
from jobscout.role_recommender import RoleRecommender

def test_full_flow():
    """Test the complete job search flow with new features."""

    print("🧪 Testing Full Flow with New Changes")
    print("=" * 60)

    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    serper_key = os.getenv("SERPER_API_KEY")

    if not openai_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    if not serper_key:
        print("❌ SERPER_API_KEY not found")
        return False

    print("✅ API keys found")

    # Sample user profile
    user_skills = {
        "python", "django", "fastapi", "postgresql", "docker",
        "kubernetes", "redis", "github_actions", "aws"
    }
    user_seniority = "senior"
    user_years = 8.0

    # Test 1: Role Recommender
    print("\n1️⃣ Testing Role Recommender")
    print("-" * 60)

    try:
        advisor = RoleRecommender(api_key=openai_key)

        # Create a mock resume
        resume = ParsedResume(
            raw_text="Senior Python Engineer with 8 years of experience. Expert in Django, FastAPI, PostgreSQL, Docker, Kubernetes. Worked with AWS and Redis.",
            skills=user_skills,
            seniority=user_seniority,
            years_experience=user_years,
            role_keywords=[]
        )

        roles = advisor.recommend_roles(resume)
        print(f"✅ Role Recommender returned {len(roles)} roles:")
        for role in roles:
            print(f"   - {role}")

    except Exception as e:
        print(f"❌ Role Recommender failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Boolean Search with AI-recommended roles
    print("\n2️⃣ Testing Boolean Search with AI Roles")
    print("-" * 60)

    try:
        boolean_source = BooleanSearchSource(
            resume_skills=user_skills,
            role_keywords=roles or ["backend engineer"],  # Use AI roles
            seniority=user_seniority,
            location_preference="remote",
            max_job_age_days=7,
            serper_api_key=serper_key
        )

        # Show the query being built
        queries = boolean_source._build_boolean_queries()
        print(f"✅ Boolean query built:")
        print(f"   {queries[0][:100]}...")

    except Exception as e:
        print(f"❌ Boolean search setup failed: {e}")
        return False

    # Test 3: Job Parser with user context
    print("\n3️⃣ Testing Job Parser with User Context")
    print("-" * 60)

    try:
        config = JobScoutConfig(
            resume_path="",
            use_llm_parser=True,
            openai_api_key=openai_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini")
        )

        parser = JobParser(config)
        parser._user_seniority = user_seniority
        parser._user_years_experience = user_years

        print(f"✅ JobParser initialized with LLM support")
        print(f"   User seniority: {parser._user_seniority}")
        print(f"   User years: {parser._user_years_experience}")

    except Exception as e:
        print(f"❌ JobParser init failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Sample job parsing
    print("\n4️⃣ Testing Sample Job Parsing")
    print("-" * 60)

    try:
        from jobscout.job_sources.base import JobListing

        sample_job = JobListing(
            title="Senior Python Engineer",
            company="TestCorp",
            location="Remote",
            description="We are looking for a Senior Python Engineer with 5+ years of experience. Must have Python, Django, FastAPI, PostgreSQL, Docker, Kubernetes. Nice to have: Redis, AWS, GitHub Actions.",
            apply_url="https://example.com/job",
            source="Test"
        )

        parsed = parser.parse(sample_job, user_skills=user_skills)

        print(f"✅ Job parsed successfully")
        print(f"   Must-have skills: {sorted(parsed.must_have_skills)}")
        print(f"   Seniority: {parsed.seniority_level}")
        print(f"   Min years: {parsed.min_years_experience}")

    except Exception as e:
        print(f"❌ Job parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ All tests passed! Everything is working.")
    return True

if __name__ == "__main__":
    success = test_full_flow()
    sys.exit(0 if success else 1)
