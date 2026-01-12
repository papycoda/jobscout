#!/usr/bin/env python3
"""Test script for enhanced boolean search."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add to path
sys.path.insert(0, os.path.dirname(__file__))

from jobscout.job_sources.boolean_search import BooleanSearchSource

def test_boolean_search():
    """Test boolean search with sample resume data."""

    # Sample resume data (Python backend developer)
    resume_skills = {
        "python", "django", "fastapi", "postgresql", "docker",
        "kubernetes", "redis", "github_actions"
    }
    role_keywords = ["backend engineer", "python developer", "software engineer"]
    seniority = "mid"
    location_preference = "remote"
    max_job_age_days = 7

    # Get Serper API key
    serper_api_key = os.getenv("SERPER_API_KEY")

    if not serper_api_key:
        print("❌ SERPER_API_KEY not found in .env file")
        print("Please add: SERPER_API_KEY=your-key-here")
        return False

    print("🔍 Testing Enhanced Boolean Search")
    print("=" * 50)
    print(f"Skills: {sorted(resume_skills)}")
    print(f"Roles: {role_keywords}")
    print(f"Location: {location_preference}")
    print(f"Max age: {max_job_age_days} days")
    print("=" * 50)

    # Create boolean search source
    source = BooleanSearchSource(
        resume_skills=resume_skills,
        role_keywords=role_keywords,
        seniority=seniority,
        location_preference=location_preference,
        max_job_age_days=max_job_age_days,
        serper_api_key=serper_api_key
    )

    # Check the query it will build
    queries = source._build_boolean_queries()

    if not queries:
        print("❌ No queries generated")
        return False

    print("\n📝 Generated Query:")
    print("-" * 50)
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
    print("-" * 50)

    # Actually run the search
    print("\n🚀 Running Boolean Search...")
    print("-" * 50)

    try:
        jobs = source.fetch_jobs(limit=10)

        if not jobs:
            print("❌ No jobs found")
            return False

        print(f"✅ Found {len(jobs)} jobs!")

        for i, job in enumerate(jobs[:5], 1):
            print(f"\n{i}. {job.title}")
            print(f"   Company: {job.company}")
            print(f"   Location: {job.location}")
            print(f"   Source: {job.source}")
            print(f"   URL: {job.apply_url[:80]}...")

        if len(jobs) > 5:
            print(f"\n... and {len(jobs) - 5} more jobs")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_boolean_search()
    sys.exit(0 if success else 1)
