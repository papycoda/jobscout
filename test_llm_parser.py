#!/usr/bin/env python3
"""Test script for LLM job parser."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jobscout.llm_parser import LLMJobParser


def test_llm_parser():
    """Test the LLM job parser with a sample job description."""

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Run: export OPENAI_API_KEY='sk-...'")
        return False

    print("🔑 API key found")

    # Sample job description
    job_description = """
    Senior Backend Engineer - Remote

    We're looking for a Senior Backend Engineer to join our growing team.

    Requirements:
    - 5+ years of experience building backend services
    - Strong proficiency in Python or Go
    - Experience with PostgreSQL and Redis
    - Knowledge of AWS/GCP cloud services
    - Experience with microservices architecture
    - Bachelor's degree in CS or equivalent

    Nice to have:
    - Experience with Kubernetes
    - Familiarity with GraphQL
    - Contributed to open source projects

    We offer:
    - Competitive salary ($150k-$200k)
    - Remote-first culture
    - Health insurance
    - Unlimited PTO

    Apply at: https://example.com/apply/123
    """

    job_metadata = {
        "title": "Senior Backend Engineer",
        "company": "TechCorp Inc.",
        "location": "Remote",
        "apply_url": "https://example.com/apply/123",
        "source": "Test"
    }

    print("\n📄 Sample Job Description:")
    print("=" * 80)
    print(job_description[:200] + "...")
    print("=" * 80)

    # Test with gpt-5-mini (latest cost-effective)
    print("\n🤖 Testing LLM parser with gpt-5-mini...")

    try:
        parser = LLMJobParser(api_key=api_key, model="gpt-5-mini")

        result = parser.parse(job_description, job_metadata)

        print("\n✅ LLM Parsing successful!")
        print("\n📊 Extracted Information:")
        print(f"  Title: {result.title}")
        print(f"  Company: {result.company}")
        print(f"  Location: {result.location}")
        print(f"  Seniority: {result.seniority_level}")
        print(f"  Min Years: {result.min_years_experience}")

        print(f"\n  Must-Have Skills ({len(result.must_have_skills)}):")
        for skill in sorted(result.must_have_skills):
            print(f"    • {skill}")

        print(f"\n  Nice-to-Have Skills ({len(result.nice_to_have_skills)}):")
        for skill in sorted(result.nice_to_have_skills):
            print(f"    • {skill}")

        return True

    except Exception as e:
        print(f"\n❌ LLM Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison():
    """Compare LLM vs regex parsing."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return False

    job_description = """
    Senior Full Stack Developer

    We need a developer with:
    - React, TypeScript, and Node.js experience
    - PostgreSQL database skills
    - AWS deployment experience
    - 5+ years of professional experience

    Bonus points for:
    - Docker/Kubernetes
    - GraphQL
    - CI/CD pipeline experience
    """

    print("\n🔬 Comparing LLM vs Regex parsing...")

    # Test LLM parser
    try:
        llm_parser = LLMJobParser(api_key=api_key, model="gpt-5-mini")
        llm_result = llm_parser.parse(job_description, {
            "title": "Senior Full Stack Developer",
            "company": "TestCo",
            "location": "Remote",
            "apply_url": "https://test.com",
            "source": "Test"
        })

        print("\n🤖 LLM Results:")
        print(f"  Must-have: {sorted(llm_result.must_have_skills)}")
        print(f"  Nice-to-have: {sorted(llm_result.nice_to_have_skills)}")
        print(f"  Seniority: {llm_result.seniority_level}")

    except Exception as e:
        print(f"❌ LLM parsing failed: {e}")
        return False

    # Test regex parser (fallback)
    try:
        from jobscout.job_parser import JobParser
        from jobscout.job_sources.base import JobListing

        regex_parser = JobParser()
        mock_job = JobListing(
            title="Senior Full Stack Developer",
            company="TestCo",
            location="Remote",
            description=job_description,
            apply_url="https://test.com",
            source="Test"
        )
        regex_result = regex_parser.parse(mock_job)

        print("\n🔍 Regex Results:")
        print(f"  Must-have: {sorted(regex_result.must_have_skills)}")
        print(f"  Nice-to-have: {sorted(regex_result.nice_to_have_skills)}")
        print(f"  Seniority: {regex_result.seniority_level}")

        print("\n📈 Comparison:")
        print(f"  LLM found {len(llm_result.must_have_skills)} must-haves vs regex {len(regex_result.must_have_skills)}")
        print(f"  LLM found {len(llm_result.nice_to_have_skills)} nice-to-haves vs regex {len(regex_result.nice_to_have_skills)}")

    except Exception as e:
        print(f"❌ Regex parsing failed: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🧪 JobScout LLM Parser Test")
    print("=" * 80)

    # Test basic parsing
    success = test_llm_parser()

    if success:
        print("\n" + "=" * 80)
        # Test comparison
        test_comparison()

    print("\n" + "=" * 80)
    print("✨ Test complete!")
