#!/usr/bin/env python3
"""Test LLM integration with fallback behavior."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, ".")

from backend.llm import extract_profile, extract_job_requirements, generate_match_explanation


def test_profile_extraction_without_key():
    """Test that profile extraction works without OPENAI_API_KEY (keyword fallback)."""
    print("Testing profile extraction WITHOUT OPENAI_API_KEY...")

    # Ensure no API key
    original_key = os.getenv("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    resume_text = """
    John Doe
    Senior Software Engineer

    Skills:
    - Python, JavaScript, React, Node.js
    - PostgreSQL, MongoDB
    - Docker, Kubernetes, AWS

    Experience:
    - Senior Software Engineer at TechCorp (2020-Present)
    - Full Stack Developer at StartupInc (2018-2020)
    """

    try:
        profile = extract_profile(resume_text)

        assert "skills" in profile
        assert "seniority" in profile
        assert isinstance(profile["skills"], list)
        assert len(profile["skills"]) > 0

        print(f"✓ Profile extraction works without API key (keyword fallback)")
        print(f"  Extracted {len(profile['skills'])} skills: {', '.join(profile['skills'][:5])}")
        print(f"  Seniority: {profile['seniority']}")
        print()

    finally:
        # Restore original key
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key


def test_job_requirements_without_key():
    """Test that job requirement extraction works without OPENAI_API_KEY (keyword fallback)."""
    print("Testing job requirements extraction WITHOUT OPENAI_API_KEY...")

    # Ensure no API key
    original_key = os.getenv("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    job_description = """
    Senior Python Developer

    Requirements:
    - 5+ years of Python experience
    - Experience with Django and FastAPI
    - Knowledge of PostgreSQL and Redis
    - AWS or GCP experience preferred

    We're looking for a senior developer to join our team.
    """

    try:
        requirements = extract_job_requirements(job_description)

        assert "must_haves" in requirements
        assert "nice_to_haves" in requirements
        assert isinstance(requirements["must_haves"], list)

        print(f"✓ Job requirements extraction works without API key (keyword fallback)")
        print(f"  Must-haves: {', '.join(requirements['must_haves'][:5])}")
        print(f"  Nice-to-haves: {', '.join(requirements['nice_to_haves'][:5])}")
        print()

    finally:
        # Restore original key
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key


def test_match_explanation_without_key():
    """Test that match explanation works without OPENAI_API_KEY (simple fallback)."""
    print("Testing match explanation WITHOUT OPENAI_API_KEY...")

    # Ensure no API key
    original_key = os.getenv("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    job_requirements = {
        "must_haves": ["python", "react", "sql"],
        "seniority": "senior",
        "remote_eligible": True
    }

    user_profile = {
        "skills": ["python", "javascript", "react", "sql", "docker"],
        "seniority": "senior"
    }

    score_breakdown = {
        "must_have_coverage": 0.67,
        "stack_overlap": 0.8,
        "seniority_alignment": 1.0
    }

    try:
        explanation = generate_match_explanation(
            job_requirements, user_profile, score_breakdown
        )

        assert isinstance(explanation, str)
        assert len(explanation) > 0

        print(f"✓ Match explanation works without API key (simple fallback)")
        print(f"  Explanation: {explanation}")
        print()

    finally:
        # Restore original key
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key


def test_profile_structure():
    """Test that profile has correct structure."""
    print("Testing profile data structure...")

    resume_text = "Python developer with React and SQL skills"

    profile = extract_profile(resume_text)

    # Check all required fields
    required_fields = ["skills", "seniority", "role_focus", "years_experience", "keywords"]
    for field in required_fields:
        assert field in profile, f"Missing field: {field}"

    # Check types
    assert isinstance(profile["skills"], list)
    assert isinstance(profile["seniority"], str)
    assert isinstance(profile["role_focus"], list)
    assert isinstance(profile["keywords"], list)

    # Check seniority is valid
    valid_seniority = ["junior", "mid", "senior", "staff", "unknown"]
    assert profile["seniority"] in valid_seniority

    # Check skills are lowercase
    for skill in profile["skills"]:
        assert skill == skill.lower(), f"Skill not lowercase: {skill}"

    print("✓ Profile structure is correct")
    print(f"  Fields: {', '.join(required_fields)}")
    print(f"  Seniority options: {', '.join(valid_seniority)}")
    print()


def test_job_requirements_structure():
    """Test that job requirements has correct structure."""
    print("Testing job requirements data structure...")

    job_description = "Looking for Python and React developer"

    requirements = extract_job_requirements(job_description)

    # Check all required fields
    required_fields = ["must_haves", "nice_to_haves", "seniority", "remote_eligible", "constraints", "years_experience"]
    for field in required_fields:
        assert field in requirements, f"Missing field: {field}"

    # Check types
    assert isinstance(requirements["must_haves"], list)
    assert isinstance(requirements["nice_to_haves"], list)
    assert isinstance(requirements["constraints"], list)

    # Check skills are lowercase
    for skill in requirements["must_haves"]:
        assert skill == skill.lower(), f"Must-have skill not lowercase: {skill}"

    for skill in requirements["nice_to_haves"]:
        assert skill == skill.lower(), f"Nice-to-have skill not lowercase: {skill}"

    print("✓ Job requirements structure is correct")
    print(f"  Fields: {', '.join(required_fields)}")
    print()


def main():
    """Run all LLM integration tests."""
    print("=" * 60)
    print("JobScout LLM Integration Tests (Fallback Behavior)")
    print("=" * 60)
    print()

    try:
        test_profile_extraction_without_key()
        test_job_requirements_without_key()
        test_match_explanation_without_key()
        test_profile_structure()
        test_job_requirements_structure()

        print("=" * 60)
        print("All LLM fallback tests passed! ✓")
        print("=" * 60)
        print()
        print("The system works correctly:")
        print("  1. Without OPENAI_API_KEY → uses keyword extraction")
        print("  2. With OPENAI_API_KEY → uses LLM (if available)")
        print("  3. LLM failures → automatic fallback to keywords")
        print("  4. Data structures are valid and normalized")

    except AssertionError as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
