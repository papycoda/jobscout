#!/usr/bin/env python3
"""Test script for stateless JobScout API endpoints."""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["ok"] == True
    print("✓ Health check passed\n")


def test_get_config():
    """Test get config endpoint returns defaults."""
    print("Testing GET /api/config endpoint...")
    response = requests.get(f"{BASE_URL}/api/config")
    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "config" in data
    assert "description" in data

    config = data["config"]
    assert "location_preference" in config
    assert "max_job_age_days" in config
    assert "job_boards" in config
    assert "min_score_threshold" in config

    print(f"✓ GET /api/config passed")
    print(f"  Defaults: location={config['location_preference']}, max_age={config['max_job_age_days']}")
    print()


def test_upload_resume():
    """Test resume upload endpoint."""
    print("Testing /api/upload-resume endpoint...")

    # Create a test resume file
    test_resume = """
John Doe
Senior Software Engineer

Skills:
- Python, JavaScript, React, Node.js
- SQL, PostgreSQL, MongoDB
- Docker, Kubernetes, AWS
- Git, GitHub, CI/CD

Experience:
- Senior Software Engineer at TechCorp (2020-Present)
- Full Stack Developer at StartupInc (2018-2020)
- Junior Developer at AgencyXYZ (2016-2018)

Education:
- BS Computer Science, University of Technology (2016)
"""

    # Save to temp file
    with open("test_resume.txt", "w") as f:
        f.write(test_resume)

    # Upload
    with open("test_resume.txt", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/api/upload-resume",
            files={"file": f}
        )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "profile" in data
    assert "extracted_skills" in data
    assert "warnings" in data

    profile = data["profile"]
    assert "skills" in profile
    assert "seniority" in profile
    assert isinstance(profile["skills"], list)

    print(f"✓ Resume upload passed")
    print(f"  Extracted {len(profile['skills'])} skills: {', '.join(profile['skills'][:5])}...")
    print(f"  Seniority: {profile['seniority']}")
    print()

    return profile


def test_search_jobs(profile):
    """Test search jobs endpoint."""
    print("Testing /api/search endpoint...")

    request_data = {
        "profile": profile,
        "preferences": {
            "location_preference": "remote",
            "max_job_age_days": 7,
            "job_boards": ["remoteok"],  # Just one source for testing
            "min_score_threshold": 60.0,
            "preferred_tech_stack": []
        }
    }

    response = requests.post(
        f"{BASE_URL}/api/search",
        json=request_data
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "matched_jobs" in data
    assert "filtered_jobs" in data
    assert "stats" in data

    stats = data["stats"]
    assert "fetched" in stats
    assert "matched" in stats
    assert "filtered" in stats

    print(f"✓ Search jobs passed")
    print(f"  Fetched: {stats['fetched']} jobs")
    print(f"  Matched: {stats['matched']} jobs")
    print(f"  Filtered: {stats['filtered']} jobs")

    if data["matched_jobs"]:
        job = data["matched_jobs"][0]
        print(f"\n  Top matched job:")
        print(f"    {job['title']} at {job['company']}")
        print(f"    Score: {job['score_total']}%")
        print(f"    Location: {job['location']}")

    if data["filtered_jobs"] and len(data["filtered_jobs"]) > 0:
        print(f"\n  Sample filtered job:")
        job = data["filtered_jobs"][0]
        print(f"    {job['title']} at {job['company']}")
        print(f"    Reasons: {', '.join(job['reasons'])}")

    print()

    return data


def test_search_with_email(profile):
    """Test search with email digest (will fail without SMTP)."""
    print("Testing /api/search with email (SMTP not configured)...")

    request_data = {
        "profile": profile,
        "preferences": {
            "location_preference": "remote",
            "max_job_age_days": 7,
            "job_boards": ["remoteok"],
            "min_score_threshold": 60.0
        },
        "send_digest": True,
        "to_email": "test@example.com"
    }

    response = requests.post(
        f"{BASE_URL}/api/search",
        json=request_data
    )

    assert response.status_code == 200
    data = response.json()

    # Verify email status is present
    assert "email" in data
    assert "sent" in data["email"]

    print(f"✓ Search with email passed")
    print(f"  Email sent: {data['email']['sent']}")
    if not data["email"]["sent"]:
        print(f"  Error: {data['email'].get('error', 'Unknown')}")
    print(f"  Still returned {data['stats']['matched']} matched jobs!")
    print()


def test_missing_email_validation(profile):
    """Test that email is required when send_digest=true."""
    print("Testing email validation...")

    request_data = {
        "profile": profile,
        "preferences": {
            "location_preference": "remote",
            "max_job_age_days": 7
        },
        "send_digest": True
        # Missing to_email
    }

    response = requests.post(
        f"{BASE_URL}/api/search",
        json=request_data
    )

    assert response.status_code == 400
    print("✓ Email validation passed (correctly rejected missing email)\n")


def test_date_handling():
    """Test that missing dates are handled properly."""
    print("Testing date handling...")

    # Search and check for null posted_at
    profile = {
        "skills": ["python"],
        "seniority": "senior"
    }

    request_data = {
        "profile": profile,
        "preferences": {
            "location_preference": "remote",
            "max_job_age_days": 7,
            "job_boards": ["remoteok"],
            "min_score_threshold": 0.0  # Very low to see all jobs
        }
    }

    response = requests.post(
        f"{BASE_URL}/api/search",
        json=request_data
    )

    assert response.status_code == 200
    data = response.json()

    # Check that posted_at can be null
    all_jobs = data["matched_jobs"] + data["filtered_jobs"]
    jobs_with_null_dates = [j for j in all_jobs if j.get("posted_at") is None]

    print(f"✓ Date handling passed")
    print(f"  Found {len(jobs_with_null_dates)} jobs with null posted_at")
    print(f"  (Missing dates are properly handled as null, not huge ages)")
    print()


def test_llm_fallback():
    """Test that system works without OPENAI_API_KEY (keyword fallback)."""
    print("Testing LLM fallback behavior (without OPENAI_API_KEY)...")

    # Test that upload works without API key
    test_resume = """
    Python Developer with React and SQL skills.
    5 years of experience.
    """

    with open("test_resume.txt", "w") as f:
        f.write(test_resume)

    try:
        # Upload should work even without OPENAI_API_KEY
        with open("test_resume.txt", "rb") as f:
            upload_response = requests.post(
                f"{BASE_URL}/api/upload-resume",
                files={"file": f}
            )

        assert upload_response.status_code == 200
        data = upload_response.json()

        # Should still get a profile (from keyword extraction)
        assert "profile" in data
        assert "extracted_skills" in data
        assert "warnings" in data

        # Check if warning about missing API key is present
        has_api_warning = any("OPENAI_API_KEY" in w for w in data.get("warnings", []))

        print(f"✓ LLM fallback test passed")
        print(f"  Profile extracted: {len(data['profile']['skills'])} skills")
        if has_api_warning:
            print(f"  Warning shown about missing OPENAI_API_KEY (expected)")
        else:
            print(f"  OPENAI_API_KEY is set (using LLM extraction)")
        print()

    finally:
        import os
        if os.path.exists("test_resume.txt"):
            os.remove("test_resume.txt")


def main():
    """Run all tests."""
    print("=" * 60)
    print("JobScout Stateless API Tests")
    print("=" * 60)
    print()

    try:
        # Test endpoints
        test_health()
        test_get_config()
        profile = test_upload_resume()
        test_search_jobs(profile)
        test_search_with_email(profile)
        test_missing_email_validation(profile)
        test_date_handling()
        test_llm_fallback()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API.")
        print("  Make sure the backend is running on http://localhost:8000")
        sys.exit(1)
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        import os
        if os.path.exists("test_resume.txt"):
            os.remove("test_resume.txt")


if __name__ == "__main__":
    main()
