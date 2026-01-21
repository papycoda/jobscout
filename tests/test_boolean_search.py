"""Tests for Boolean search query building."""

from datetime import datetime

from jobscout.job_sources.boolean_search import BooleanSearchSource


def test_boolean_query_includes_filters():
    source = BooleanSearchSource(
        resume_skills={"python", "django"},
        role_keywords=["backend engineer"],
        seniority="junior",
        location_preference="remote",
        max_job_age_days=7,
        serper_api_key="test",
        now=datetime(2025, 1, 10)
    )

    queries = source._build_boolean_queries()

    assert queries
    # Check that domains are in the site:(...) clause format
    assert any("boards.greenhouse.io" in q for q in queries)
    assert any("jobs.lever.co" in q for q in queries)
    assert any("jobs.breezy.hr" in q for q in queries)
    assert any("jobs.ashbyhq.com" in q for q in queries)
    # Verify the site: clause exists with proper grouping
    assert any("site:(" in q for q in queries)
    assert all('after:2025-01-03' in q for q in queries)
    assert all('"remote"' in q for q in queries)
    assert any('-"senior"' in q for q in queries)


def test_boolean_search_skips_without_key():
    source = BooleanSearchSource(
        resume_skills={"python"},
        role_keywords=["backend engineer"],
        seniority="mid",
        location_preference="remote",
        max_job_age_days=7,
        serper_api_key=None,
        now=datetime(2025, 1, 10)
    )

    assert source.fetch_jobs(limit=5) == []


def test_breezy_hr_link_validation():
    source = BooleanSearchSource(
        resume_skills={"python"},
        role_keywords=["backend engineer"],
        seniority="mid",
        location_preference="remote",
        max_job_age_days=7,
        serper_api_key="test",
        now=datetime(2025, 1, 10)
    )

    # Valid Breezy HR URLs
    assert source._is_supported_link("https://jobs.breezy.hr/company/position-name")
    assert source._is_supported_link("http://jobs.breezy.hr/acme-corp/senior-engineer")

    # Invalid Breezy HR URLs (missing path components)
    assert not source._is_supported_link("https://jobs.breezy.hr/")
    assert not source._is_supported_link("https://jobs.breezy.hr/company/")


def test_breezy_hr_company_extraction():
    source = BooleanSearchSource(
        resume_skills={"python"},
        role_keywords=["backend engineer"],
        seniority="mid",
        location_preference="remote",
        max_job_age_days=7,
        serper_api_key="test",
        now=datetime(2025, 1, 10)
    )

    # Test company name extraction from URL
    assert source._company_from_url("https://jobs.breezy.hr/acme-corp/senior-engineer") == "Acme Corp"
    assert source._company_from_url("https://jobs.breezy.hr/stripe/backend-engineer") == "Stripe"

    # Test title cleaning - Breezy titles typically have company at the end after "at"
    assert source._company_from_title("Senior Engineer at Acme Corp - Breezy") == "Acme Corp"
    assert source._company_from_title("Backend Developer at Stripe - Breezy HR") == "Stripe"
