"""Tests for CompanyBoardsSource (free scraping-based job source)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock

from jobscout.job_sources.company_boards import (
    CompanyBoardsSource,
    discover_company_board,
    GREENHOUSE_COMPANIES,
    LEVER_COMPANIES,
    ASHBY_COMPANIES,
)


class TestCompanyBoardsSource:
    """Test CompanyBoardsSource functionality."""

    def test_initialization(self):
        """Test source initializes correctly."""
        source = CompanyBoardsSource(
            resume_skills={"python", "django"},
            role_keywords=["backend engineer"],
            location_preference="remote",
            max_job_age_days=7,
        )
        assert source.name == "CompanyBoards"
        assert source.resume_skills == {"python", "django"}
        assert source.location_preference == "remote"
        assert source.max_job_age_days == 7

    def test_get_companies_to_fetch_default(self):
        """Test default company list is returned."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
        )
        companies = source._get_companies_to_fetch()
        # Should have companies from curated lists
        assert len(companies) > 0
        assert "airbnb" in companies  # From Greenhouse list

    def test_get_companies_to_fetch_custom(self):
        """Test custom company list is used."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
            companies=["stripe", "vercel"],
        )
        companies = source._get_companies_to_fetch()
        assert companies == ["stripe", "vercel"]

    def test_is_greenhouse_company(self):
        """Test Greenhouse company detection."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
        )
        # Known Greenhouse company
        assert source._is_greenhouse_company("airbnb")
        assert source._is_greenhouse_company("stripe")
        # Custom domain with dot
        assert source._is_greenhouse_company("custom-domain.com")

    def test_is_lever_company(self):
        """Test Lever company detection."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
        )
        assert source._is_lever_company("netflix")
        assert source._is_lever_company("spotify")

    def test_is_ashby_company(self):
        """Test Ashby company detection."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
        )
        assert source._is_ashby_company("ramp")
        assert source._is_ashby_company("mercury")

    def test_is_relevant_job_with_skill_match(self):
        """Test relevant job detection with skill match."""
        source = CompanyBoardsSource(
            resume_skills={"python", "django", "fastapi"},
            role_keywords=["backend engineer"],
        )
        # Direct skill match
        assert source._is_relevant_job("Senior Python Developer")
        assert source._is_relevant_job("Django Backend Engineer")
        # Role match
        assert source._is_relevant_job("Backend Engineer")

    def test_is_relevant_job_excludes_non_tech(self):
        """Test non-tech roles are excluded."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
        )
        assert not source._is_relevant_job("Marketing Manager")
        assert not source._is_relevant_job("Sales Representative")
        assert not source._is_relevant_job("Customer Success Manager")
        assert not source._is_relevant_job("Recruiter")
        assert not source._is_relevant_job("Product Manager")

    def test_is_location_match_remote(self):
        """Test location matching for remote preference."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
            location_preference="remote",
        )
        # Accept remote
        assert source._is_location_match("Remote")
        assert source._is_location_match("Remote - US")
        assert source._is_location_match("Anywhere")
        assert source._is_location_match("Distributed Team")

        # Dict format (from API)
        assert source._is_location_match({"name": "Remote"})

    def test_is_location_match_hybrid(self):
        """Test location matching for hybrid preference."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
            location_preference="hybrid",
        )
        # Accept remote or hybrid
        assert source._is_location_match("Remote")
        assert source._is_location_match("Hybrid - San Francisco")
        assert source._is_location_match("New York - Hybrid")

    def test_is_location_match_onsite(self):
        """Test location matching for onsite preference."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
            location_preference="onsite",
        )
        # Accept onsite, reject remote
        assert source._is_location_match("San Francisco, CA")
        assert source._is_location_match("New York, NY")
        assert not source._is_location_match("Remote")
        assert not source._is_location_match("Anywhere")

    def test_is_age_valid_within_limit(self):
        """Test age validation for jobs within limit."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
            max_job_age_days=7,
        )

        now = datetime.now(timezone.utc)
        # Within limit - should pass
        assert source._is_age_valid(now)
        assert source._is_age_valid(now - timedelta(days=1))
        assert source._is_age_valid(now - timedelta(days=7))

    def test_is_age_valid_exceeds_limit(self):
        """Test age validation for jobs exceeding limit."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
            max_job_age_days=7,
        )

        now = datetime.now(timezone.utc)
        # Exceeds limit - should fail
        assert not source._is_age_valid(now - timedelta(days=8))
        assert not source._is_age_valid(now - timedelta(days=27))

    def test_is_age_valid_none_rejected(self):
        """Test that jobs without dates are rejected."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
            max_job_age_days=7,
        )
        # This is the critical fix - jobs without dates should be rejected
        assert not source._is_age_valid(None)

    def test_format_location_dict(self):
        """Test location formatting from dict."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
        )
        # Dict format
        assert source._format_location({"name": "Remote"}) == "Remote"
        assert source._format_location({"city": "SF", "state": "CA"}) == "SF, CA"

        # String format
        assert source._format_location("Remote") == "Remote"
        assert source._format_location("San Francisco, CA") == "San Francisco, CA"

    def test_parse_iso_date_valid(self):
        """Test ISO date parsing for valid formats."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
        )

        # Standard ISO format
        result = source._parse_iso_date("2024-01-15T10:30:00Z")
        assert result is not None
        assert isinstance(result, datetime)

        # ISO format with timezone
        result = source._parse_iso_date("2024-01-15T10:30:00+00:00")
        assert result is not None

        # ISO format with microseconds
        result = source._parse_iso_date("2024-01-15T10:30:00.123456+00:00")
        assert result is not None

    def test_parse_iso_date_invalid(self):
        """Test ISO date parsing for invalid formats."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
        )
        assert source._parse_iso_date(None) is None
        assert source._parse_iso_date("") is None
        assert source._parse_iso_date("invalid-date") is None

    def test_parse_greenhouse_date(self):
        """Test Greenhouse date parsing."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
        )

        # Test various date keys
        job = {"updated_at": "2024-01-15T10:30:00Z"}
        result = source._parse_greenhouse_date(job)
        assert result is not None

        job = {"createdAt": "2024-01-15T10:30:00Z"}
        result = source._parse_greenhouse_date(job)
        assert result is not None

    def test_parse_lever_date(self):
        """Test Lever date parsing."""
        source = CompanyBoardsSource(
            resume_skills={"python"},
            role_keywords=["engineer"],
        )

        posting = {"createdAt": "2024-01-15T10:30:00Z"}
        result = source._parse_lever_date(posting)
        assert result is not None


class TestDiscoverCompanyBoard:
    """Test company board discovery function."""

    @patch('jobscout.job_sources.company_boards.requests.head')
    def test_discover_greenhouse_company(self, mock_head):
        """Test discovering a Greenhouse company."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        result = discover_company_board("airbnb")
        assert result == "https://boards.greenhouse.io/airbnb/jobs"

    @patch('jobscout.job_sources.company_boards.requests.head')
    def test_discover_lever_company(self, mock_head):
        """Test discovering a Lever company."""
        # Greenhouse fails, Lever succeeds
        def side_effect(url, *args, **kwargs):
            response = Mock()
            if "greenhouse" in url:
                response.status_code = 404
            else:
                response.status_code = 200
            return response

        mock_head.side_effect = side_effect

        result = discover_company_board("netflix")
        assert "lever.co" in result

    @patch('jobscout.job_sources.company_boards.requests.head')
    def test_discover_unknown_company(self, mock_head):
        """Test discovering a company that doesn't exist."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_head.return_value = mock_response

        result = discover_company_board("nonexistent-company-12345")
        assert result is None
