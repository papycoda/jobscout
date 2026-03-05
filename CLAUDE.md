# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## JobScout Architecture

JobScout is a conservative job-search assistant with two entry points:
- **CLI** (`jobscout_cli.py`): Single-run or scheduled job processing with state persistence
- **FastAPI Backend** (`backend/main.py`): Stateless REST API for frontend integration

### Core Processing Pipeline

Both entry points follow this flow:

1. **Fetch Jobs** (`job_sources/`) → From RSS feeds, APIs, and direct board scraping
2. **Cross-run Deduplication** (`dedup.py`) → Filter jobs seen in previous runs via JSON cache
3. **Parse Job Descriptions** (`job_parser.py`) → Extract must-have/nice-to-have skills (regex + optional LLM)
4. **Apply Hard Filters** (`filters.py`) → Exclude by location, age, content quality, role mismatch
5. **Score Jobs** (`scoring.py`) → Conservative scoring with language/framework gates
6. **Within-run Deduplication** → Remove duplicates by apply_url
7. **Email Digest** (`emailer.py`) → SMTP delivery or outbox fallback

### Key Modules

**`job_sources/`**: Job fetching implementations
- `base.py`: `JobListing` dataclass and `JobSource` ABC
- `company_boards.py`: Primary free source - scrapes Greenhouse/Lever/Ashby boards directly
- `rss_feeds.py`: RemoteOK, WeWorkRemotely, Himalayas, JavaScriptJobs
- `remotive_api.py`: Remotive API source
- `greenhouse_api.py` / `lever_api.py`: Company-specific board APIs
- `boolean_search.py`: Serper API-powered search (premium feature)

**`scoring.py`**: Conservative scoring system
- Weights: Must-have coverage (55%), Stack overlap (30%), Role alignment (10%), Seniority (5%)
- **Language/Framework Gate**: Candidate must have ALL required languages/frameworks (unless `soft_language_gate=True`)
- Title skill extraction (e.g., "Ruby Engineer" → Ruby is always required)
- Apply-ready threshold: 60% score + 60% must-have coverage + 2+ matching skills
- Role categories: backend, frontend, fullstack, devops, data, mobile

**`filters.py`**: Hard exclusion filters applied before scoring
- Missing/invalid apply URL, location mismatch, job too old
- Content quality checks (spam indicators, truncation detection)
- **Role mismatch**: Backend profile → no frontend-only roles (and vice versa)
- Fullstack is compatible with both backend and frontend

**`job_parser.py`**: Extracts skills from job descriptions
- Primary: Regex-based extraction using canonical skill dictionary
- Optional: LLM-enhanced parsing (`llm_parser.py`) when `OPENAI_API_KEY` is set
- Always extracts skills from job title (title skills are mandatory)

**`config.py`**: YAML-based configuration with dataclass validation
- Environment variable overrides supported (e.g., `JOBSCOUT_MIN_SCORE_THRESHOLD`)
- Nested configs: `EmailConfig`, `ScheduleConfig`, `JobPreferences`

### Important Patterns

**Dual AI Strategy**: Always-on regex parsing with optional LLM enhancement. LLM failures fall back gracefully to regex.

**Free vs API Sources**: Default to free sources (company boards, RSS feeds). API sources (Serper for Boolean search) are optional.

**Stateless vs Stateful**: CLI maintains state across runs (resume cache, dedup). Backend API is stateless (each request independent).

**Data Transformation Pipeline**: `JobListing` → `ParsedJob` → `ScoredJob` with immutable data objects.

**Role Compatibility**: Fullstack roles accept backend/frontend candidates and vice versa. Hard mismatches (backend profile → frontend-only job) are filtered.

## Common Commands

### Running the Application

```bash
# Single run (CLI)
python jobscout_cli.py

# Scheduled mode (daemon)
python jobscout_cli.py --schedule

# With custom config
python jobscout_cli.py --config my-config.yaml
```

### Running the Backend API

```bash
# Start FastAPI server
uvicorn backend.main:app --reload

# Production
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_scoring.py -v

# With coverage
pytest tests/ --cov=jobscout --cov-report=html
```

### Dependencies

```bash
# Install dependencies
pip install -r requirements.txt
```

## Configuration

Required environment variables:
- `OPENAI_API_KEY`: For LLM-enhanced job parsing (optional but recommended)
- `SERPER_API_KEY`: For Boolean search (optional)
- `SMTP_*`: For email delivery (optional - falls back to outbox)

See `config.example.yaml` for configuration structure.

## Key Files

- `jobscout_cli.py`: CLI entry point
- `jobscout/main.py`: Core orchestration (`JobScout.run_search()`)
- `jobscout/scoring.py`: Scoring algorithm and role logic
- `jobscout/filters.py`: Hard exclusion filters
- `jobscout/job_parser.py`: Job description parsing
- `backend/main.py`: FastAPI REST API
