# JobScout

A scheduled job-search assistant that emails you ONLY jobs you can realistically apply to without editing your resume.

**Conservatism is a feature.** If a job is borderline, unclear, or aspirational, JobScout excludes it.

## What JobScout Is (and Isn't)

**JobScout IS:**
- A quiet, trustworthy assistant that surfaces "apply-now" jobs
- Conservative by design — missing a good job is better than wasting your time
- Focused on high-confidence matches that align with your actual resume

**JobScout is NOT:**
- A job explorer or recommendation engine
- A dashboard-heavy app
- A tool for aspirational or stretch applications

## Features

- **Resume Parsing**: Extract skills from PDF, DOCX, or TXT resumes
- **Conservative Scoring**: 65% match threshold with 60% must-have skill coverage
- **Multiple Job Sources**: RemoteOK, We Work Remotely, Remotive, and Boolean search
- **Hard Exclusion Filters**: Location, age, content quality, missing data
- **Email Digests**: Clean, calm emails with max 10 high-confidence matches
- **Scheduling**: Daily or weekday runs with timezone support
- **Outbox Fallback**: Writes emails to files if SMTP not configured

## Quick Start

### 1. Install Dependencies

```bash
cd jobscout
pip install -r requirements.txt
```

### 2. Configure JobScout

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` with your details:

```yaml
# REQUIRED: Path to your resume
resume_path: "./resume.pdf"

# Email delivery
email:
  enabled: true
  to_address: "your-email@example.com"
  # Optional: Configure SMTP to send real emails
  # smtp_host: "smtp.gmail.com"
  # smtp_port: 587
  # smtp_username: "your-email@example.com"
  # smtp_password: "your-app-password"

# Scheduling
schedule:
  enabled: true
  frequency: "daily"  # or "weekdays"
  time: "09:00"
  timezone: "America/New_York"

# Job preferences
job_preferences:
  location_preference: "remote"  # "remote", "hybrid", "onsite", or "any"
  max_job_age_days: 7
```

### 3. Run JobScout

**Manual run (test):**
```bash
python jobscout_cli.py
```

**Scheduled mode (daemon):**
```bash
python jobscout_cli.py --schedule
```

**With custom config:**
```bash
python jobscout_cli.py --config my-config.yaml
```

### 4. Check Results

If SMTP is not configured, emails are written to `./outbox/`:

```bash
ls -la outbox/
# jobscout_digest_20240105_090001.html
```

## How It Works

### Runtime Flow

1. **Parse Resume** → Extract skills, seniority, experience
2. **Fetch Jobs** → From RSS feeds, APIs, and Boolean search
3. **Apply Hard Filters** → Exclude missing URLs, wrong location, too old, low quality
4. **Parse Job Descriptions** → Extract must-have and nice-to-have skills
5. **Score Conservatively** → 60% must-have coverage, 25% stack overlap, 15% seniority
6. **Keep Only Apply-Ready** → Score ≥ 65% AND must-have coverage ≥ 60%
7. **Deduplicate** → Within a single run only
8. **Email Digest** → Max 10 jobs per email

### Scoring System

JobScout uses a weighted scoring system:

- **Must-Have Coverage (60%)**: How many required skills you have
- **Stack Overlap (25%)**: Overall skill alignment with job
- **Seniority Alignment (15%)**: Experience level match

**Apply-Ready Criteria:**
- Score ≥ 65%
- AND must-have coverage ≥ 60%

**Special Case:**
- If job has no clear must-have skills → require score ≥ 75%

### Job Sources

JobScout fetches from multiple sources:

1. **RemoteOK** (RSS)
   - Remote tech jobs
   - Auto-filtered to dev roles

2. **We Work Remotely** (RSS)
   - Programming category only

3. **Remotive** (API)
   - Tech jobs filtered by category

4. **Boolean Search** (Simulated in MVP)
   - Greenhouse (boards.greenhouse.io)
   - Lever (jobs.lever.co)
   - Hard-capped at 30 results per run

### Hard Exclusion Rules

Jobs are immediately discarded if:

- Apply URL is missing
- Location conflicts with preference
- Job is older than `max_job_age_days` (default: 7)
- Posting date is missing AND source isn't guaranteed fresh
- Boolean-sourced page doesn't show an active job
- Content is spammy, vague, or low-signal

### Resume Skill Extraction

JobScout uses a canonical skill dictionary with 50+ skills across:

- **Languages**: Python, JavaScript, TypeScript, Go, Java, C#, Ruby, Rust
- **Frameworks**: Django, FastAPI, Flask, Spring, Express, NestJS
- **Infrastructure**: Docker, Kubernetes, AWS, GCP, Azure, Terraform
- **Databases**: PostgreSQL, MySQL, Redis, MongoDB, Elasticsearch
- **Messaging**: Kafka, RabbitMQ, SQS
- **Testing/CI**: pytest, JUnit, GitHub Actions, GitLab CI

## Configuration

### Email Settings

**Option 1: SMTP (recommended for production)**

```yaml
email:
  enabled: true
  to_address: "your-email@example.com"
  smtp_host: "smtp.gmail.com"
  smtp_port: 587
  smtp_username: "your-email@example.com"
  smtp_password: "your-app-password"  # Use app password, not regular password
  smtp_from: "JobScout <noreply@jobscout.example.com>"
```

**Option 2: Outbox (default)**

If SMTP is not configured, emails are written to `./outbox/` as HTML files.

### Scheduling

```yaml
schedule:
  enabled: true
  frequency: "daily"    # "daily" or "weekdays"
  time: "09:00"         # 24-hour format
  timezone: "America/New_York"  # IANA timezone name
```

**Available frequencies:**
- `daily`: Every day at specified time
- `weekdays`: Mon-Fri only

### Job Preferences

```yaml
job_preferences:
  # Optional: Add preferred tech stack (lower weight than resume skills)
  preferred_tech_stack:
    - "rust"
    - "graphql"

  # Location preference
  location_preference: "remote"  # "remote", "hybrid", "onsite", or "any"

  # Optional: Specify which job boards to use
  # If empty, uses all defaults
  job_boards:
    - "remoteok"
    - "remotive"
    - "weworkremotely"
    # - "boolean"  # Disabled in MVP

  # Maximum job age
  max_job_age_days: 7
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=jobscout --cov-report=html
```

## Project Structure

```
jobscout/
├── jobscout/
│   ├── __init__.py
│   ├── main.py              # Main orchestration
│   ├── config.py            # Configuration management
│   ├── resume_parser.py     # Resume parsing and skill extraction
│   ├── job_parser.py        # Job description parsing
│   ├── filters.py           # Hard exclusion filters
│   ├── scoring.py           # Conservative scoring system
│   ├── emailer.py           # Email delivery
│   ├── scheduler.py         # Job scheduling
│   └── job_sources/
│       ├── __init__.py
│       ├── base.py          # Base classes
│       ├── rss_feeds.py     # RemoteOK, We Work Remotely
│       ├── remotive_api.py  # Remotive API
│       └── boolean_search.py # Boolean search (Greenhouse/Lever)
├── tests/
│   ├── test_scoring.py
│   └── test_job_sources.py
├── outbox/                  # Email files when SMTP not configured
├── config.example.yaml
├── requirements.txt
└── README.md
```

## Troubleshooting

### No jobs in email

JobScout is conservative by design. If you're not seeing jobs:

1. **Check logs**: Look for why jobs are being filtered
   ```bash
   python jobscout_cli.py --config config.yaml
   ```

2. **Review thresholds**: Consider lowering `min_score_threshold` (default: 65)

3. **Check preferences**: Ensure `location_preference` and `max_job_age_days` are reasonable

4. **Verify resume**: Ensure your resume skills are being extracted correctly

### Resume not parsing correctly

- Ensure resume is PDF, DOCX, or TXT format
- Check that text is extractable (not scanned images)
- Review logs for extracted skills

### SMTP not working

- Use app-specific password (not account password)
- Check firewall/port settings
- Verify SMTP settings for your email provider

## Development Priorities

When enhancing JobScout, prioritize:

1. **Conservatism > coverage** — Better to miss a good job than include a bad one
2. **Trust > cleverness** — Transparent scoring, clear explanations
3. **Shipping > perfection** — Working MVP today vs. perfect system tomorrow

## License

MIT License — Feel free to use and modify for your job search.

## Contributing

This is an MVP built in ~1 day. Enhancements welcome:

- Boolean search integration with search APIs
- Additional job sources
- Enhanced skill extraction
- LLM integration for job parsing
- Persistence layer for job history
- Web interface

Remember: Conservatism is the product, not a bug to fix.
