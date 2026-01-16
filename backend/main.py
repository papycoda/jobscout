"""FastAPI backend for JobScout."""

import os
import sys
import logging
import math
import time
import hashlib
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env file not required, can use system env vars

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.security import (
    SecurityHeadersMiddleware,
    APIKeyAuthMiddleware,
    RateLimitMiddleware,
    SecurityValidator,
    SecurityLogger,
    SecurityConfig
)

# Add parent directory to path to import jobscout
sys.path.insert(0, str(Path(__file__).parent.parent))

from jobscout.config import JobScoutConfig
from jobscout.role_recommender import RoleRecommender
from backend.storage import Storage
from backend.adapter import JobScoutAdapter, send_email_digest_from_jobs
from backend.models import *
from backend.metrics import metrics_tracker
from backend.llm import generate_match_explanation


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize FastAPI
app = FastAPI(
    title="JobScout API",
    description="Conservative job search assistant backend"
)


# Security middleware (added before CORS for proper execution order)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(APIKeyAuthMiddleware)
app.add_middleware(RateLimitMiddleware)

# CORS configuration (with improved security)
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080,https://jobscoutpro.netlify.app").split(",")
# Validate and sanitize CORS origins
valid_origins = []
for origin in cors_origins:
    origin = origin.strip()
    if origin and origin.startswith(('http://', 'https://')):
        valid_origins.append(origin)
    else:
        logger.warning(f"Invalid CORS origin in config: {origin}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=valid_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Restrict methods
    allow_headers=["*"],
)


# Initialize storage
storage = Storage()


# Helper functions
def create_config_from_env():
    """
    Create JobScoutConfig from environment variables.

    This allows configuration without committing config.yaml.
    Falls back to loading from config.yaml if env vars not set.
    """
    import yaml
    from jobscout.config import EmailConfig, ScheduleConfig, JobPreferences

    # Check if we should use env vars
    use_env = os.getenv("USE_ENV_CONFIG", "false").lower() == "true"

    if use_env:
        def _split_env_list(var_name: str) -> list[str]:
            value = os.getenv(var_name, "")
            if not value:
                return []
            return [item.strip() for item in value.split(",") if item.strip()]

        # Build config from environment variables
        email = EmailConfig(
            enabled=os.getenv("EMAIL_ENABLED", "true").lower() == "true",
            to_address=os.getenv("SMTP_TO", ""),
            smtp_host=os.getenv("SMTP_HOST"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_username=os.getenv("SMTP_USER"),
            smtp_password=os.getenv("SMTP_PASS"),
            smtp_from=os.getenv("SMTP_FROM", f"JobScout <{os.getenv('SMTP_TO', 'noreply@example.com')}>")
        )

        schedule = ScheduleConfig(
            enabled=os.getenv("SCHEDULE_ENABLED", "false").lower() == "true",
            frequency=os.getenv("SCHEDULE_FREQUENCY", "weekdays"),
            time=os.getenv("SCHEDULE_TIME", "09:00"),
            timezone=os.getenv("SCHEDULE_TIMEZONE", "America/New_York")
        )

        job_prefs = JobPreferences(
            preferred_tech_stack=os.getenv("PREFERRED_TECH_STACK", "").split(",") if os.getenv("PREFERRED_TECH_STACK") else [],
            location_preference=os.getenv("LOCATION_PREFERENCE", "remote"),
            job_boards=[],  # Will use defaults if empty
            greenhouse_boards=_split_env_list("GREENHOUSE_BOARDS"),
            lever_companies=_split_env_list("LEVER_COMPANIES"),
            max_job_age_days=int(os.getenv("MAX_JOB_AGE_DAYS", "7"))
        )

        config = JobScoutConfig(
            resume_path="",  # Will be set after upload
            email=email,
            schedule=schedule,
            job_preferences=job_prefs,
            serper_api_key=os.getenv("SERPER_API_KEY"),
            outbox_dir=os.getenv("OUTBOX_DIR", "./outbox"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            min_score_threshold=float(os.getenv("MIN_SCORE_THRESHOLD", "65.0")),
            fallback_min_score=float(os.getenv("NO_MUST_HAVE_MIN_SCORE", "75.0"))
        )

        return config

    else:
        # Load from config.yaml
        config_yaml = storage.load_config()
        if not config_yaml:
            raise ValueError("No configuration found. Set USE_ENV_CONFIG=true or provide config.yaml")

        return load_config_from_yaml(config_yaml)


def get_or_create_config() -> JobScoutConfig:
    """Load config from env vars or config.yaml."""
    use_env = os.getenv("USE_ENV_CONFIG", "false").lower() == "true"

    if use_env:
        return create_config_from_env()

    # Load from config.yaml
    config_yaml = storage.load_config()

    if not config_yaml:
        raise HTTPException(
            status_code=404,
            detail="No configuration found. Set USE_ENV_CONFIG=true or create config.yaml"
        )

    try:
        config = JobScoutConfig.from_yaml_str(config_yaml)
        errors = config.validate()
        if errors:
            resume_profile = storage.load_latest_resume_profile()
            if resume_profile and "Resume file not found" in " ".join(errors):
                errors = [e for e in errors if not e.startswith("Resume file not found")]
            if errors:
                logger.warning(f"Config validation errors: {errors}")

        return config
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid configuration: {str(e)}"
        )


def load_config_from_yaml(config_yaml: str) -> JobScoutConfig:
    """Load config from YAML string."""
    try:
        # Use a temporary method to load from string
        import yaml
        from jobscout.config import EmailConfig, ScheduleConfig, JobPreferences

        data = yaml.safe_load(config_yaml)

        # Convert nested dicts
        email_data = data.pop("email", {})
        schedule_data = data.pop("schedule", {})
        job_prefs_data = data.pop("job_preferences", {})

        email = EmailConfig(**email_data)
        schedule = ScheduleConfig(**schedule_data)
        job_prefs = JobPreferences(**job_prefs_data)

        config = JobScoutConfig(
            email=email,
            schedule=schedule,
            job_preferences=job_prefs,
            **data
        )

        return config
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse configuration: {str(e)}"
        )


# Add from_yaml_str classmethod to JobScoutConfig
def _from_yaml_str(cls, yaml_str: str):
    """Load config from YAML string."""
    import yaml
    from jobscout.config import EmailConfig, ScheduleConfig, JobPreferences

    data = yaml.safe_load(yaml_str)

    email_data = data.pop("email", {})
    schedule_data = data.pop("schedule", {})
    job_prefs_data = data.pop("job_preferences", {})

    email = EmailConfig(**email_data)
    schedule = ScheduleConfig(**schedule_data)
    job_prefs = JobPreferences(**job_prefs_data)

    return cls(
        email=email,
        schedule=schedule,
        job_preferences=job_prefs,
        **data
    )


JobScoutConfig.from_yaml_str = classmethod(_from_yaml_str)


# Endpoints

@app.get("/health")
async def health_check_render():
    """Health check endpoint for Render (no /api prefix)."""
    return {"ok": True}

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check disk configuration. Only available when DEBUG=true."""
    import os
    from pathlib import Path

    if os.getenv("DEBUG", "").lower() != "true":
        raise HTTPException(
            status_code=404,
            detail="Not found"
        )

    data_dir = os.getenv("DATA_DIR", "/var/data")

    return {
        "data_dir_env": data_dir,
        "data_dir_exists": os.path.exists(data_dir),
        "data_dir_isdir": os.path.isdir(data_dir) if os.path.exists(data_dir) else False,
        "data_dir_writable": os.access(data_dir, os.W_OK) if os.path.exists(data_dir) else False,
        "cwd": os.getcwd(),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "env_vars": {
            "DATA_DIR": os.getenv("DATA_DIR"),
            "RESUMES_DIR": os.getenv("RESUMES_DIR"),
            "JOBSCOUT_DATA_DIR": os.getenv("JOBSCOUT_DATA_DIR"),
        }
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {"ok": True}


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics(request: Request):
    """
    Get application metrics for PMF tracking.

    Requires API key authentication.
    """
    # Always require auth for metrics (even when API key auth is disabled globally)
    from backend.security import SecurityConfig

    api_key = request.headers.get(SecurityConfig.API_KEY_HEADER)
    metrics_api_key = os.getenv("METRICS_API_KEY", os.getenv("API_KEY", ""))

    if not metrics_api_key or not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required for metrics"
        )

    import hmac
    if not hmac.compare_digest(api_key, metrics_api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return metrics_tracker.get_metrics()


@app.post("/api/metrics/reset")
async def reset_metrics(request: Request):
    """
    Reset all metrics (admin only).

    Requires API key authentication.
    """
    from backend.security import SecurityConfig

    api_key = request.headers.get(SecurityConfig.API_KEY_HEADER)
    metrics_api_key = os.getenv("METRICS_API_KEY", os.getenv("API_KEY", ""))

    if not metrics_api_key or not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required"
        )

    import hmac
    if not hmac.compare_digest(api_key, metrics_api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    metrics_tracker.reset()
    return {"ok": True, "message": "Metrics reset"}


@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload and parse a resume file (stateless - no disk storage).

    Extracts skills and profile from PDF/DOCX/TXT using LLM if available.
    Returns structured profile for use in /api/search.
    """
    try:
        # Enhanced file validation with security checks
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No filename provided"
            )

        # Validate file extension and size
        is_valid, error_msg = SecurityValidator.validate_file_upload(
            file.filename,
            file.size or 0  # size might be None
        )

        if not is_valid:
            SecurityLogger.log_security_event(
                "file_upload_validation_failed",
                {"filename": file.filename, "error": error_msg}
            )
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )

        # Read file content into memory (no disk write)
        content = await file.read()

        # Double-check actual file size
        if len(content) > SecurityConfig.MAX_FILE_SIZE:
            SecurityLogger.log_security_event(
                "file_size_exceeded",
                {"filename": file.filename, "size": len(content)}
            )
            raise HTTPException(
                status_code=400,
                detail=f"File too large (max {SecurityConfig.MAX_FILE_SIZE} bytes)"
            )

        # Extract text from file (in-memory)
        from jobscout.resume_parser import ResumeParser
        parser = ResumeParser()

        # Parse to get raw text (different logic for different file types)
        suffix = Path(file.filename).suffix.lower()
        if suffix == '.pdf':
            import io
            import pdfplumber
            text_chunks = []
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_chunks.append(page_text)
            resume_text = "\n".join(text_chunks)
        elif suffix == '.docx':
            import io
            import docx
            doc = docx.Document(io.BytesIO(content))
            resume_text = "\n".join([para.text for para in doc.paragraphs])
        else:  # .txt
            resume_text = content.decode('utf-8', errors='ignore')

        # Use LLM to extract profile (with fallback to keywords)
        from backend.llm import extract_profile
        profile = extract_profile(resume_text)

        # Build warnings
        warnings = []
        if not profile["skills"]:
            warnings.append("No skills detected - resume may need better formatting")

        if not os.getenv("OPENAI_API_KEY"):
            warnings.append("LLM extraction not available (OPENAI_API_KEY not set) - using keyword extraction")

        # Return results
        skills_list = sorted(profile["skills"])

        # Track metrics
        metrics_tracker.record_resume_upload(
            skills_count=len(profile["skills"]),
            seniority=profile.get("seniority", "unknown")
        )

        return {
            "profile": {
                "skills": skills_list,
                "seniority": profile["seniority"],
                "role_focus": profile["role_focus"],
                "years_experience": profile["years_experience"],
                "keywords": profile["keywords"]
            },
            "extracted_skills": skills_list,
            "warnings": warnings
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload resume: {str(e)}"
        )


@app.get("/api/config")
async def get_config():
    """
    Get default configuration (stateless, no disk access).

    Returns source-of-truth defaults used when user doesn't specify preferences.
    Frontend can override these by sending preferences in /api/search request.
    """
    try:
        # Build default job boards list (only include configured boards that can return data)
        default_boards = ["remoteok", "weworkremotely", "remotive"]
        if os.getenv("GREENHOUSE_BOARDS"):
            default_boards.append("greenhouse")
        if os.getenv("LEVER_COMPANIES"):
            default_boards.append("lever")
        # Boolean search is now enabled by default when SERPER_API_KEY is set
        if os.getenv("SERPER_API_KEY"):
            default_boards.append("boolean")

        # Return default config as JSON (no disk access)
        default_config = {
            "location_preference": "remote",
            "max_job_age_days": 7,
            "job_boards": default_boards,
            "min_score_threshold": 60.0,
            "preferred_tech_stack": [],
            "use_llm_parsing": False
        }

        return {
            "config": default_config,
            "description": "Default configuration. Override by sending preferences in /api/search request.",
            "note": "Job boards include: RemoteOK, We Work Remotely, Remotive, Greenhouse, Lever" + (", Boolean search with Greenhouse/Lever/Breezy HR/Ashby HQ (enabled by default when SERPER_API_KEY set)" if os.getenv("SERPER_API_KEY") else "")
        }

    except Exception as e:
        logger.error(f"Failed to load default config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load configuration: {str(e)}"
        )


@app.put("/api/config")
async def update_config(request: ConfigUpdateRequest):
    """
    Update configuration - DISABLED in stateless mode.

    In stateless mode, user preferences are sent with each /api/search request
    and are not persisted server-side. Use /api/search with preferences instead.
    """
    raise HTTPException(
        status_code=410,
        detail="Config persistence disabled in stateless mode. Send preferences with /api/search request instead."
    )


@app.post("/api/search-legacy", response_model=SearchResponse)
async def run_search_legacy():
    """
    Run a job search.

    Creates a new run, fetches and scores jobs, and saves results.
    Returns run_id for polling.
    """
    run_id, run_dir = storage.create_run_dir()

    # Write initial run metadata immediately
    from datetime import datetime
    storage.save_run_meta(run_dir, {
        "run_id": run_id,
        "status": "running",
        "phase": "starting",
        "message": "Initializing job search",
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "error": None
    })

    # Run in background would be ideal, but for simplicity we'll run synchronously
    # In production, you'd want to use Celery or similar for true async
    logger.info(f"Starting JobScout run {run_id}")

    try:
        # Check for resume
        resume_path = storage.get_latest_resume_path()
        resume_profile = storage.load_latest_resume_profile()

        if not resume_path and not resume_profile:
            raise HTTPException(
                status_code=400,
                detail="No resume uploaded. Please upload a resume first."
            )

        # Load config (from env vars or config.yaml)
        config = get_or_create_config()

        # Override resume path to use latest uploaded if present
        config.resume_path = resume_path or ""

        # Update phase: fetching
        storage.save_run_meta(run_dir, {
            "run_id": run_id,
            "status": "running",
            "phase": "fetching",
            "message": "Fetching jobs from sources",
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "error": None
        })

        # Run JobScout via adapter
        adapter = JobScoutAdapter(config, resume_profile=resume_profile)
        results = adapter.run_and_capture()

        # Update phase: completing
        storage.save_run_meta(run_dir, {
            "run_id": run_id,
            "status": "completed",
            "phase": "done",
            "message": f"Completed with {len(results['jobs'])} matching jobs",
            "started_at": results["metadata"]["start_time"],
            "finished_at": results["metadata"]["end_time"],
            "error": None,
            **{k: v for k, v in results["metadata"].items() if k not in ["start_time", "end_time"]}
        })

        # Save results
        storage.save_run_jobs(run_dir, results["jobs"])
        storage.save_run_filtered_jobs(run_dir, results["filtered_jobs"])

        logger.info(f"JobScout run {run_id} completed: {len(results['jobs'])} matching jobs")

        return SearchResponse(run_id=run_id)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"JobScout run {run_id} failed: {e}", exc_info=True)

        # Save error metadata
        storage.save_run_meta(run_dir, {
            "run_id": run_id,
            "status": "failed",
            "phase": "error",
            "message": "Search failed",
            "started_at": datetime.now().isoformat(),
            "finished_at": datetime.now().isoformat(),
            "error": str(e)
        })

        raise HTTPException(
            status_code=500,
            detail=f"Job search failed: {str(e)}"
        )


@app.post("/api/search")
async def search_jobs(request: dict):
    """
    Stateless job search endpoint (no disk persistence).

    Accepts profile from upload-resume and search preferences.
    Returns matched jobs, filtered jobs, and optional email digest.
    """
    try:
        from datetime import datetime
        import hashlib
        from jobscout.job_sources.rss_feeds import RemoteOKSource, WeWorkRemotelySource
        from jobscout.job_sources.remotive_api import RemotiveSource
        from jobscout.job_sources.greenhouse_api import GreenhouseSource
        from jobscout.job_sources.lever_api import LeverSource
        from jobscout.job_parser import JobParser
        from jobscout.scoring import JobScorer
        from jobscout.resume_parser import ParsedResume
        from jobscout.filters import HardExclusionFilters

        # Parse request
        profile_data = request.get("profile", {})
        preferences = request.get("preferences", {})
        send_digest = request.get("send_digest", False)
        to_email = request.get("to_email")

        # Validate email requirement
        if send_digest and not to_email:
            raise HTTPException(
                status_code=400,
                detail="to_email is required when send_digest=true"
            )

        # Build profile from request data
        user_skills = set(profile_data.get("skills", []))
        user_seniority = profile_data.get("seniority", "unknown")
        user_years = profile_data.get("years_experience", 0) or 0
        preferred_stack = set(preferences.get("preferred_tech_stack", []))
        location_pref = preferences.get("location_preference", "remote")
        max_age = preferences.get("max_job_age_days", 7)
        min_threshold = preferences.get("min_score_threshold", 60.0)
        llm_pref = preferences.get("use_llm_parsing")
        has_api_key = bool(os.getenv("OPENAI_API_KEY"))
        use_llm = (has_api_key and (llm_pref is None or bool(llm_pref)))

        # Default job boards (only include boards that can return data without extra config)
        default_boards = ["remoteok", "weworkremotely", "remotive"]
        if os.getenv("GREENHOUSE_BOARDS"):
            default_boards.append("greenhouse")
        if os.getenv("LEVER_COMPANIES"):
            default_boards.append("lever")
        # Boolean search is now enabled by default when SERPER_API_KEY is set
        if os.getenv("SERPER_API_KEY"):
            default_boards.append("boolean")

        job_boards = preferences.get("job_boards", default_boards)

        def _split_env_list(var_name: str) -> list[str]:
            value = os.getenv(var_name, "")
            if not value:
                return []
            return [item.strip() for item in value.split(",") if item.strip()]

        greenhouse_boards = preferences.get("greenhouse_boards") or _split_env_list("GREENHOUSE_BOARDS")
        lever_companies = preferences.get("lever_companies") or _split_env_list("LEVER_COMPANIES")

        # Create ParsedResume from profile
        role_focus = profile_data.get("role_focus", []) or []
        keywords = profile_data.get("keywords", []) or []
        resume_role_keywords = list({*role_focus, *keywords})

        resume = ParsedResume(
            raw_text="",
            skills=user_skills,
            tools=set(),
            seniority=user_seniority,
            years_experience=float(user_years),
            role_keywords=resume_role_keywords
        )

        # If the client didn't supply roles, use AI to recommend some for search queries
        role_keywords_for_search = list(resume_role_keywords) or ["software engineer"]
        if use_llm and has_api_key and not resume_role_keywords:
            try:
                advisor = RoleRecommender(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
                )
                recommended_roles = advisor.recommend_roles(resume)
                if recommended_roles:
                    role_keywords_for_search = recommended_roles
                    logger.info(f"Using AI-recommended role keywords: {role_keywords_for_search}")
            except Exception as exc:
                logger.warning(f"Role recommendation failed; using fallback keywords: {exc}")

        # Coarse role intent for filtering (backend/frontend/fullstack/etc.)
        def _infer_user_roles(role_keywords: list[str], skills: set) -> set:
            user_roles = set()

            for keyword in role_keywords or []:
                kw = keyword.lower()
                if 'backend' in kw:
                    user_roles.add('backend')
                if 'frontend' in kw or 'front-end' in kw:
                    user_roles.add('frontend')
                if 'fullstack' in kw or 'full-stack' in kw or 'full stack' in kw:
                    user_roles.add('fullstack')
                if 'devops' in kw or 'sre' in kw or 'site reliability' in kw:
                    user_roles.add('devops')
                if 'data' in kw or 'machine learning' in kw or 'ml ' in kw or kw.endswith(' ml'):
                    user_roles.add('data')
                if 'mobile' in kw or 'ios' in kw or 'android' in kw:
                    user_roles.add('mobile')

            if user_roles:
                return user_roles

            backend_indicators = {'python', 'django', 'fastapi', 'flask', 'java', 'go', 'ruby', 'php', 'rust'}
            frontend_indicators = {'react', 'vue', 'angular', 'javascript', 'typescript', 'css', 'html'}

            backend_count = sum(1 for skill in skills if skill in backend_indicators)
            frontend_count = sum(1 for skill in skills if skill in frontend_indicators)

            if backend_count >= 3 and backend_count > frontend_count * 3:
                user_roles.add('backend')
            elif frontend_count >= 3 and frontend_count > backend_count * 3:
                user_roles.add('frontend')
            elif backend_count >= 1 and frontend_count >= 1:
                user_roles.add('fullstack')

            return user_roles if user_roles else {'unknown'}

        user_role_categories = _infer_user_roles(role_keywords_for_search, user_skills)

        # Fetch jobs from selected sources
        logger.info(f"Fetching jobs from: {', '.join(job_boards)}")
        all_jobs = []

        for board in job_boards:
            try:
                if board.lower() == "remoteok":
                    source = RemoteOKSource("RemoteOK")
                    jobs = source.fetch_jobs(limit=50)
                    all_jobs.extend(jobs)
                elif board.lower() == "weworkremotely":
                    source = WeWorkRemotelySource("We Work Remotely")
                    jobs = source.fetch_jobs(limit=50)
                    all_jobs.extend(jobs)
                elif board.lower() == "remotive":
                    source = RemotiveSource("Remotive")
                    jobs = source.fetch_jobs(limit=50)
                    all_jobs.extend(jobs)
                elif board.lower() == "greenhouse":
                    # Greenhouse requires company list
                    if not greenhouse_boards:
                        logger.info("Skipping Greenhouse: no boards configured")
                        continue
                    source = GreenhouseSource(greenhouse_boards)
                    jobs = source.fetch_jobs(limit=50)
                    all_jobs.extend(jobs)
                elif board.lower() == "lever":
                    # Lever requires company list
                    if not lever_companies:
                        logger.info("Skipping Lever: no companies configured")
                        continue
                    source = LeverSource(lever_companies)
                    jobs = source.fetch_jobs(limit=50)
                    all_jobs.extend(jobs)
                elif board.lower() == "boolean":
                    # Boolean search using Serper API
                    from jobscout.job_sources.boolean_search import BooleanSearchSource
                    boolean_source = BooleanSearchSource(
                        resume_skills=user_skills,
                        role_keywords=role_keywords_for_search,
                        seniority=user_seniority,
                        location_preference=location_pref,
                        max_job_age_days=max_age,
                        serper_api_key=os.getenv("SERPER_API_KEY")
                    )
                    jobs = boolean_source.fetch_jobs(limit=30)
                    all_jobs.extend(jobs)
                logger.info(f"Fetched {len(jobs)} jobs from {board}")
            except Exception as e:
                logger.warning(f"Failed to fetch from {board}: {e}")
                continue

        logger.info(f"Total jobs fetched: {len(all_jobs)}")

        # Parse jobs using smart hybrid approach
        from jobscout.job_parser import JobParser
        from jobscout.config import JobScoutConfig
        parsed_jobs = []

        # Create parser with optional LLM support (default to on when API key present)
        parser_config = JobScoutConfig(
            resume_path="",
            use_llm_parser=use_llm,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini")
        )
        parser = JobParser(parser_config)
        parser._user_seniority = user_seniority
        parser._user_years_experience = float(user_years)

        # Use smart hybrid parsing (fast regex + selective LLM)
        if use_llm:
            logger.info("Using smart hybrid parsing (regex + selective LLM)")
            parsed_jobs = parser.parse_batch(all_jobs, user_skills=user_skills)
        else:
            logger.info("Using keyword extraction for all jobs (fast mode)")
            parsed_jobs = parser.parse_batch(all_jobs, user_skills=user_skills)

        # Store requirements in expected format for compatibility
        for parsed in parsed_jobs:
            remote_eligible = getattr(parsed, "remote_eligible", None)
            parsed._llm_requirements = {
                "must_haves": list(parsed.must_have_skills),
                "nice_to_haves": list(parsed.nice_to_have_skills),
                "seniority": parsed.seniority_level if parsed.seniority_level != "unknown" else None,
                "remote_eligible": remote_eligible,
                "constraints": [],
                "years_experience": parsed.min_years_experience
            }

        logger.info(f"Successfully parsed {len(parsed_jobs)} jobs")

        # Score and filter jobs
        from jobscout.config import JobScoutConfig, JobPreferences, EmailConfig, ScheduleConfig
        temp_config = JobScoutConfig(
            resume_path="",
            email=EmailConfig(enabled=False),
            schedule=ScheduleConfig(enabled=False),
            job_preferences=JobPreferences(location_preference=location_pref, max_job_age_days=max_age),
            min_score_threshold=min_threshold,
            fallback_min_score=min_threshold
        )

        scorer = JobScorer(resume, temp_config, preferred_stack)
        filters = HardExclusionFilters(
            temp_config,
            user_roles=user_role_categories if user_role_categories else None
        )

        # Process each job
        matched_jobs = []
        filtered_jobs = []
        filter_reason_counts = {}

        for job in parsed_jobs:
            reasons = []
            score = None
            is_matched = True
            job_requirements = getattr(job, '_llm_requirements', {})
            has_must_haves = bool(job.must_have_skills)

            # Apply hard exclusion filters (content quality, location, truncation, role mismatch, etc.)
            exclusion_reason = filters._check_exclusion(job)
            if exclusion_reason:
                reasons.append(exclusion_reason)
                is_matched = False

            # Score the job if it passed hard filters
            if is_matched:
                scored = scorer._score_job(job)
                score = scored.score

                # Check if apply-ready (FIX: proper must-have logic + minimum skill count)
                must_have_coverage = scored.must_have_coverage
                has_must_haves = len(job.must_have_skills) > 0

                # Calculate total matching skills (must-have + nice-to-have)
                all_job_skills = job.must_have_skills | job.nice_to_have_skills
                all_user_skills = user_skills | preferred_stack
                matching_skills = all_job_skills & all_user_skills
                matching_count = len(matching_skills)

                # Require at least 2 matching skills
                min_matching_skills = 2

                # If no must-haves, use fallback threshold
                if not has_must_haves:
                    if score < min_threshold:
                        reasons.append(f"Score below threshold ({score:.0f}% < {min_threshold}%)")
                        is_matched = False

                    if matching_count < min_matching_skills:
                        reasons.append(f"Not enough matching skills ({matching_count} < {min_matching_skills})")
                        is_matched = False
                else:
                    # Normal case: check score, coverage, AND matching skill count
                    if score < min_threshold:
                        reasons.append(f"Score below threshold ({score:.0f}% < {min_threshold}%)")
                        is_matched = False

                    if must_have_coverage < 0.6:
                        reasons.append(f"Must-have coverage too low ({must_have_coverage:.0%} < 60%)")
                        is_matched = False

                    if matching_count < min_matching_skills:
                        reasons.append(f"Not enough matching skills ({matching_count} < {min_matching_skills})")
                        is_matched = False

            # Create job dict
            job_id = hashlib.sha256(job.apply_url.encode()).hexdigest()[:16]

            # Calculate skill matches
            all_job_skills = job.must_have_skills | job.nice_to_have_skills
            all_user_skills = user_skills | preferred_stack
            matched_skills = sorted(all_job_skills & all_user_skills)
            missing_skills = sorted(all_job_skills - all_user_skills)

            must_have_matched = sorted(scored.matching_skills) if is_matched and score is not None else []
            must_have_missing = sorted(scored.missing_must_haves) if is_matched and score is not None else []

            # If no must-haves, show stack as must-have
            if not has_must_haves:
                must_have_matched = matched_skills
                must_have_missing = missing_skills

            job_dict = {
                "id": job_id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "posted_at": job.posted_date,  # Can be null if date missing/invalid
                "url": job.apply_url,
                "source": job.source,
                "description": job.description[:1000] + "..." if len(job.description) > 1000 else job.description,
            }

            if is_matched and score is not None:
                # Add scoring details for matched jobs
                seniority_expl = "Strong match" if scored.seniority_alignment >= 0.9 else \
                                "Good match" if scored.seniority_alignment >= 0.7 else \
                                "Possible mismatch" if scored.seniority_alignment >= 0.5 else "Poor match"

                score_breakdown = {
                    "must_have_coverage": must_have_coverage if has_must_haves else None,
                    "stack_overlap": round(scored.stack_overlap, 2),
                    "seniority_alignment": round(scored.seniority_alignment, 2)
                }

                job_dict.update({
                    "score_total": round(score, 1),
                    "breakdown": score_breakdown,
                    "must_have": {
                        "matched": must_have_matched,
                        "missing": must_have_missing
                    },
                    "stack": {
                        "matched": matched_skills,
                        "missing": missing_skills
                    },
                    "seniority": {
                        "expected": job.seniority_level,
                        "found": user_seniority,
                        "explanation": seniority_expl
                    }
                })

                # Generate match explanation (optional but preferred)
                try:
                    user_profile_for_match = {
                        "skills": list(user_skills),
                        "seniority": user_seniority
                    }
                    match_explanation = generate_match_explanation(
                        job_requirements, user_profile_for_match, score_breakdown
                    )
                    job_dict["match_explanation"] = match_explanation
                except Exception as e:
                    logger.debug(f"Failed to generate match explanation: {e}")

                matched_jobs.append(job_dict)
            else:
                # Add minimal details for filtered jobs
                job_dict.update({
                    "score_total": round(score, 1) if score is not None else None,
                    "reasons": reasons
                })
                filtered_jobs.append(job_dict)

                # Track filter reasons
                for reason in reasons:
                    filter_reason_counts[reason] = filter_reason_counts.get(reason, 0) + 1

        # Sort matched jobs by score
        matched_jobs.sort(key=lambda j: j.get("score_total", 0), reverse=True)

        # Build stats
        top_filter_reasons = [
            {"reason": reason, "count": count}
            for reason, count in sorted(filter_reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        stats = {
            "fetched": len(all_jobs),
            "parsed": len(parsed_jobs),
            "scored": len(matched_jobs) + len(filtered_jobs),
            "matched": len(matched_jobs),
            "filtered": len(filtered_jobs),
            "top_filter_reasons": top_filter_reasons
        }

        # Prepare response
        response = {
            "matched_jobs": matched_jobs,
            "filtered_jobs": filtered_jobs,
            "stats": stats
        }

        # Track metrics
        import hashlib
        run_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        metrics_tracker.record_search(
            run_id=run_id,
            matched_count=len(matched_jobs),
            filtered_count=len(filtered_jobs)
        )

        # Optional: Send email digest
        email_status = None
        if send_digest and matched_jobs:
            try:
                # Create temp config for email
                email_config = JobScoutConfig(
                    resume_path="",
                    email=EmailConfig(
                        enabled=True,
                        to_address=to_email,
                        smtp_host=os.getenv("SMTP_HOST"),
                        smtp_port=int(os.getenv("SMTP_PORT", "587")),
                        smtp_username=os.getenv("SMTP_USER"),
                        smtp_password=os.getenv("SMTP_PASS"),
                        smtp_from=os.getenv("SMTP_FROM", f"JobScout <{to_email}>")
                    ),
                    schedule=ScheduleConfig(enabled=False),
                    job_preferences=JobPreferences(location_preference=location_pref),
                    outbox_dir="",
                    log_level="INFO",
                    min_score_threshold=min_threshold
                )

                # Build digest HTML
                digest_html = f"""
                <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; }}
                        .job {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                        .score {{ font-weight: bold; color: {"green" if j["score_total"] >= 75 else "orange"}; }}
                        .skills {{ font-size: 0.9em; color: #666; }}
                    </style>
                </head>
                <body>
                    <h1>JobScout Digest: {len(matched_jobs)} Matching Jobs</h1>
                    <p>Based on your profile with {len(user_skills)} skills</p>
                """

                for job in matched_jobs[:10]:  # Top 10
                    digest_html += f"""
                    <div class="job">
                        <h3>{job['title']} at {job['company']}</h3>
                        <p><strong>Location:</strong> {job['location']}</p>
                        <p><strong>Score:</strong> <span class="score">{job['score_total']}%</span></p>
                        <p><strong>Matched Skills:</strong> {', '.join(job['must_have']['matched'][:5])}</p>
                        <p><a href="{job['url']}">Apply Now</a></p>
                    </div>
                    """

                digest_html += """
                </body>
                </html>
                """

                # Send email
                from jobscout.emailer import EmailDelivery
                emailer = EmailDelivery(email_config)

                # Build scored jobs for emailer
                from backend.adapter import _build_scored_jobs_from_json
                scored_for_email = _build_scored_jobs_from_json(matched_jobs)

                success = emailer.send_digest(scored_for_email[:10])

                if success:
                    digest_id = datetime.now().strftime('%Y%m%d_%H%M%S')
                    # Track metrics
                    metrics_tracker.record_email_sent(
                        digest_id=digest_id,
                        job_count=len(matched_jobs),
                        mode="smtp"
                    )
                    email_status = {
                        "sent": True,
                        "digest_id": digest_id
                    }
                else:
                    email_status = {
                        "sent": False,
                        "error": "Email delivery failed (SMTP may not be configured)"
                    }
            except Exception as e:
                logger.warning(f"Email digest failed: {e}")
                email_status = {
                    "sent": False,
                    "error": str(e)
                }

        if email_status:
            response["email"] = email_status

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/api/runs/{run_id}")
async def get_run_status(run_id: str):
    """
    Get run status for frontend polling.

    Returns the current status, phase, and progress message for a run.
    """
    try:
        # Check if run exists
        run_dir = storage.get_run_dir(run_id)

        if not run_dir:
            raise HTTPException(
                status_code=404,
                detail=f"Run {run_id} not found"
            )

        # Load run metadata
        metadata = storage.load_run_meta(run_id)

        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Run metadata not found for {run_id}"
            )

        return metadata

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load run status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load run status: {str(e)}"
        )


@app.get("/api/jobs", response_model=JobsListResponse)
async def get_jobs(
    run_id: str = Query(..., description="Run ID from /api/search"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page")
):
    """Get matching jobs from a run."""
    try:
        jobs = storage.load_run_jobs(run_id)

        if jobs is None:
            raise HTTPException(
                status_code=404,
                detail=f"Run {run_id} not found"
            )

        total = len(jobs)
        total_pages = math.ceil(total / page_size) if total else 0
        start = (page - 1) * page_size
        end = start + page_size
        paged = jobs[start:end]

        return JobsListResponse(
            jobs=paged,
            pagination=Pagination(
                page=page,
                page_size=page_size,
                total=total,
                total_pages=total_pages
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load jobs: {str(e)}"
        )


@app.get("/api/jobs/{job_id}", response_model=JobDetails)
async def get_job_details(job_id: str, run_id: str = Query(..., description="Run ID from /api/search")):
    """Get details for a specific job."""
    try:
        jobs = storage.load_run_jobs(run_id)

        if jobs is None:
            raise HTTPException(
                status_code=404,
                detail=f"Run {run_id} not found"
            )

        # Find job by ID
        job = next((j for j in jobs if j["id"] == job_id), None)

        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found in run {run_id}"
            )

        return JobDetails(**job)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load job details: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load job details: {str(e)}"
        )


@app.get("/api/filtered-jobs", response_model=FilteredJobsListResponse)
async def get_filtered_jobs(
    run_id: str = Query(..., description="Run ID from /api/search"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page")
):
    """Get filtered-out jobs from a run."""
    try:
        filtered_jobs = storage.load_run_filtered_jobs(run_id)

        if filtered_jobs is None:
            raise HTTPException(
                status_code=404,
                detail=f"Run {run_id} not found"
            )

        total = len(filtered_jobs)
        total_pages = math.ceil(total / page_size) if total else 0
        start = (page - 1) * page_size
        end = start + page_size
        paged = filtered_jobs[start:end]

        return FilteredJobsListResponse(
            filtered_jobs=paged,
            pagination=Pagination(
                page=page,
                page_size=page_size,
                total=total,
                total_pages=total_pages
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load filtered jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load filtered jobs: {str(e)}"
        )


@app.post("/api/send-digest", response_model=SendDigestResponse)
async def send_digest(request: SendDigestRequest):
    """
    Generate and send email digest.

    Uses the most recent run if run_id is not provided.
    """
    try:
        # Determine which run to use
        run_id = request.run_id or storage.get_latest_run_id()

        if not run_id:
            raise HTTPException(
                status_code=404,
                detail="No runs found. Please run a search first."
            )

        # Load jobs
        jobs = storage.load_run_jobs(run_id)

        if jobs is None:
            raise HTTPException(
                status_code=404,
                detail=f"Run {run_id} not found"
            )

        if not jobs:
            from datetime import datetime
            digest_id = f"empty_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            storage.save_digest(
                digest_id=digest_id,
                html=f"<html><body><h1>No matching jobs</h1><p>Run {run_id}</p></body></html>",
                subject="JobScout: 0 apply-ready matches",
                meta={
                    "created_at": datetime.now().isoformat(),
                    "mode": "noop",
                    "run_id": run_id,
                    "job_count": 0
                }
            )
            return SendDigestResponse(
                digest_id=digest_id,
                mode="noop"
            )

        # Load config (from env vars or config.yaml)
        config = get_or_create_config()

        # Check if outbox mode
        outbox_mode = os.getenv("OUTBOX_MODE", "false").lower() == "true"

        # Send digest without reparsing resume
        result = send_email_digest_from_jobs(jobs, config, outbox_mode=outbox_mode)

        # Save digest to storage
        from datetime import datetime
        digest_id = result["digest_id"]

        # Generate HTML digest (simplified - reuse from adapter)
        # For now, just save metadata
        storage.save_digest(
            digest_id=digest_id,
            html=f"<html><body><h1>Digest {digest_id}</h1><p>{len(jobs)} jobs</p></body></html>",
            subject=f"JobScout: {len(jobs)} apply-ready matches",
            meta={
                "created_at": datetime.now().isoformat(),
                "mode": result["mode"],
                "run_id": run_id,
                "job_count": len(jobs)
            }
        )

        # Track metrics
        if result["mode"] in ("smtp", "outbox"):
            metrics_tracker.record_email_sent(
                digest_id=digest_id,
                job_count=len(jobs),
                mode=result["mode"]
            )

        return SendDigestResponse(
            digest_id=digest_id,
            mode=result["mode"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send digest: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send digest: {str(e)}"
        )


@app.post("/api/send-test-email", response_model=SendTestEmailResponse)
async def send_test_email():
    """Send a test email to verify SMTP configuration."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # Load config (from env vars or config.yaml)
        config = get_or_create_config()

        # Check SMTP configuration
        if not all([
            config.email.smtp_host,
            config.email.smtp_username,
            config.email.smtp_password,
            config.email.smtp_from,
            config.email.to_address
        ]):
            raise HTTPException(
                status_code=400,
                detail="SMTP not configured. Please configure SMTP settings first."
            )

        # Send test email
        msg = MIMEMultipart()
        msg['Subject'] = "JobScout Test Email"
        msg['From'] = config.email.smtp_from
        msg['To'] = config.email.to_address

        body = """
        <html>
        <body>
            <h2>JobScout Test Email</h2>
            <p>This is a test email from JobScout.</p>
            <p>Your SMTP configuration is working correctly!</p>
        </body>
        </html>
        """

        msg.attach(MIMEText(body, 'html'))

        with smtplib.SMTP(config.email.smtp_host, config.email.smtp_port) as server:
            server.starttls()
            server.login(config.email.smtp_username, config.email.smtp_password)
            server.send_message(msg)

        logger.info(f"Test email sent to {config.email.to_address}")

        return SendTestEmailResponse(ok=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send test email: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send test email: {str(e)}"
        )


@app.get("/api/digests", response_model=DigestsListResponse)
async def list_digests():
    """List all email digests."""
    try:
        digests = storage.list_digests()
        return DigestsListResponse(digests=digests)

    except Exception as e:
        logger.error(f"Failed to list digests: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list digests: {str(e)}"
        )


@app.get("/api/digests/{digest_id}", response_model=DigestResponse)
async def get_digest(digest_id: str):
    """Get a specific digest."""
    try:
        digest = storage.load_digest(digest_id)

        if not digest:
            raise HTTPException(
                status_code=404,
                detail=f"Digest {digest_id} not found"
            )

        return DigestResponse(**digest)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load digest: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load digest: {str(e)}"
        )


# LLM Model Management Endpoints

@app.get("/api/llm/models", response_model=AvailableModelsResponse)
async def list_available_models():
    """Get all available LLM models grouped by provider."""
    try:
        from jobscout.llm_providers import get_available_models_for_frontend, DEFAULT_MODEL

        models = get_available_models_for_frontend()

        return AvailableModelsResponse(
            models=models,
            default_model=DEFAULT_MODEL
        )

    except Exception as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@app.post("/api/llm/select-model", response_model=UpdateLLMModelResponse)
async def select_llm_model(request: UpdateLLMModelRequest):
    """
    Select and configure an LLM model for job parsing.

    If api_key is provided, it will be stored for future use.
    Otherwise, uses the existing stored key for that provider.
    """
    try:
        from jobscout.llm_providers import get_model

        # Validate model ID
        model = get_model(request.model_id)
        if not model:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model_id}"
            )

        # Store the selection
        if request.api_key:
            # Store the API key securely (in practice, you'd encrypt this)
            storage.save_llm_config({
                "model_id": request.model_id,
                "api_key": request.api_key,
                "provider": model.provider.value
            })
        else:
            # Just update the model selection
            storage.save_llm_config({
                "model_id": request.model_id,
                "provider": model.provider.value
            })

        return UpdateLLMModelResponse(
            success=True,
            model_id=request.model_id,
            model_name=model.name
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to select model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to select model: {str(e)}"
        )


@app.get("/api/llm/current-model")
async def get_current_model():
    """Get the currently selected LLM model."""
    try:
        llm_config = storage.load_llm_config()

        if not llm_config:
            # Return default
            from jobscout.llm_providers import DEFAULT_MODEL, get_model
            model = get_model(DEFAULT_MODEL)

            return {
                "model_id": DEFAULT_MODEL,
                "model_name": model.name if model else "Unknown",
                "provider": model.provider.value if model else "unknown",
                "configured": False
            }

        from jobscout.llm_providers import get_model
        model = get_model(llm_config.get("model_id", DEFAULT_MODEL))

        return {
            "model_id": llm_config.get("model_id"),
            "model_name": model.name if model else "Unknown",
            "provider": llm_config.get("provider", "unknown"),
            "configured": True
        }

    except Exception as e:
        logger.error(f"Failed to get current model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current model: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
