"""FastAPI backend for JobScout."""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path to import jobscout
sys.path.insert(0, str(Path(__file__).parent.parent))

from jobscout.config import JobScoutConfig
from backend.storage import Storage
from backend.adapter import JobScoutAdapter
from backend.models import *


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


# CORS configuration
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,https://jobscoutpro.netlify.app").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
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
    """Debug endpoint to check disk configuration."""
    import os
    from pathlib import Path

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


@app.post("/api/upload-resume", response_model=ResumeUploadResponse)
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a resume file.

    Saves the resume and extracts skills/metadata.
    """
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".docx", ".txt"}
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )

        # Read file content
        content = await file.read()

        # Save to storage
        resume_path = storage.save_resume(file.filename, content)

        # Parse resume
        from jobscout.resume_parser import ResumeParser
        parser = ResumeParser()
        parsed = parser.parse(resume_path)

        # Extract metadata
        metadata = {
            "seniority": parsed.seniority,
            "years_experience": parsed.years_experience,
            "role_keywords": parsed.role_keywords,
            "skills_count": len(parsed.skills)
        }

        return ResumeUploadResponse(
            resume_path=resume_path,
            skills=sorted(parsed.skills),
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload resume: {str(e)}"
        )


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration."""
    try:
        if os.getenv("USE_ENV_CONFIG", "false").lower() == "true":
            from dataclasses import asdict
            import yaml

            config = create_config_from_env()
            config_data = asdict(config)
            email_data = config_data.get("email", {})
            if email_data.get("to_address"):
                email_data["to_address"] = "******"
            if email_data.get("smtp_username"):
                email_data["smtp_username"] = "******"
            if email_data.get("smtp_password"):
                email_data["smtp_password"] = "******"
            if email_data.get("smtp_from"):
                email_data["smtp_from"] = "******"
            if config_data.get("openai_api_key"):
                config_data["openai_api_key"] = "******"

            config_yaml = yaml.safe_dump(config_data, sort_keys=False)
            return ConfigResponse(config_yaml=config_yaml)

        config_yaml = storage.load_config()

        if not config_yaml:
            # Return empty config if none exists
            return ConfigResponse(config_yaml="")

        return ConfigResponse(config_yaml=config_yaml)

    except Exception as e:
        logger.error(f"Failed to load config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load configuration: {str(e)}"
        )


@app.put("/api/config")
async def update_config(request: ConfigUpdateRequest):
    """Update configuration."""
    try:
        # Validate YAML syntax
        config = load_config_from_yaml(request.config_yaml)

        # Save to storage
        storage.save_config(request.config_yaml)

        return {"ok": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save configuration: {str(e)}"
        )


@app.post("/api/search", response_model=SearchResponse)
async def run_search():
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

        if not resume_path:
            raise HTTPException(
                status_code=400,
                detail="No resume uploaded. Please upload a resume first."
            )

        # Load config (from env vars or config.yaml)
        config = get_or_create_config()

        # Override resume path to use latest uploaded
        config.resume_path = resume_path

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
        adapter = JobScoutAdapter(config)
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
async def get_jobs(run_id: str = Query(..., description="Run ID from /api/search")):
    """Get matching jobs from a run."""
    try:
        jobs = storage.load_run_jobs(run_id)

        if jobs is None:
            raise HTTPException(
                status_code=404,
                detail=f"Run {run_id} not found"
            )

        return JobsListResponse(jobs=jobs)

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
async def get_filtered_jobs(run_id: str = Query(..., description="Run ID from /api/search")):
    """Get filtered-out jobs from a run."""
    try:
        filtered_jobs = storage.load_run_filtered_jobs(run_id)

        if filtered_jobs is None:
            raise HTTPException(
                status_code=404,
                detail=f"Run {run_id} not found"
            )

        return FilteredJobsListResponse(filtered_jobs=filtered_jobs)

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

        # Use adapter to send email
        adapter = JobScoutAdapter(config)
        result = adapter.send_email_digest(jobs, outbox_mode=outbox_mode)

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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
