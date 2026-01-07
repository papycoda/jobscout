"""Storage layer for persisting data on Render."""

import os
import json
import uuid
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict


logger = logging.getLogger(__name__)


class Storage:
    """Handle persistent storage on Render."""

    def __init__(self, base_dir: str = None):
        """Initialize storage with base directory."""
        self.database_url = os.getenv("DATABASE_URL")
        if self.database_url and "sslmode=" not in self.database_url:
            separator = "&" if "?" in self.database_url else "?"
            self.database_url = f"{self.database_url}{separator}sslmode=require"

        self._db_enabled = bool(self.database_url)
        self._db_schema_initialized = False

        if self._db_enabled:
            try:
                import psycopg2  # noqa: F401
            except Exception:
                logger.warning("DATABASE_URL set but psycopg2 is unavailable; falling back to filesystem storage.")
                self._db_enabled = False

        if base_dir is None:
            # Use environment variable or default to local directory
            base_dir = os.getenv("JOBSCOUT_DATA_DIR", os.getenv("DATA_DIR", "./data"))

        self.base_dir = Path(base_dir)
        self._base_dir_initialized = False
        self._use_memory_runs = False
        self._memory_runs: Dict[str, Dict[str, object]] = {}
        self._memory_run_timestamps: Dict[str, float] = {}
        self._memory_resume_profile: Optional[Dict] = None

        # Create subdirectories (allow override via env vars)
        self.resumes_dir = Path(os.getenv("RESUMES_DIR", str(self.base_dir / "resumes")))
        self.runs_dir = Path(os.getenv("RUNS_DIR", str(self.base_dir / "runs")))
        self.digests_dir = Path(os.getenv("DIGESTS_DIR", str(self.base_dir / "digests")))
        self.outbox_dir = Path(os.getenv("OUTBOX_DIR", str(self.base_dir / "outbox")))

    def _ensure_directories(self):
        """Ensure all directories exist (lazy initialization)."""
        if not self._base_dir_initialized:
            try:
                self.base_dir.mkdir(parents=True, exist_ok=True)

                # Create subdirectories
                for dir_path in [self.resumes_dir, self.runs_dir, self.digests_dir, self.outbox_dir]:
                    dir_path.mkdir(parents=True, exist_ok=True)

                self._base_dir_initialized = True
            except PermissionError:
                logger.error(f"Permission denied creating {self.base_dir}. Render disk must be mounted first.")
                raise

    def _ensure_db_schema(self) -> bool:
        if not self._db_enabled:
            return False
        if self._db_schema_initialized:
            return True
        try:
            import psycopg2
            with psycopg2.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS jobscout_runs (
                            run_id TEXT PRIMARY KEY,
                            meta JSONB,
                            jobs JSONB,
                            filtered_jobs JSONB,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS jobscout_digests (
                            digest_id TEXT PRIMARY KEY,
                            html TEXT,
                            subject TEXT,
                            meta JSONB,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS jobscout_resume_profiles (
                            profile_id TEXT PRIMARY KEY,
                            profile JSONB,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        """
                    )
                conn.commit()
            self._db_schema_initialized = True
            return True
        except Exception as exc:
            logger.warning(f"Failed to initialize database schema: {exc}")
            return False

    def _db_connect(self):
        import psycopg2
        return psycopg2.connect(self.database_url)

    def _store_run_in_memory(self, run_id: str, key: str, value: object) -> None:
        """Store run data in memory as a fallback."""
        entry = self._memory_runs.setdefault(run_id, {})
        entry[key] = value
        self._memory_run_timestamps[run_id] = time.time()

    def get_config_path(self) -> Path:
        """Get path to config.yaml."""
        return self.base_dir / "config.yaml"

    def get_example_config_path(self) -> Path:
        """Get path to example config (in repo root)."""
        # This assumes we're running from the backend/ directory
        return Path(__file__).parent.parent / "config.example.yaml"

    def load_config(self) -> Optional[str]:
        """Load config YAML content."""
        config_path = self.get_config_path()

        if not config_path.exists():
            # Try to copy from example
            example_path = self.get_example_config_path()
            if example_path.exists():
                example_content = example_path.read_text()
                self.save_config(example_content)
                return example_content
            return None

        return config_path.read_text()

    def save_config(self, config_yaml: str) -> None:
        """Save config YAML content."""
        self._ensure_directories()
        config_path = self.get_config_path()
        config_path.write_text(config_yaml)

    def save_resume(self, filename: str, content: bytes) -> str:
        """
        Save uploaded resume and return path.

        Returns: Path to saved resume
        """
        self._ensure_directories()

        # Generate unique filename
        unique_id = uuid.uuid4().hex[:8]
        safe_filename = f"{unique_id}-{filename}"
        resume_path = self.resumes_dir / safe_filename

        # Save file
        resume_path.write_bytes(content)

        # Update "latest resume" pointer
        latest_path = self.resumes_dir / ".latest"
        latest_path.write_text(str(resume_path))

        return str(resume_path)

    def save_resume_profile(self, profile: Dict) -> None:
        """Save parsed resume profile for scheduled runs."""
        if self._db_enabled and self._ensure_db_schema():
            try:
                import psycopg2.extras
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO jobscout_resume_profiles (profile_id, profile, updated_at)
                            VALUES (%s, %s, NOW())
                            ON CONFLICT (profile_id)
                            DO UPDATE SET profile = EXCLUDED.profile, updated_at = NOW()
                            """,
                            ("latest", psycopg2.extras.Json(profile))
                        )
                    conn.commit()
                return
            except Exception as exc:
                logger.warning(f"Failed to write resume profile to database: {exc}")

        try:
            self._ensure_directories()
            profile_path = self.resumes_dir / "resume_profile.json"
            profile_path.write_text(json.dumps(profile, indent=2, default=str))
            return
        except (PermissionError, OSError):
            logger.warning("Failed to write resume profile to disk; using in-memory profile.", exc_info=True)
            self._memory_resume_profile = profile

    def load_latest_resume_profile(self) -> Optional[Dict]:
        """Load the most recent resume profile."""
        if self._db_enabled and self._ensure_db_schema():
            try:
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT profile::text FROM jobscout_resume_profiles WHERE profile_id = %s",
                            ("latest",)
                        )
                        row = cur.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
            except Exception as exc:
                logger.warning(f"Failed to load resume profile from database: {exc}")

        profile_path = self.resumes_dir / "resume_profile.json"
        if profile_path.exists():
            try:
                return json.loads(profile_path.read_text())
            except Exception:
                return None

        return self._memory_resume_profile

    def get_latest_resume_path(self) -> Optional[str]:
        """Get path to latest uploaded resume."""
        latest_path = self.resumes_dir / ".latest"

        if not latest_path.exists():
            return None

        return latest_path.read_text().strip()

    def create_run_dir(self) -> tuple[str, Path]:
        """
        Create a new run directory.

        Returns: (run_id, run_dir_path)
        """
        run_id = uuid.uuid4().hex
        run_dir = self.runs_dir / run_id

        if self._db_enabled:
            try:
                self._ensure_directories()
                run_dir.mkdir(exist_ok=True)
            except (PermissionError, OSError):
                pass
            return run_id, run_dir

        try:
            self._ensure_directories()
            run_dir.mkdir(exist_ok=True)
        except (PermissionError, OSError):
            if not self._use_memory_runs:
                logger.warning(
                    f"Run storage unavailable at {self.runs_dir}; using in-memory runs.",
                    exc_info=True
                )
            self._use_memory_runs = True

        return run_id, run_dir

    def save_run_jobs(self, run_dir: Path, jobs: List[Dict]) -> None:
        """Save matching jobs to jobs.json."""
        run_id = run_dir.name
        if self._db_enabled and self._ensure_db_schema():
            try:
                import psycopg2.extras
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO jobscout_runs (run_id, jobs, updated_at)
                            VALUES (%s, %s, NOW())
                            ON CONFLICT (run_id)
                            DO UPDATE SET jobs = EXCLUDED.jobs, updated_at = NOW()
                            """,
                            (run_id, psycopg2.extras.Json(jobs))
                        )
                    conn.commit()
                return
            except Exception as exc:
                logger.warning(f"Failed to write jobs to database for run {run_id}: {exc}")

        if self._use_memory_runs:
            self._store_run_in_memory(run_id, "jobs", jobs)
            return

        jobs_path = run_dir / "jobs.json"
        try:
            jobs_path.write_text(json.dumps(jobs, indent=2, default=str))
        except (PermissionError, OSError):
            logger.warning(f"Failed to write jobs for run {run_id}; using in-memory storage.", exc_info=True)
            self._use_memory_runs = True
            self._store_run_in_memory(run_id, "jobs", jobs)

    def save_run_filtered_jobs(self, run_dir: Path, filtered_jobs: List[Dict]) -> None:
        """Save filtered jobs to filtered_jobs.json."""
        run_id = run_dir.name
        if self._db_enabled and self._ensure_db_schema():
            try:
                import psycopg2.extras
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO jobscout_runs (run_id, filtered_jobs, updated_at)
                            VALUES (%s, %s, NOW())
                            ON CONFLICT (run_id)
                            DO UPDATE SET filtered_jobs = EXCLUDED.filtered_jobs, updated_at = NOW()
                            """,
                            (run_id, psycopg2.extras.Json(filtered_jobs))
                        )
                    conn.commit()
                return
            except Exception as exc:
                logger.warning(f"Failed to write filtered jobs to database for run {run_id}: {exc}")

        if self._use_memory_runs:
            self._store_run_in_memory(run_id, "filtered_jobs", filtered_jobs)
            return

        filtered_path = run_dir / "filtered_jobs.json"
        try:
            filtered_path.write_text(json.dumps(filtered_jobs, indent=2, default=str))
        except (PermissionError, OSError):
            logger.warning(
                f"Failed to write filtered jobs for run {run_id}; using in-memory storage.",
                exc_info=True
            )
            self._use_memory_runs = True
            self._store_run_in_memory(run_id, "filtered_jobs", filtered_jobs)

    def save_run_meta(self, run_dir: Path, meta: Dict) -> None:
        """Save run metadata to run_meta.json."""
        run_id = meta.get("run_id", run_dir.name)
        if self._db_enabled and self._ensure_db_schema():
            try:
                import psycopg2.extras
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO jobscout_runs (run_id, meta, updated_at)
                            VALUES (%s, %s, NOW())
                            ON CONFLICT (run_id)
                            DO UPDATE SET meta = EXCLUDED.meta, updated_at = NOW()
                            """,
                            (run_id, psycopg2.extras.Json(meta))
                        )
                    conn.commit()
                return
            except Exception as exc:
                logger.warning(f"Failed to write run metadata to database for {run_id}: {exc}")

        if self._use_memory_runs:
            self._store_run_in_memory(run_id, "meta", meta)
            return

        meta_path = run_dir / "run_meta.json"
        try:
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
        except (PermissionError, OSError):
            logger.warning(
                f"Failed to write run metadata for {run_id}; using in-memory storage.",
                exc_info=True
            )
            self._use_memory_runs = True
            self._store_run_in_memory(run_id, "meta", meta)

    def load_run_meta(self, run_id: str) -> Optional[Dict]:
        """Load run metadata by run ID."""
        if self._db_enabled and self._ensure_db_schema():
            try:
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT meta::text FROM jobscout_runs WHERE run_id = %s",
                            (run_id,)
                        )
                        row = cur.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
            except Exception as exc:
                logger.warning(f"Failed to load run metadata from database for {run_id}: {exc}")

        meta_path = self.runs_dir / run_id / "run_meta.json"

        if not meta_path.exists():
            return self._memory_runs.get(run_id, {}).get("meta")

        return json.loads(meta_path.read_text())

    def load_run_jobs(self, run_id: str) -> Optional[List[Dict]]:
        """Load jobs from a run."""
        if self._db_enabled and self._ensure_db_schema():
            try:
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT jobs::text FROM jobscout_runs WHERE run_id = %s",
                            (run_id,)
                        )
                        row = cur.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
            except Exception as exc:
                logger.warning(f"Failed to load jobs from database for run {run_id}: {exc}")

        jobs_path = self.runs_dir / run_id / "jobs.json"

        if not jobs_path.exists():
            return self._memory_runs.get(run_id, {}).get("jobs")

        return json.loads(jobs_path.read_text())

    def load_run_filtered_jobs(self, run_id: str) -> Optional[List[Dict]]:
        """Load filtered jobs from a run."""
        if self._db_enabled and self._ensure_db_schema():
            try:
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT filtered_jobs::text FROM jobscout_runs WHERE run_id = %s",
                            (run_id,)
                        )
                        row = cur.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
            except Exception as exc:
                logger.warning(f"Failed to load filtered jobs from database for run {run_id}: {exc}")

        filtered_path = self.runs_dir / run_id / "filtered_jobs.json"

        if not filtered_path.exists():
            return self._memory_runs.get(run_id, {}).get("filtered_jobs")

        return json.loads(filtered_path.read_text())

    def get_run_dir(self, run_id: str) -> Optional[Path]:
        """Get run directory path."""
        if self._db_enabled and self._ensure_db_schema():
            try:
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT 1 FROM jobscout_runs WHERE run_id = %s",
                            (run_id,)
                        )
                        row = cur.fetchone()
                if row:
                    return self.runs_dir / run_id
            except Exception as exc:
                logger.warning(f"Failed to check run {run_id} in database: {exc}")

        run_dir = self.runs_dir / run_id

        if not run_dir.exists():
            if run_id in self._memory_runs:
                return run_dir
            return None

        return run_dir

    def save_digest(self, digest_id: str, html: str, subject: str, meta: Dict) -> None:
        """Save email digest."""
        if self._db_enabled and self._ensure_db_schema():
            try:
                import psycopg2.extras
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO jobscout_digests (digest_id, html, subject, meta)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (digest_id)
                            DO UPDATE SET html = EXCLUDED.html, subject = EXCLUDED.subject, meta = EXCLUDED.meta
                            """,
                            (digest_id, html, subject, psycopg2.extras.Json(meta))
                        )
                    conn.commit()
                return
            except Exception as exc:
                logger.warning(f"Failed to write digest to database for {digest_id}: {exc}")

        self._ensure_directories()

        # Save HTML
        digest_path = self.digests_dir / f"{digest_id}.html"
        digest_path.write_text(html)

        # Save metadata
        meta_path = self.digests_dir / f"{digest_id}.json"
        meta["id"] = digest_id
        meta["subject"] = subject
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    def load_digest(self, digest_id: str) -> Optional[Dict]:
        """Load digest by ID."""
        if self._db_enabled and self._ensure_db_schema():
            try:
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT html, subject, meta::text FROM jobscout_digests WHERE digest_id = %s",
                            (digest_id,)
                        )
                        row = cur.fetchone()
                if row:
                    html, subject, meta_text = row
                    meta = json.loads(meta_text) if meta_text else {}
                    return {
                        "id": digest_id,
                        "html": html,
                        "subject": subject,
                        **meta
                    }
            except Exception as exc:
                logger.warning(f"Failed to load digest from database for {digest_id}: {exc}")

        html_path = self.digests_dir / f"{digest_id}.html"
        meta_path = self.digests_dir / f"{digest_id}.json"

        if not html_path.exists() or not meta_path.exists():
            return None

        html = html_path.read_text()
        meta = json.loads(meta_path.read_text())

        return {
            "id": digest_id,
            "html": html,
            **meta
        }

    def list_digests(self) -> List[Dict]:
        """List all digests."""
        digests = []

        if self._db_enabled and self._ensure_db_schema():
            try:
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT digest_id, meta::text
                            FROM jobscout_digests
                            ORDER BY created_at DESC
                            """
                        )
                        rows = cur.fetchall()
                for digest_id, meta_text in rows:
                    try:
                        meta = json.loads(meta_text) if meta_text else {}
                        digests.append({
                            "id": meta.get("id", digest_id),
                            "created_at": meta.get("created_at", ""),
                            "subject": meta.get("subject", ""),
                            "mode": meta.get("mode", "unknown")
                        })
                    except Exception:
                        continue
                return digests
            except Exception as exc:
                logger.warning(f"Failed to list digests from database: {exc}")

        for meta_path in self.digests_dir.glob("*.json"):
            try:
                meta = json.loads(meta_path.read_text())
                digests.append({
                    "id": meta.get("id", meta_path.stem),
                    "created_at": meta.get("created_at", ""),
                    "subject": meta.get("subject", ""),
                    "mode": meta.get("mode", "unknown")
                })
            except Exception:
                continue

        # Sort by created_at descending
        digests.sort(key=lambda d: d.get("created_at", ""), reverse=True)

        return digests

    def get_latest_run_id(self) -> Optional[str]:
        """Get the most recent run ID."""
        try:
            run_dirs = [d for d in self.runs_dir.iterdir() if d.is_dir()]
        except FileNotFoundError:
            run_dirs = []

        latest_db_id = None
        latest_db_time = None
        if self._db_enabled and self._ensure_db_schema():
            try:
                with self._db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT run_id, updated_at FROM jobscout_runs ORDER BY updated_at DESC LIMIT 1"
                        )
                        row = cur.fetchone()
                if row:
                    latest_db_id, latest_db_time = row
            except Exception as exc:
                logger.warning(f"Failed to load latest run from database: {exc}")

        latest_disk_id = None
        latest_disk_time = None
        if run_dirs:
            latest_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
            latest_disk_id = latest_dir.name
            latest_disk_time = latest_dir.stat().st_mtime

        latest_memory_id = None
        latest_memory_time = None
        if self._memory_run_timestamps:
            latest_memory_id = max(self._memory_run_timestamps, key=self._memory_run_timestamps.get)
            latest_memory_time = self._memory_run_timestamps[latest_memory_id]

        candidates = []
        if latest_db_time is not None:
            try:
                latest_db_timestamp = latest_db_time.timestamp()
            except AttributeError:
                latest_db_timestamp = float(latest_db_time)
            candidates.append((latest_db_timestamp, latest_db_id))
        if latest_disk_time is not None:
            candidates.append((latest_disk_time, latest_disk_id))
        if latest_memory_time is not None:
            candidates.append((latest_memory_time, latest_memory_id))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
