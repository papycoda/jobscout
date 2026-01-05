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
        if base_dir is None:
            # Use environment variable or default to local directory
            base_dir = os.getenv("JOBSCOUT_DATA_DIR", os.getenv("DATA_DIR", "./data"))

        self.base_dir = Path(base_dir)
        self._base_dir_initialized = False
        self._use_memory_runs = False
        self._memory_runs: Dict[str, Dict[str, object]] = {}
        self._memory_run_timestamps: Dict[str, float] = {}

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
        meta_path = self.runs_dir / run_id / "run_meta.json"

        if not meta_path.exists():
            return self._memory_runs.get(run_id, {}).get("meta")

        return json.loads(meta_path.read_text())

    def load_run_jobs(self, run_id: str) -> Optional[List[Dict]]:
        """Load jobs from a run."""
        jobs_path = self.runs_dir / run_id / "jobs.json"

        if not jobs_path.exists():
            return self._memory_runs.get(run_id, {}).get("jobs")

        return json.loads(jobs_path.read_text())

    def load_run_filtered_jobs(self, run_id: str) -> Optional[List[Dict]]:
        """Load filtered jobs from a run."""
        filtered_path = self.runs_dir / run_id / "filtered_jobs.json"

        if not filtered_path.exists():
            return self._memory_runs.get(run_id, {}).get("filtered_jobs")

        return json.loads(filtered_path.read_text())

    def get_run_dir(self, run_id: str) -> Optional[Path]:
        """Get run directory path."""
        run_dir = self.runs_dir / run_id

        if not run_dir.exists():
            if run_id in self._memory_runs:
                return run_dir
            return None

        return run_dir

    def save_digest(self, digest_id: str, html: str, subject: str, meta: Dict) -> None:
        """Save email digest."""
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

        if latest_disk_time is None and latest_memory_time is None:
            return None
        if latest_disk_time is None:
            return latest_memory_id
        if latest_memory_time is None:
            return latest_disk_id
        if latest_memory_time >= latest_disk_time:
            return latest_memory_id
        return latest_disk_id
