"""Storage layer for persisting data on Render."""

import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import yaml


class Storage:
    """Handle persistent storage on Render."""

    def __init__(self, base_dir: str = None):
        """Initialize storage with base directory."""
        if base_dir is None:
            # Use environment variable or default to local directory
            base_dir = os.getenv("JOBSCOUT_DATA_DIR", "./data")

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.resumes_dir = self.base_dir / "resumes"
        self.runs_dir = self.base_dir / "runs"
        self.digests_dir = self.base_dir / "digests"
        self.outbox_dir = self.base_dir / "outbox"

        for dir_path in [self.resumes_dir, self.runs_dir, self.digests_dir, self.outbox_dir]:
            dir_path.mkdir(exist_ok=True)

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
        config_path = self.get_config_path()
        config_path.write_text(config_yaml)

    def save_resume(self, filename: str, content: bytes) -> str:
        """
        Save uploaded resume and return path.

        Returns: Path to saved resume
        """
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
        run_dir.mkdir(exist_ok=True)

        return run_id, run_dir

    def save_run_jobs(self, run_dir: Path, jobs: List[Dict]) -> None:
        """Save matching jobs to jobs.json."""
        jobs_path = run_dir / "jobs.json"
        jobs_path.write_text(json.dumps(jobs, indent=2, default=str))

    def save_run_filtered_jobs(self, run_dir: Path, filtered_jobs: List[Dict]) -> None:
        """Save filtered jobs to filtered_jobs.json."""
        filtered_path = run_dir / "filtered_jobs.json"
        filtered_path.write_text(json.dumps(filtered_jobs, indent=2, default=str))

    def save_run_meta(self, run_dir: Path, meta: Dict) -> None:
        """Save run metadata to run_meta.json."""
        meta_path = run_dir / "run_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    def load_run_jobs(self, run_id: str) -> Optional[List[Dict]]:
        """Load jobs from a run."""
        jobs_path = self.runs_dir / run_id / "jobs.json"

        if not jobs_path.exists():
            return None

        return json.loads(jobs_path.read_text())

    def load_run_filtered_jobs(self, run_id: str) -> Optional[List[Dict]]:
        """Load filtered jobs from a run."""
        filtered_path = self.runs_dir / run_id / "filtered_jobs.json"

        if not filtered_path.exists():
            return None

        return json.loads(filtered_path.read_text())

    def get_run_dir(self, run_id: str) -> Optional[Path]:
        """Get run directory path."""
        run_dir = self.runs_dir / run_id

        if not run_dir.exists():
            return None

        return run_dir

    def save_digest(self, digest_id: str, html: str, subject: str, meta: Dict) -> None:
        """Save email digest."""
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
        run_dirs = [d for d in self.runs_dir.iterdir() if d.is_dir()]

        if not run_dirs:
            return None

        # Sort by modification time
        latest_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
        return latest_dir.name
