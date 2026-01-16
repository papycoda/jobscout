"""Simple file-based metrics tracker for MVP - persists across restarts."""

import json
import time
import logging
import os
from datetime import datetime
from typing import Optional
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

METRICS_FILE = Path(os.getenv("METRICS_FILE", "./data/metrics.json"))


class MetricsTracker:
    """Track key metrics for PMF validation with disk persistence."""

    def __init__(self, max_recent: int = 100):
        self._searches: deque = deque(maxlen=max_recent)
        self._resume_uploads: deque = deque(maxlen=max_recent)
        self._emails_sent: deque = deque(maxlen=max_recent)
        self._dirty = False
        self._last_save = 0
        self._save_interval = 30  # seconds between saves

        # Load existing metrics on startup
        self._load()

    def _load(self):
        """Load metrics from disk."""
        try:
            if METRICS_FILE.exists():
                data = json.loads(METRICS_FILE.read_text())

                # Restore deques from saved data
                for item in data.get("searches", []):
                    self._searches.append(item)
                for item in data.get("resume_uploads", []):
                    self._resume_uploads.append(item)
                for item in data.get("emails_sent", []):
                    self._emails_sent.append(item)

                logger.info(f"Loaded {len(self._searches)} searches, {len(self._resume_uploads)} uploads, {len(self._emails_sent)} emails from disk")
        except Exception as e:
            logger.warning(f"Failed to load metrics from disk: {e}")

    def _save(self, force: bool = False):
        """Save metrics to disk (throttled)."""
        now = time.time()
        if not force and now - self._last_save < self._save_interval:
            return

        try:
            # Ensure directory exists
            METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "searches": list(self._searches),
                "resume_uploads": list(self._resume_uploads),
                "emails_sent": list(self._emails_sent),
                "saved_at": datetime.now().isoformat()
            }

            METRICS_FILE.write_text(json.dumps(data, indent=2))
            self._last_save = now
            self._dirty = False
            logger.debug("Metrics saved to disk")
        except Exception as e:
            logger.warning(f"Failed to save metrics to disk: {e}")

    def record_search(self, run_id: str, matched_count: int, filtered_count: int, source: str = "api"):
        """Record a job search."""
        self._searches.append({
            "run_id": run_id,
            "matched": matched_count,
            "filtered": filtered_count,
            "total": matched_count + filtered_count,
            "source": source,
            "timestamp": datetime.now().isoformat()
        })
        self._dirty = True
        self._save()
        logger.info(f"Metrics: search recorded - {matched_count} matched, {filtered_count} filtered")

    def record_resume_upload(self, skills_count: int, seniority: str):
        """Record a resume upload."""
        self._resume_uploads.append({
            "skills_count": skills_count,
            "seniority": seniority,
            "timestamp": datetime.now().isoformat()
        })
        self._dirty = True
        self._save()
        logger.info(f"Metrics: resume upload recorded - {skills_count} skills, {seniority}")

    def record_email_sent(self, digest_id: str, job_count: int, mode: str):
        """Record an email digest sent."""
        self._emails_sent.append({
            "digest_id": digest_id,
            "job_count": job_count,
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        })
        self._dirty = True
        self._save()
        logger.info(f"Metrics: email sent - {job_count} jobs via {mode}")

    def get_metrics(self) -> dict:
        """Get all metrics."""
        # Calculate averages
        total_searches = len(self._searches)
        avg_jobs = 0
        avg_match_rate = 0

        if self._searches:
            total_jobs = sum(s["total"] for s in self._searches)
            avg_jobs = total_jobs / total_searches

            total_matched = sum(s["matched"] for s in self._searches)
            avg_match_rate = (total_matched / total_jobs * 100) if total_jobs > 0 else 0

        return {
            "total_searches": total_searches,
            "total_resume_uploads": len(self._resume_uploads),
            "total_emails_sent": len(self._emails_sent),
            "recent_searches": list(self._searches)[-20:],  # Last 20
            "avg_jobs_per_search": round(avg_jobs, 1),
            "avg_match_rate": round(avg_match_rate, 1)
        }

    def reset(self):
        """Reset all metrics (for testing/admin)."""
        self._searches.clear()
        self._resume_uploads.clear()
        self._emails_sent.clear()
        self._save(force=True)
        logger.info("Metrics reset")


# Global instance
metrics_tracker = MetricsTracker()
