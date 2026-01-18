"""Cross-run job deduplication cache."""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)


class JobDeduplicator:
    """
    Track seen jobs across runs to avoid processing duplicates.

    Uses a simple JSON file cache with timestamps.
    Jobs are tracked by URL hash and automatically cleaned up after max_age_days.
    """

    def __init__(self, cache_file: str = "./data/seen_jobs.json", max_age_days: int = 7):
        """
        Initialize deduplicator.

        Args:
            cache_file: Path to JSON cache file
            max_age_days: Days to keep entries before cleanup
        """
        self.cache_file = Path(cache_file)
        self.max_age_days = max_age_days
        self._cache: dict = {}
        self._dirty = False

        # Ensure data directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing cache
        self._load()

    def _get_hash(self, url: str) -> str:
        """Generate consistent hash for URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def _load(self):
        """Load cache from disk."""
        if not self.cache_file.exists():
            logger.debug(f"No cache file found at {self.cache_file}, starting fresh")
            self._cache = {}
            return

        try:
            data = json.loads(self.cache_file.read_text())
            # Clean old entries on load
            self._cache = self._clean_old_entries(data)
            logger.debug(f"Loaded {len(self._cache)} seen jobs from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Starting fresh.")
            self._cache = {}

    def _save(self, force: bool = False):
        """Save cache to disk (idempotent writes only)."""
        if not force and not self._dirty:
            return

        try:
            self.cache_file.write_text(json.dumps(self._cache, indent=2))
            self._dirty = False
            logger.debug(f"Saved {len(self._cache)} entries to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _clean_old_entries(self, cache: dict) -> dict:
        """Remove entries older than max_age_days."""
        cutoff = datetime.now() - timedelta(days=self.max_age_days)
        cutoff_str = cutoff.isoformat()

        cleaned = {}
        for url_hash, entry in cache.items():
            if isinstance(entry, dict) and entry.get("last_seen"):
                try:
                    last_seen = datetime.fromisoformat(entry["last_seen"])
                    if last_seen >= cutoff:
                        cleaned[url_hash] = entry
                except (ValueError, TypeError):
                    # Invalid date, skip
                    continue
            else:
                # Old format or invalid, skip
                continue

        removed = len(cache) - len(cleaned)
        if removed > 0:
            logger.info(f"Cleaned {removed} old entries from cache (older than {self.max_age_days} days)")

        return cleaned

    def is_seen(self, apply_url: str) -> bool:
        """
        Check if job URL has been seen recently.

        Args:
            apply_url: The job's application URL

        Returns:
            True if job was seen within max_age_days
        """
        if not apply_url:
            return False  # Can't dedup without URL

        url_hash = self._get_hash(apply_url)

        if url_hash not in self._cache:
            return False

        # Check if entry is still valid (not too old)
        entry = self._cache[url_hash]
        if isinstance(entry, dict) and entry.get("last_seen"):
            try:
                last_seen = datetime.fromisoformat(entry["last_seen"])
                cutoff = datetime.now() - timedelta(days=self.max_age_days)
                if last_seen >= cutoff:
                    return True
            except (ValueError, TypeError):
                pass

        # Old entry, remove it
        del self._cache[url_hash]
        self._dirty = True
        return False

    def mark_seen(self, apply_url: str, title: str = "", source: str = ""):
        """
        Mark a job as seen.

        Args:
            apply_url: The job's application URL
            title: Job title (for debugging)
            source: Job source (for debugging)
        """
        if not apply_url:
            return

        url_hash = self._get_hash(apply_url)
        now = datetime.now().isoformat()

        # Update existing entry or create new one
        if url_hash in self._cache:
            self._cache[url_hash]["last_seen"] = now
            self._cache[url_hash]["seen_count"] = self._cache[url_hash].get("seen_count", 1) + 1
        else:
            self._cache[url_hash] = {
                "first_seen": now,
                "last_seen": now,
                "seen_count": 1,
                "title": title[:100],  # Truncate long titles
                "source": source
            }

        self._dirty = True

    def mark_seen_batch(self, jobs: list):
        """
        Mark multiple jobs as seen in one operation.

        Args:
            jobs: List of objects with apply_url, title, source attributes
        """
        for job in jobs:
            self.mark_seen(
                apply_url=getattr(job, "apply_url", ""),
                title=getattr(job, "title", ""),
                source=getattr(job, "source", "")
            )
        # Save after batch
        self._save(force=True)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = len(self._cache)
        sources = {}
        for entry in self._cache.values():
            if isinstance(entry, dict):
                src = entry.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1

        return {
            "total_seen": total,
            "sources": sources,
            "cache_file": str(self.cache_file),
            "max_age_days": self.max_age_days
        }

    def cleanup(self):
        """Force cleanup of old entries and save."""
        self._cache = self._clean_old_entries(self._cache)
        self._save(force=True)


# Global instance for backward compatibility
_global_deduplicator: Optional[JobDeduplicator] = None


def get_deduplicator(cache_file: str = "./data/seen_jobs.json", max_age_days: int = 7) -> JobDeduplicator:
    """Get or create global deduplicator instance."""
    global _global_deduplicator
    if _global_deduplicator is None:
        _global_deduplicator = JobDeduplicator(cache_file, max_age_days)
    return _global_deduplicator
