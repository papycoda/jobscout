"""Configuration management for JobScout."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path
import yaml


@dataclass
class ScheduleConfig:
    """Scheduling configuration."""
    enabled: bool = True
    frequency: Literal["daily", "weekdays"] = "daily"
    time: str = "09:00"  # HH:MM format
    timezone: str = "America/New_York"


@dataclass
class EmailConfig:
    """Email delivery configuration."""
    enabled: bool = True
    to_address: str = ""
    # SMTP settings (optional - will write to outbox if not provided)
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from: Optional[str] = None


@dataclass
class JobPreferences:
    """User job preferences."""
    preferred_tech_stack: List[str] = field(default_factory=list)
    location_preference: Literal["remote", "hybrid", "onsite", "any"] = "any"
    job_boards: List[str] = field(default_factory=list)  # Empty = use all defaults
    greenhouse_boards: List[str] = field(default_factory=list)
    lever_companies: List[str] = field(default_factory=list)
    max_job_age_days: int = 7


@dataclass
class JobScoutConfig:
    """Main configuration for JobScout."""
    # Required
    resume_path: str = ""

    # User preferences
    email: EmailConfig = field(default_factory=EmailConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    job_preferences: JobPreferences = field(default_factory=JobPreferences)

    # Optional API keys (for future enhancements)
    openai_api_key: Optional[str] = None

    # Paths
    outbox_dir: str = "./outbox"
    log_level: str = "INFO"

    # Scoring thresholds
    min_score_threshold: float = 65.0  # 65% for "apply-ready"
    fallback_min_score: float = 75.0  # When must-have list is empty

    @classmethod
    def from_yaml(cls, path: str) -> "JobScoutConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Convert nested dicts to config objects
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

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.resume_path:
            errors.append("resume_path is required")

        if not Path(self.resume_path).exists():
            errors.append(f"Resume file not found: {self.resume_path}")

        if self.email.enabled and not self.email.to_address:
            errors.append("email.to_address is required when email is enabled")

        # Validate time format
        try:
            hours, minutes = self.schedule.time.split(":")
            if not (0 <= int(hours) <= 23 and 0 <= int(minutes) <= 59):
                errors.append("schedule.time must be in HH:MM format (00-23:00-59)")
        except (ValueError, AttributeError):
            errors.append("schedule.time must be in HH:MM format")

        # Validate max job age
        if self.job_preferences.max_job_age_days < 1:
            errors.append("job_preferences.max_job_age_days must be at least 1")

        # Validate scoring thresholds
        if not (0 <= self.min_score_threshold <= 100):
            errors.append("min_score_threshold must be between 0 and 100")

        if not (0 <= self.fallback_min_score <= 100):
            errors.append("fallback_min_score must be between 0 and 100")

        return errors


def load_config(config_path: str) -> JobScoutConfig:
    """Load and validate configuration from file."""
    config = JobScoutConfig.from_yaml(config_path)
    errors = config.validate()

    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return config
