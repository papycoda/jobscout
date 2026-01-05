"""Job scheduling with timezone support."""

import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import pytz


logger = logging.getLogger(__name__)


class JobScoutScheduler:
    """Schedule and run job searches."""

    def __init__(self, config, job_search_function):
        """
        Initialize scheduler.

        Args:
            config: JobScoutConfig instance
            job_search_function: Function to call for each scheduled run
        """
        self.config = config
        self.job_search_function = job_search_function
        self.scheduler = BlockingScheduler()

    def start(self):
        """Start the scheduler (blocking)."""
        if not self.config.schedule.enabled:
            logger.info("Scheduling is disabled. Run manually instead.")
            return

        # Add scheduled job
        trigger = self._build_trigger()
        self.scheduler.add_job(
            self._run_job_search,
            trigger=trigger,
            name='jobscout_search',
            id='jobscout_search'
        )

        logger.info(f"Scheduler started. Next run: {self.scheduler.get_job('jobscout_search').next_run_time}")

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped")

    def run_once(self):
        """Run job search once immediately (for testing or manual runs)."""
        logger.info("Running manual job search...")
        self._run_job_search()

    def _build_trigger(self):
        """Build APScheduler trigger based on config."""
        schedule = self.config.schedule

        # Parse time
        hour, minute = map(int, schedule.time.split(':'))

        # Get timezone
        tz = pytz.timezone(schedule.timezone)

        # Build trigger based on frequency
        if schedule.frequency == "daily":
            return CronTrigger(
                hour=hour,
                minute=minute,
                timezone=tz
            )
        elif schedule.frequency == "weekdays":
            return CronTrigger(
                day_of_week='mon-fri',
                hour=hour,
                minute=minute,
                timezone=tz
            )
        else:
            raise ValueError(f"Unsupported frequency: {schedule.frequency}")

    def _run_job_search(self):
        """Execute job search function."""
        logger.info("="*60)
        logger.info(f"Starting scheduled job search at {datetime.now()}")
        logger.info("="*60)

        try:
            self.job_search_function()
        except Exception as e:
            logger.error(f"Job search failed: {e}", exc_info=True)

        logger.info("="*60)
        logger.info(f"Completed scheduled job search at {datetime.now()}")
        logger.info("="*60)
