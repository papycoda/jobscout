"""Email delivery for agent-based JobScout."""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from .agent_models import JobEvaluation


logger = logging.getLogger(__name__)


class AgentEmailer:
    """Sends job match emails with the new format."""

    def __init__(
        self,
        smtp_config: Optional[dict] = None,
        outbox_dir: str = "./outbox",
    ):
        """
        Initialize emailer.

        Args:
            smtp_config: Dict with smtp_host, smtp_port, smtp_user, smtp_password, smtp_from, to_address
            outbox_dir: Directory to write emails when SMTP not configured
        """
        self.smtp_config = smtp_config
        self.outbox_dir = Path(outbox_dir)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)

    def send_digest(
        self,
        jobs: List[JobEvaluation],
        candidate_name: str = "",
        total_evaluated: int = 0,
    ) -> dict:
        """
        Send email digest with matched jobs.

        Args:
            jobs: List of job evaluations to email
            candidate_name: Optional candidate name for subject
            total_evaluated: Total jobs evaluated (for stats)

        Returns:
            Dict with sent=True/False, mode="smtp"/"outbox", digest_id
        """
        if not jobs:
            logger.info("No jobs to send")
            return self._write_empty_digest(total_evaluated)

        html = self._render_html(jobs, candidate_name, total_evaluated)
        subject = self._get_subject(jobs, candidate_name)

        # Try SMTP if configured
        if self.smtp_config and self.smtp_config.get("smtp_host"):
            return self._send_smtp(html, subject)
        else:
            return self._write_outbox(html, subject)

    def _render_html(
        self,
        jobs: List[JobEvaluation],
        candidate_name: str,
        total_evaluated: int,
    ) -> str:
        """Render HTML email."""
        date_str = datetime.now().strftime("%B %d, %Y")
        filtered_count = total_evaluated - len(jobs)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; background: #f5f7fa; }}
        .container {{ background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 24px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 24px; font-weight: 600; }}
        .header p {{ margin: 5px 0 0; opacity: 0.95; font-size: 14px; }}
        .content {{ padding: 0; }}
        .job {{ border-bottom: 1px solid #e8eaf0; }}
        .job:last-child {{ border-bottom: none; }}
        .job-header {{ padding: 16px 20px; display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }}
        .job-info {{ flex: 1; }}
        .job-title {{ font-size: 17px; font-weight: 600; color: #1a202c; margin: 0 0 4px; }}
        .job-company {{ font-size: 14px; color: #718096; }}
        .job-score {{ flex-shrink: 0; background: #667eea; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 14px; }}
        .job-score.high {{ background: #48bb78; }}
        .job-score.medium {{ background: #ed8936; }}
        .job-body {{ padding: 0 20px 16px; }}
        .job-detail {{ margin: 6px 0; font-size: 14px; color: #4a5568; }}
        .job-detail-label {{ color: #718096; font-weight: 500; }}
        .job-section {{ margin: 12px 0; }}
        .section-title {{ font-weight: 600; font-size: 13px; color: #2d3748; margin-bottom: 6px; }}
        .match-reason {{ background: #f0fff4; border-left: 3px solid #48bb78; padding: 10px 12px; border-radius: 4px; font-size: 14px; color: #276749; }}
        .concerns {{ background: #fffaf0; border-left: 3px solid #ed8936; padding: 10px 12px; border-radius: 4px; font-size: 14px; color: #9c4221; }}
        .apply-btn {{ display: inline-block; background: #667eea; color: white; padding: 10px 20px; text-decoration: none; border-radius: 6px; font-weight: 500; font-size: 14px; }}
        .apply-btn:hover {{ background: #5568d3; }}
        .footer {{ background: #f8f9fa; padding: 16px 20px; text-align: center; font-size: 12px; color: #718096; border-radius: 0 0 12px 12px; }}
        .stats {{ background: #edf2f7; padding: 10px; border-radius: 6px; font-size: 12px; color: #4a5568; display: inline-block; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>JobScout Daily Digest</h1>
            <p>{date_str} • {len(jobs)} matches found</p>
        </div>

        <div class="content">
"""

        # Add jobs
        for i, job in enumerate(jobs, 1):
            score_class = "high" if job.match_score >= 80 else "medium"
            salary_line = f'<div class="job-detail"><span class="job-detail-label">Salary:</span> {self._escape_html(job.salary)}</div>' if job.salary else ""

            # Build match reason HTML
            match_reason_html = ""
            if job.why_match:
                match_reason_html = f"""
                <div class="job-section">
                    <div class="section-title">✓ Why it matches</div>
                    <div class="match-reason">{self._escape_html(job.why_match)}</div>
                </div>
"""

            # Build skills HTML
            skills_html = ""
            if job.required_skills_matched:
                skills_html = f"""
                <div class="job-section">
                    <div class="section-title">Skills matched</div>
                    <div class="job-detail">{', '.join(self._escape_html(s) for s in job.required_skills_matched[:8])}</div>
                </div>
"""

            # Build concerns HTML
            concerns_html = ""
            if job.concerns:
                concerns_text = ' • '.join(self._escape_html(c) for c in job.concerns[:3])
                concerns_html = f"""
                <div class="job-section">
                    <div class="section-title">⚠ Things to note</div>
                    <div class="concerns">{concerns_text}</div>
                </div>
"""

            html += f"""
            <div class="job">
                <div class="job-header">
                    <div class="job-info">
                        <div class="job-title">{i}. {self._escape_html(job.job_title)}</div>
                        <div class="job-company">{self._escape_html(job.company)}</div>
                    </div>
                    <div class="job-score {score_class}">{job.match_score:.0f}%</div>
                </div>
                <div class="job-body">
                    <div class="job-detail">
                        <span class="job-detail-label">Location:</span> {self._escape_html(job.location)}
                    </div>
                    {salary_line}
                    {match_reason_html}
                    {skills_html}
                    {concerns_html}
                    <div class="job-section">
                        <a href="{job.url}" class="apply-btn">Apply Now →</a>
                    </div>
                </div>
            </div>
"""

        # Footer with stats
        html += f"""
        </div>

        <div class="footer">
            <div class="stats">
                Showing top {len(jobs)} matches from {total_evaluated} jobs evaluated
            </div>
        </div>
    </div>
</body>
</html>
"""

        return html

    def _get_subject(self, jobs: List[JobEvaluation], candidate_name: str) -> str:
        """Generate email subject."""
        name_part = f" for {candidate_name}" if candidate_name else ""
        return f"JobScout: {len(jobs)} matching jobs{name_part}"

    def _send_smtp(self, html: str, subject: str) -> dict:
        """Send email via SMTP."""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_config.get("smtp_from") or f"JobScout <{self.smtp_config.get('smtp_user')}>"
            msg['To'] = self.smtp_config.get("to_address")

            msg.attach(MIMEText(html, 'html'))

            with smtplib.SMTP(self.smtp_config["smtp_host"], self.smtp_config["smtp_port"]) as server:
                server.starttls()
                server.login(self.smtp_config["smtp_user"], self.smtp_config["smtp_password"])
                server.send_message(msg)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"Email sent to {self.smtp_config['to_address']}")
            return {"sent": True, "mode": "smtp", "digest_id": timestamp}

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            # Fallback to outbox
            return self._write_outbox(html, subject)

    def _write_outbox(self, html: str, subject: str) -> dict:
        """Write email to outbox directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jobscout_digest_{timestamp}.html"
        filepath = self.outbox_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"<!-- Subject: {subject} -->\n")
            f.write(html)

        logger.info(f"Email written to outbox: {filepath}")
        return {"sent": True, "mode": "outbox", "digest_id": timestamp}

    def _write_empty_digest(self, total_evaluated: int = 0) -> dict:
        """Write empty digest when no jobs found."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jobscout_digest_{timestamp}.html"
        filepath = self.outbox_dir / filename

        html = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f7fa; }}
        .container {{ max-width: 500px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h2 {{ color: #48bb78; }}
        p {{ color: #718096; line-height: 1.6; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>😔 No matching jobs found today</h2>
        <p>JobScout ran but didn't find any jobs that match your profile.</p>
        <p>This is normal — the conservative filter ensures you only see relevant opportunities.</p>
        <p><small>{total_evaluated} jobs were evaluated.</small></p>
    </div>
</body>
</html>
"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Empty digest written to outbox: {filepath}")
        return {"sent": True, "mode": "outbox", "digest_id": timestamp}

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        return (str(text)
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))
