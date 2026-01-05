"""Email delivery for job digests."""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import List
from .scoring import ScoredJob


logger = logging.getLogger(__name__)


class EmailDelivery:
    """Send job digest emails via SMTP or write to outbox."""

    def __init__(self, config):
        """Initialize email delivery."""
        self.config = config
        self.outbox_dir = Path(config.outbox_dir)
        self.outbox_dir.mkdir(exist_ok=True)

    def send_digest(self, scored_jobs: List[ScoredJob]) -> bool:
        """
        Send job digest email.

        Returns True if successful, False otherwise.
        """
        if not scored_jobs:
            logger.info("No jobs to send in digest")
            return True

        # Generate email content
        subject, body = self._generate_email_content(scored_jobs)

        # Log preview
        logger.info(f"Email digest preview:\nSubject: {subject}\n")
        logger.info(f"Body preview (first 500 chars):\n{body[:500]}...")

        # Try SMTP if configured
        if self._is_smtp_configured():
            return self._send_via_smtp(subject, body, scored_jobs)
        else:
            return self._write_to_outbox(subject, body, scored_jobs)

    def _is_smtp_configured(self) -> bool:
        """Check if SMTP is properly configured."""
        email_cfg = self.config.email
        return all([
            email_cfg.smtp_host,
            email_cfg.smtp_username,
            email_cfg.smtp_password,
            email_cfg.smtp_from
        ])

    def _send_via_smtp(self, subject: str, body: str, scored_jobs: List[ScoredJob]) -> bool:
        """Send email via SMTP."""
        email_cfg = self.config.email

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = email_cfg.smtp_from
            msg['To'] = email_cfg.to_address

            # Attach HTML body
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)

            # Connect and send
            with smtplib.SMTP(email_cfg.smtp_host, email_cfg.smtp_port) as server:
                server.starttls()
                server.login(email_cfg.smtp_username, email_cfg.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent successfully to {email_cfg.to_address}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}")
            logger.info("Falling back to outbox")
            return self._write_to_outbox(subject, body, scored_jobs)

    def _write_to_outbox(self, subject: str, body: str, scored_jobs: List[ScoredJob]) -> bool:
        """Write email to outbox directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"jobscout_digest_{timestamp}.html"
        filepath = self.outbox_dir / filename

        try:
            # Write email file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Subject: {subject}\n")
                f.write(f"To: {self.config.email.to_address}\n")
                f.write(f"Date: {datetime.now().isoformat()}\n")
                f.write("\n" + "="*80 + "\n\n")
                f.write(body)

            logger.info(f"Email written to outbox: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to write email to outbox: {e}")
            return False

    def _generate_email_content(self, scored_jobs: List[ScoredJob]) -> tuple[str, str]:
        """Generate email subject and HTML body."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        count = len(scored_jobs)

        subject = f"JobScout: {count} apply-ready match{'es' if count != 1 else ''} ({date_str})"

        body = self._generate_html_body(scored_jobs, date_str)

        return subject, body

    def _generate_html_body(self, scored_jobs: List[ScoredJob], date_str: str) -> str:
        """Generate HTML email body."""
        # Limit to top 10 jobs
        jobs = scored_jobs[:10]

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2563eb; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }}
        .job {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 20px 0; background: #f9fafb; }}
        .job-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px; }}
        .job-title {{ font-size: 1.3em; font-weight: bold; color: #1f2937; margin: 0; }}
        .job-company {{ font-size: 1.1em; color: #4b5563; margin: 5px 0; }}
        .match-score {{ display: inline-block; background: #10b981; color: white; padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 0.9em; }}
        .job-meta {{ color: #6b7280; font-size: 0.95em; margin: 10px 0; }}
        .apply-btn {{ display: inline-block; background: #2563eb; color: white; padding: 10px 20px; text-decoration: none; border-radius: 6px; margin: 10px 0; }}
        .apply-btn:hover {{ background: #1d4ed8; }}
        .bullets {{ margin: 15px 0; padding-left: 20px; }}
        .bullets li {{ margin: 8px 0; line-height: 1.5; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 0.9em; }}
        .missing {{ color: #dc2626; }}
        .positive {{ color: #059669; }}
    </style>
</head>
<body>
    <h1>JobScout Daily Digest</h1>
    <p>{date_str} — {len(jobs)} high-confidence match{('es' if len(jobs) != 1 else '')} ready to apply</p>

    <p style="color: #6b7280; font-style: italic;">
        These jobs match your resume and can be applied to without edits.
        Conservatism is a feature — if you don't see a job here, it means we're not confident it's a match.
    </p>
"""

        for i, scored in enumerate(jobs, 1):
            job = scored.job

            # Build bullets
            bullets_html = self._generate_job_bullets(scored)

            html += f"""
    <div class="job">
        <div class="job-header">
            <div>
                <h2 class="job-title">{i}. {self._escape_html(job.title)}</h2>
                <div class="job-company">at {self._escape_html(job.company)}</div>
            </div>
            <div class="match-score">{scored.score:.0f}% match</div>
        </div>

        <div class="job-meta">
            <strong>Location:</strong> {self._escape_html(job.location)}
            <span style="margin: 0 10px;">|</span>
            <strong>Source:</strong> {self._escape_html(job.source)}
        </div>

        <ul class="bullets">
            {bullets_html}
        </ul>

        <div>
            <a href="{job.apply_url}" class="apply-btn">Apply Now</a>
        </div>
    </div>
"""

        html += f"""
    <div class="footer">
        <p>You received this email because you configured JobScout to send you daily job matches.</p>
        <p>To stop receiving these emails, disable scheduling in your config.yaml.</p>
    </div>
</body>
</html>
"""

        return html

    def _generate_job_bullets(self, scored: ScoredJob) -> str:
        """Generate bullet points for a job."""
        job = scored.job
        bullets = []

        # Why it matches
        if scored.matching_skills:
            skills_str = ", ".join(sorted(scored.matching_skills))
            bullets.append(f'<span class="positive">✓ Matches your skills:</span> {self._escape_html(skills_str)}')
        else:
            bullets.append('<span class="positive">✓ Aligns with your experience and role</span>')

        # Missing must-haves
        if scored.missing_must_haves:
            missing_str = ", ".join(sorted(scored.missing_must_haves))
            bullets.append(f'<span class="missing">Missing must-haves:</span> {self._escape_html(missing_str)}')
        else:
            bullets.append('<span class="positive">✓ All must-have requirements met</span>')

        # Stack alignment summary
        total_skills = len(job.must_have_skills | job.nice_to_have_skills)
        overlap = len(scored.matching_skills)
        if total_skills > 0:
            pct = int((overlap / total_skills) * 100)
            bullets.append(f'<span class="positive">✓ Stack alignment:</span> {pct}% ({overlap} of {total_skills} skills match)')
        else:
            bullets.append('<span class="positive">✓ Good overall fit for your profile</span>')

        return "\n            ".join(bullets)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#x27;'))
