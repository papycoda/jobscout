"""
Security middleware and utilities for JobScout API.
"""

import os
import re
import hashlib
import hmac
import logging
import time
from typing import Optional, Callable, Any
from functools import wraps
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers

logger = logging.getLogger(__name__)


class SecurityConfig:
    """Security configuration from environment variables."""

    # API Keys
    API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")
    API_KEY = os.getenv("API_KEY", "")
    REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"

    # Rate limiting
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # seconds

    # File upload security
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
    ALLOWED_FILE_EXTENSIONS = {".pdf", ".docx", ".txt"}

    # CORS
    ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []

    # Security headers
    ENABLE_SECURITY_HEADERS = os.getenv("ENABLE_SECURITY_HEADERS", "true").lower() == "true"


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self):
        self.requests = {}  # {identifier: [(timestamp, count)]}
        self.window = SecurityConfig.RATE_LIMIT_WINDOW
        self.max_requests = SecurityConfig.RATE_LIMIT_REQUESTS

    def _cleanup_old_requests(self, identifier: str):
        """Remove requests outside the time window."""
        if identifier not in self.requests:
            return

        cutoff_time = time.time() - self.window
        self.requests[identifier] = [
            (timestamp, count) for timestamp, count in self.requests[identifier]
            if timestamp > cutoff_time
        ]

    def check_rate_limit(self, identifier: str) -> tuple[bool, dict]:
        """Check if request is within rate limit."""
        if not SecurityConfig.RATE_LIMIT_ENABLED:
            return True, {}

        self._cleanup_old_requests(identifier)
        current_time = time.time()

        # Count requests in current window
        total_requests = sum(count for _, count in self.requests.get(identifier, []))

        if total_requests >= self.max_requests:
            # Calculate retry time
            if self.requests[identifier]:
                oldest_request = min(timestamp for timestamp, _ in self.requests[identifier])
                retry_after = int(oldest_request + self.window - current_time) + 1
            else:
                retry_after = self.window

            return False, {
                "retry_after": retry_after,
                "limit": self.max_requests,
                "window": self.window
            }

        # Add current request
        if identifier not in self.requests:
            self.requests[identifier] = []
        self.requests[identifier].append((current_time, 1))

        return True, {
            "remaining": self.max_requests - total_requests - 1,
            "limit": self.max_requests,
            "window": self.window
        }


# Global rate limiter instance
rate_limiter = RateLimiter()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        """Process request and add security headers."""
        response = await call_next(request)

        if SecurityConfig.ENABLE_SECURITY_HEADERS:
            # Content Security Policy
            csp_directives = [
                "default-src 'self'",
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
                "style-src 'self' 'unsafe-inline'",
                "img-src 'self' data: https:",
                "font-src 'self' data:",
                "connect-src 'self'",
                "frame-ancestors 'none'",
                "base-uri 'self'",
                "form-action 'self'"
            ]
            response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

            # Other security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

            # HSTS (only in production with HTTPS)
            if request.url.scheme == "https":
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""

    async def dispatch(self, request: Request, call_next):
        """Verify API key if required."""
        # Skip auth for health check and public endpoints
        if request.url.path in ["/health", "/api/health", "/debug"]:
            return await call_next(request)

        if SecurityConfig.REQUIRE_API_KEY:
            api_key = request.headers.get(SecurityConfig.API_KEY_HEADER)

            if not api_key:
                logger.warning(f"Missing API key for {request.url.path}")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "API key required"}
                )

            if not self._verify_api_key(api_key):
                logger.warning(f"Invalid API key for {request.url.path}")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Invalid API key"}
                )

            logger.debug(f"API key authenticated for {request.url.path}")

        return await call_next(request)

    def _verify_api_key(self, provided_key: str) -> bool:
        """Verify the provided API key."""
        if not SecurityConfig.API_KEY:
            logger.error("API key authentication enabled but no API_KEY configured")
            return False

        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(provided_key, SecurityConfig.API_KEY)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting."""

    async def dispatch(self, request: Request, call_next):
        """Check rate limits before processing request."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/api/health", "/debug"]:
            return await call_next(request)

        # Get client identifier
        identifier = self._get_client_identifier(request)

        # Check rate limit
        allowed, info = rate_limiter.check_rate_limit(identifier)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {identifier}")
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": info.get("retry_after", 60)
                }
            )
            response.headers["Retry-After"] = str(info.get("retry_after", 60))
            return response

        # Add rate limit info to response headers
        response = await call_next(request)
        if SecurityConfig.RATE_LIMIT_ENABLED:
            response.headers["X-RateLimit-Limit"] = str(info.get("limit", ""))
            response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", ""))
            response.headers["X-RateLimit-Window"] = str(info.get("window", ""))

        return response

    def _get_client_identifier(self, request: Request) -> str:
        """Get a unique identifier for the client."""
        # Try to get API key first (more specific)
        api_key = request.headers.get(SecurityConfig.API_KEY_HEADER)
        if api_key:
            # Hash the API key for privacy
            return hashlib.sha256(api_key.encode()).hexdigest()[:16]

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Get the first IP in the chain
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return ip


class SecurityValidator:
    """Input validation and sanitization."""

    @staticmethod
    def validate_file_upload(filename: str, content_length: int) -> tuple[bool, Optional[str]]:
        """Validate file upload parameters."""
        # Check file size
        if content_length > SecurityConfig.MAX_FILE_SIZE:
            return False, f"File too large (max {SecurityConfig.MAX_FILE_SIZE} bytes)"

        # Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in SecurityConfig.ALLOWED_FILE_EXTENSIONS:
            return False, f"Invalid file type. Allowed: {', '.join(SecurityConfig.ALLOWED_FILE_EXTENSIONS)}"

        # Check for suspicious patterns in filename
        if ".." in filename or "/" in filename or "\\" in filename:
            return False, "Invalid filename"

        return True, None

    @staticmethod
    def sanitize_email(email: str) -> str:
        """Sanitize email address."""
        # Basic email validation and sanitization
        email = email.strip().lower()
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            raise ValueError("Invalid email format")
        return email

    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize URL."""
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return url

    @staticmethod
    def validate_config_yaml(yaml_content: str) -> tuple[bool, Optional[str]]:
        """Validate YAML configuration for security issues."""
        # Check for dangerous patterns
        dangerous_patterns = [
            (r'__import__', "Direct import usage"),
            (r'eval\s*\(', "eval() usage"),
            (r'exec\s*\(', "exec() usage"),
            (r'\$\{.*\}', "Template injection pattern"),
        ]

        for pattern, description in dangerous_patterns:
            if re.search(pattern, yaml_content):
                return False, f"Potentially dangerous content: {description}"

        return True, None

    @staticmethod
    def mask_sensitive_data(data: str, data_type: str = "generic") -> str:
        """Mask sensitive data for logging."""
        if data_type == "api_key":
            if len(data) <= 8:
                return "*" * len(data)
            return data[:4] + "*" * (len(data) - 8) + data[-4:]
        elif data_type == "email":
            parts = data.split("@")
            if len(parts) == 2:
                return parts[0][:2] + "***@" + parts[1]
            return "***@***"
        else:
            if len(data) <= 4:
                return "*" * len(data)
            return data[:2] + "*" * (len(data) - 4) + data[-2:]


def require_api_key(func: Callable) -> Callable:
    """Decorator to require API key for specific endpoints."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if SecurityConfig.REQUIRE_API_KEY:
            # Extract request from args
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if request:
                api_key = request.headers.get(SecurityConfig.API_KEY_HEADER)
                if not api_key or not hmac.compare_digest(api_key, SecurityConfig.API_KEY):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="API key required"
                    )

        return await func(*args, **kwargs)
    return wrapper


class SecurityLogger:
    """Enhanced security logging."""

    @staticmethod
    def log_security_event(event_type: str, details: dict, request: Optional[Request] = None):
        """Log security events with standardized format."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details
        }

        if request:
            log_data["request"] = {
                "method": request.method,
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent", "unknown"),
                "client_ip": SecurityLogger._get_client_ip(request)
            }

        logger.warning(f"SECURITY: {event_type} - {log_data}")

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"