"""
Security Module for Quantum Traffic Optimizer.

This module provides authentication, authorization, rate limiting,
and security headers for the FastAPI application.
"""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# Conditional imports for JWT - gracefully handle if not installed
try:
    from jose import JWTError, jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    JWTError = Exception

from .config import Settings, get_settings


# =========================
# Security Schemes
# =========================

# Bearer token authentication
bearer_scheme = HTTPBearer(auto_error=False)

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# =========================
# Token Models
# =========================

class Token(BaseModel):
    """JWT Token response model."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class TokenData(BaseModel):
    """Token payload data."""
    sub: str = Field(..., description="Subject (user ID)")
    exp: datetime = Field(..., description="Expiration time")
    iat: datetime = Field(..., description="Issued at time")
    scopes: list[str] = Field(default_factory=list, description="Token scopes")
    jti: Optional[str] = Field(default=None, description="JWT ID for revocation")


class User(BaseModel):
    """User model for authentication."""
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(default=None, description="Email address")
    is_active: bool = Field(default=True, description="User is active")
    scopes: list[str] = Field(default_factory=list, description="User scopes")


# =========================
# JWT Functions
# =========================

def create_access_token(
    data: dict,
    settings: Settings,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload data to encode.
        settings: Application settings.
        expires_delta: Optional custom expiration time.

    Returns:
        Encoded JWT token string.

    Raises:
        RuntimeError: If JWT library is not available.
    """
    if not JWT_AVAILABLE:
        raise RuntimeError("JWT authentication requires 'python-jose' package")

    to_encode = data.copy()

    # Set expiration
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    # Add standard claims
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_urlsafe(16)
    })

    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str, settings: Settings) -> TokenData:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string.
        settings: Application settings.

    Returns:
        TokenData with decoded payload.

    Raises:
        HTTPException: If token is invalid or expired.
    """
    if not JWT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT authentication not available"
        )

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )

        return TokenData(
            sub=payload.get("sub", ""),
            exp=datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc),
            iat=datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc),
            scopes=payload.get("scopes", []),
            jti=payload.get("jti")
        )

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )


# =========================
# Authentication Dependencies
# =========================

async def verify_bearer_token(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)],
    settings: Annotated[Settings, Depends(get_settings)]
) -> Optional[TokenData]:
    """
    Verify Bearer token if provided.

    Args:
        credentials: HTTP Bearer credentials.
        settings: Application settings.

    Returns:
        TokenData if valid token, None if no token provided.

    Raises:
        HTTPException: If token is invalid.
    """
    if credentials is None:
        return None

    return decode_token(credentials.credentials, settings)


async def verify_api_key(
    api_key: Annotated[Optional[str], Depends(api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)]
) -> Optional[str]:
    """
    Verify API key if provided.

    Args:
        api_key: API key from header.
        settings: Application settings.

    Returns:
        API key if valid, None if not provided.

    Raises:
        HTTPException: If API key is invalid.
    """
    if api_key is None:
        return None

    if not settings.API_KEYS_ENABLED:
        return api_key

    valid_keys = settings.valid_api_keys_list

    if not valid_keys:
        # No API keys configured, allow all
        return api_key

    # Use constant-time comparison to prevent timing attacks
    for valid_key in valid_keys:
        if secrets.compare_digest(api_key, valid_key):
            return api_key

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "API-Key"}
    )


async def require_auth(
    token_data: Annotated[Optional[TokenData], Depends(verify_bearer_token)],
    api_key: Annotated[Optional[str], Depends(verify_api_key)],
    settings: Annotated[Settings, Depends(get_settings)]
) -> User:
    """
    Require authentication via JWT or API key.

    This is a strict authentication dependency that requires
    either a valid JWT token or API key.

    Args:
        token_data: Decoded JWT token data.
        api_key: Verified API key.
        settings: Application settings.

    Returns:
        User object representing the authenticated user.

    Raises:
        HTTPException: If neither authentication method succeeds.
    """
    # In development mode, authentication may be optional
    if settings.is_development and not settings.API_KEYS_ENABLED:
        return User(
            id="dev-user",
            username="developer",
            scopes=["read", "write", "admin"]
        )

    if token_data is not None:
        return User(
            id=token_data.sub,
            username=token_data.sub,
            scopes=token_data.scopes
        )

    if api_key is not None:
        # API key authentication - create a service user
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8]
        return User(
            id=f"api-key-{key_hash}",
            username=f"api-user-{key_hash}",
            scopes=["read", "write"]
        )

    # No authentication provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer, API-Key"}
    )


async def optional_auth(
    token_data: Annotated[Optional[TokenData], Depends(verify_bearer_token)],
    api_key: Annotated[Optional[str], Depends(verify_api_key)],
    settings: Annotated[Settings, Depends(get_settings)]
) -> Optional[User]:
    """
    Optional authentication - returns None if not authenticated.

    Use this for endpoints that work both with and without authentication.

    Args:
        token_data: Decoded JWT token data.
        api_key: Verified API key.
        settings: Application settings.

    Returns:
        User if authenticated, None otherwise.
    """
    try:
        return await require_auth(token_data, api_key, settings)
    except HTTPException:
        return None


def require_scopes(*required_scopes: str):
    """
    Dependency factory that requires specific scopes.

    Args:
        required_scopes: Scopes that the user must have.

    Returns:
        Dependency function that checks scopes.
    """
    async def check_scopes(user: Annotated[User, Depends(require_auth)]) -> User:
        for scope in required_scopes:
            if scope not in user.scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required scope: {scope}"
                )
        return user

    return check_scopes


# =========================
# Rate Limiting
# =========================

class RateLimiter:
    """
    Simple in-memory rate limiter.

    For production, use Redis-backed rate limiting.
    """

    def __init__(self):
        self._requests: dict[str, list[float]] = {}

    def _cleanup_old_requests(self, key: str, window_seconds: int) -> None:
        """Remove requests outside the time window."""
        if key not in self._requests:
            return

        cutoff = datetime.now(timezone.utc).timestamp() - window_seconds
        self._requests[key] = [
            ts for ts in self._requests[key]
            if ts > cutoff
        ]

    def is_allowed(self, key: str, limit: int, window_seconds: int) -> bool:
        """
        Check if a request is allowed under the rate limit.

        Args:
            key: Unique identifier (e.g., IP address).
            limit: Maximum requests allowed.
            window_seconds: Time window in seconds.

        Returns:
            True if request is allowed, False otherwise.
        """
        self._cleanup_old_requests(key, window_seconds)

        if key not in self._requests:
            self._requests[key] = []

        if len(self._requests[key]) >= limit:
            return False

        self._requests[key].append(datetime.now(timezone.utc).timestamp())
        return True

    def get_remaining(self, key: str, limit: int, window_seconds: int) -> int:
        """Get remaining requests in the window."""
        self._cleanup_old_requests(key, window_seconds)
        current = len(self._requests.get(key, []))
        return max(0, limit - current)


# Global rate limiter instance
rate_limiter = RateLimiter()


def get_client_ip(request: Request) -> str:
    """
    Get client IP address from request.

    Handles X-Forwarded-For header for proxied requests.

    Args:
        request: FastAPI request object.

    Returns:
        Client IP address string.
    """
    # Check for proxy headers
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct connection
    return request.client.host if request.client else "unknown"


async def check_rate_limit(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)]
) -> None:
    """
    Rate limiting dependency.

    Args:
        request: FastAPI request object.
        settings: Application settings.

    Raises:
        HTTPException: If rate limit exceeded.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    client_ip = get_client_ip(request)

    # Check per-minute limit
    minute_key = f"rate:{client_ip}:minute"
    if not rate_limiter.is_allowed(minute_key, settings.RATE_LIMIT_PER_MINUTE, 60):
        remaining = rate_limiter.get_remaining(minute_key, settings.RATE_LIMIT_PER_MINUTE, 60)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": str(settings.RATE_LIMIT_PER_MINUTE),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int((datetime.now(timezone.utc) + timedelta(minutes=1)).timestamp()))
            }
        )

    # Check per-hour limit
    hour_key = f"rate:{client_ip}:hour"
    if not rate_limiter.is_allowed(hour_key, settings.RATE_LIMIT_PER_HOUR, 3600):
        remaining = rate_limiter.get_remaining(hour_key, settings.RATE_LIMIT_PER_HOUR, 3600)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Hourly rate limit exceeded",
            headers={
                "Retry-After": "3600",
                "X-RateLimit-Limit": str(settings.RATE_LIMIT_PER_HOUR),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()))
            }
        )


# =========================
# Security Headers Middleware
# =========================

class SecurityHeadersMiddleware:
    """
    Middleware to add security headers to all responses.

    Implements OWASP security headers recommendations.
    """

    # Security headers configuration
    HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
        "Pragma": "no-cache",
    }

    # Content Security Policy for HTML responses
    CSP = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https://*.tile.openstreetmap.org; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))

                # Add security headers
                for key, value in self.HEADERS.items():
                    headers[key.lower().encode()] = value.encode()

                # Add CSP for HTML responses
                content_type = headers.get(b"content-type", b"").decode()
                if "text/html" in content_type:
                    headers[b"content-security-policy"] = self.CSP.encode()

                message["headers"] = list(headers.items())

            await send(message)

        await self.app(scope, receive, send_wrapper)


# =========================
# Utility Functions
# =========================

def generate_api_key() -> str:
    """
    Generate a secure API key.

    Returns:
        Random 32-character API key.
    """
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for storage.

    Args:
        api_key: Plain text API key.

    Returns:
        SHA-256 hash of the API key.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key_hash(api_key: str, hashed_key: str) -> bool:
    """
    Verify an API key against its hash.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        api_key: Plain text API key to verify.
        hashed_key: Stored hash to compare against.

    Returns:
        True if the key matches, False otherwise.
    """
    return hmac.compare_digest(hash_api_key(api_key), hashed_key)
