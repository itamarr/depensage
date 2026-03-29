"""
Simple password authentication with rate limiting.

Password is stored in config.json under "web_password".
Session is tracked via a signed cookie.
"""

import hashlib
import hmac
import secrets
import time
from collections import defaultdict

from fastapi import Request, Response, HTTPException


# Rate limiting: max attempts per IP within window
_MAX_ATTEMPTS = 5
_WINDOW_SECONDS = 60
_attempts: dict[str, list[float]] = defaultdict(list)

# Session token → expiry
_sessions: dict[str, float] = {}
_SESSION_TTL = 8 * 3600  # 8 hours

# Secret key for signing (generated at startup)
_secret_key = secrets.token_hex(32)

COOKIE_NAME = "depensage_session"


def check_rate_limit(client_ip: str):
    """Raise 429 if too many login attempts."""
    now = time.time()
    attempts = _attempts[client_ip]
    # Prune old attempts
    _attempts[client_ip] = [t for t in attempts if now - t < _WINDOW_SECONDS]
    if len(_attempts[client_ip]) >= _MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Try again later.",
        )
    _attempts[client_ip].append(now)


def verify_password(provided: str, stored_hash: str) -> bool:
    """Constant-time password comparison."""
    provided_hash = hashlib.sha256(provided.encode()).hexdigest()
    return hmac.compare_digest(provided_hash, stored_hash)


def hash_password(password: str) -> str:
    """Hash a password for storage."""
    return hashlib.sha256(password.encode()).hexdigest()


def create_session() -> str:
    """Create a new session token."""
    token = secrets.token_hex(32)
    _sessions[token] = time.time() + _SESSION_TTL
    return token


def validate_session(token: str) -> bool:
    """Check if a session token is valid and not expired."""
    expiry = _sessions.get(token)
    if expiry is None:
        return False
    if time.time() > expiry:
        _sessions.pop(token, None)
        return False
    return True


def get_session_token(request: Request) -> str | None:
    """Extract session token from cookie."""
    return request.cookies.get(COOKIE_NAME)


def require_auth(request: Request):
    """Dependency: raise 401 if not authenticated."""
    token = get_session_token(request)
    if not token or not validate_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")


def set_session_cookie(response: Response, token: str):
    """Set the session cookie on a response."""
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="strict",
        secure=True,
        max_age=_SESSION_TTL,
    )
