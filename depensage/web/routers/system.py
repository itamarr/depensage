"""
System endpoints: health, config, auth.
"""

from fastapi import APIRouter, Request, Response, Depends, HTTPException

from depensage.config.settings import load_settings, get_all_years
from depensage.web.auth import (
    check_rate_limit, verify_password, hash_password,
    create_session, set_session_cookie, require_auth,
)
from depensage.web.models import (
    LoginRequest, LoginResponse, HealthResponse, ConfigResponse,
)

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.1.0")


@router.post("/auth", response_model=LoginResponse)
async def login(req: LoginRequest, request: Request, response: Response):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    try:
        settings = load_settings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config error: {e}")

    stored_password = settings.get("web_password", "")
    if not stored_password:
        raise HTTPException(
            status_code=500,
            detail="web_password not configured in config.json",
        )

    # Support both plain text and hashed passwords
    if len(stored_password) == 64 and all(c in "0123456789abcdef" for c in stored_password):
        # Looks like a SHA-256 hash
        if not verify_password(req.password, stored_password):
            return LoginResponse(ok=False, message="Wrong password")
    else:
        # Plain text comparison (for initial setup convenience)
        if req.password != stored_password:
            return LoginResponse(ok=False, message="Wrong password")

    token = create_session()
    set_session_cookie(response, token)
    return LoginResponse(ok=True)


@router.get("/config", response_model=ConfigResponse)
async def config(request: Request = Depends(require_auth)):
    settings = load_settings()
    spreadsheets = {}
    for key, entry in settings["spreadsheets"].items():
        spreadsheets[key] = {
            "year": entry.get("year"),
            "default": entry.get("default", False),
        }
    return ConfigResponse(
        spreadsheets=spreadsheets,
        years=get_all_years(settings),
    )
