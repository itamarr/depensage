"""
System endpoints: health, config, auth.
"""

from fastapi import APIRouter, Request, Response, Depends, HTTPException
from pydantic import BaseModel

import hashlib
import json
import os

from depensage.config.settings import load_settings, get_all_years, get_config_path
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
    return HealthResponse(status="ok", version="1.0.0")


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


@router.get("/config", response_model=ConfigResponse, dependencies=[Depends(require_auth)])
async def config():
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


class SpreadsheetEntry(BaseModel):
    spreadsheet_id: str
    year: int | None = None
    default: bool = False


class ConfigSetRequest(BaseModel):
    key: str
    value: str


@router.get("/spreadsheets", dependencies=[Depends(require_auth)])
async def list_spreadsheets():
    """List all spreadsheet entries with full details."""
    settings = load_settings(force_reload=True)
    entries = {}
    for key, entry in settings["spreadsheets"].items():
        entries[key] = {
            "id": entry["id"][:20] + "...",
            "year": entry.get("year"),
            "default": entry.get("default", False),
        }
    return {"spreadsheets": entries}


@router.post("/spreadsheets/{key}", dependencies=[Depends(require_auth)])
async def add_spreadsheet(key: str, entry: SpreadsheetEntry):
    """Add a new spreadsheet entry."""
    config_path = get_config_path(None)
    with open(config_path) as f:
        config = json.load(f)

    if key in config.get("spreadsheets", {}):
        raise HTTPException(status_code=409, detail=f"'{key}' already exists")

    new_entry: dict = {"id": entry.spreadsheet_id}
    if entry.year is not None:
        new_entry["year"] = entry.year
    if entry.default:
        new_entry["default"] = True
        # Unset other defaults for same year
        if entry.year:
            for k, e in config["spreadsheets"].items():
                if e.get("year") == entry.year:
                    e.pop("default", None)

    config.setdefault("spreadsheets", {})[key] = new_entry
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    load_settings(force_reload=True)
    return {"status": "added", "key": key}


@router.put("/spreadsheets/{key}", dependencies=[Depends(require_auth)])
async def update_spreadsheet(key: str, entry: SpreadsheetEntry):
    """Update an existing spreadsheet entry."""
    config_path = get_config_path(None)
    with open(config_path) as f:
        config = json.load(f)

    if key not in config.get("spreadsheets", {}):
        raise HTTPException(status_code=404, detail=f"'{key}' not found")

    existing = config["spreadsheets"][key]
    if entry.spreadsheet_id:
        existing["id"] = entry.spreadsheet_id
    if entry.year is not None:
        existing["year"] = entry.year
    if entry.default:
        existing["default"] = True
        for k, e in config["spreadsheets"].items():
            if k != key and e.get("year") == existing.get("year"):
                e.pop("default", None)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    load_settings(force_reload=True)
    return {"status": "updated", "key": key}


@router.delete("/spreadsheets/{key}", dependencies=[Depends(require_auth)])
async def remove_spreadsheet(key: str):
    """Remove a spreadsheet entry."""
    config_path = get_config_path(None)
    with open(config_path) as f:
        config = json.load(f)

    if key not in config.get("spreadsheets", {}):
        raise HTTPException(status_code=404, detail=f"'{key}' not found")

    del config["spreadsheets"][key]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    load_settings(force_reload=True)
    return {"status": "deleted", "key": key}


class PasswordChange(BaseModel):
    password: str


@router.post("/password", dependencies=[Depends(require_auth)])
async def change_password(req: PasswordChange):
    """Change the web app password."""
    config_path = get_config_path(None)
    with open(config_path) as f:
        config = json.load(f)
    config["web_password"] = hashlib.sha256(req.password.encode()).hexdigest()
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return {"status": "updated"}
