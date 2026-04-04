"""
FastAPI application factory.
"""

import asyncio
import os

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from depensage.web.auth import get_session_token, validate_session
from depensage.web.session import SessionStore
from depensage.web.routers import system, pipeline, staging, lookups, categories, months, stats, budget


# Paths that don't require authentication
_PUBLIC_PATHS = {"/api/system/auth", "/api/system/health"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Gate all routes behind authentication except login and health."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow public API paths
        if path in _PUBLIC_PATHS:
            return await call_next(request)

        # Allow static assets needed for login page (JS, CSS, fonts)
        if path.startswith("/_app/") or path == "/favicon.ico":
            return await call_next(request)

        # Check auth
        token = get_session_token(request)
        if not token or not validate_session(token):
            # For API calls, return 401 JSON
            if path.startswith("/api/"):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Not authenticated"},
                )
            # For page requests, serve the login page
            static_dir = os.path.join(os.path.dirname(__file__), "static")
            index = os.path.join(static_dir, "index.html")
            if os.path.exists(index):
                return FileResponse(index)
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"},
            )

        return await call_next(request)


def create_app() -> FastAPI:
    app = FastAPI(
        title="DepenSage",
        description="Household expense tracking",
        version="1.0.0",
    )

    # Auth middleware — gates everything behind login
    app.add_middleware(AuthMiddleware)

    # Session store
    app.state.session_store = SessionStore(ttl_hours=2)

    # Routers
    app.include_router(system.router)
    app.include_router(pipeline.router)
    app.include_router(staging.router)
    app.include_router(lookups.router)
    app.include_router(categories.router)
    app.include_router(months.router)
    app.include_router(stats.router)
    app.include_router(budget.router)

    # Background task: sweep expired sessions every 5 minutes
    @app.on_event("startup")
    async def start_session_sweeper():
        async def _sweep():
            while True:
                await asyncio.sleep(300)
                app.state.session_store.sweep_expired()
        asyncio.create_task(_sweep())

    # Serve frontend static files with SPA fallback
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        # Mount static assets (JS, CSS bundles)
        app.mount(
            "/_app", StaticFiles(directory=os.path.join(static_dir, "_app")),
            name="app-assets",
        )

        # SPA fallback: any non-API path serves index.html
        @app.get("/{path:path}")
        async def spa_fallback(path: str):
            # Try to serve the exact file first (for robots.txt etc.)
            file_path = os.path.join(static_dir, path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            # Otherwise serve index.html for client-side routing
            return FileResponse(os.path.join(static_dir, "index.html"))

    return app
