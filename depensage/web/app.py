"""
FastAPI application factory.
"""

import asyncio
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from depensage.web.session import SessionStore
from depensage.web.routers import system, pipeline


def create_app() -> FastAPI:
    app = FastAPI(
        title="DepenSage",
        description="Household expense tracking",
        version="0.1.0",
    )

    # Session store
    app.state.session_store = SessionStore(ttl_hours=2)

    # Routers
    app.include_router(system.router)
    app.include_router(pipeline.router)

    # Background task: sweep expired sessions every 5 minutes
    @app.on_event("startup")
    async def start_session_sweeper():
        async def _sweep():
            while True:
                await asyncio.sleep(300)
                app.state.session_store.sweep_expired()
        asyncio.create_task(_sweep())

    # Serve frontend static files (if built)
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app
