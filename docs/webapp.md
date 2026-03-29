# Web App

## Overview

The web app is the end-user product — a wife-friendly interface for processing statements, reviewing data, and managing expense tracking. It wraps the same core library as the CLI.

**Stack**: FastAPI (backend) + Svelte/SvelteKit (frontend) + Tailwind CSS. Single process serves both API and static frontend.

## Architecture

```
Browser (any device on LAN)
    ↓ HTTPS
FastAPI (depensage/web/)
    ├── /api/* → REST endpoints (pipeline, staging, lookups, months, categories)
    ├── /_app/* → static JS/CSS bundles
    └── /* → SPA fallback (index.html for client-side routing)
    ↓
Core Library (depensage/engine/, classifier/, sheets/)
    ↓
Google Sheets API
```

The web layer is a thin wrapper — same position as the CLI. All business logic stays in the engine modules.

## Project Structure

```
depensage/web/              # Backend (Python)
├── app.py                  # FastAPI app factory, auth middleware, SPA fallback
├── main.py                 # Entry point: python -m depensage.web.main
├── auth.py                 # Password auth, session cookies, rate limiting
├── session.py              # In-memory pipeline session store (2h TTL)
├── models.py               # Pydantic request/response models
└── routers/
    ├── pipeline.py         # Upload → run (SSE) → review → commit
    ├── staging.py          # Classification, editing (Phase 2)
    ├── lookups.py          # Lookup table CRUD (Phase 3)
    ├── categories.py       # Category management (Phase 3)
    ├── months.py           # Month viewer/editor (Phase 3)
    ├── stats.py            # Statistics (Phase 5)
    └── system.py           # Health, config, login

frontend/                   # Frontend (Svelte)
├── src/
│   ├── routes/             # SvelteKit pages
│   │   ├── +layout.svelte  # NavSidebar + auth guard
│   │   ├── +page.svelte    # Dashboard
│   │   ├── pipeline/       # Pipeline wizard (multi-step)
│   │   ├── months/         # Month viewer
│   │   ├── lookups/        # Lookup management
│   │   ├── categories/     # Category management
│   │   ├── stats/          # Statistics (placeholder)
│   │   └── login/          # Login page
│   └── lib/
│       ├── api.ts          # Fetch wrapper, SSE helper
│       ├── stores.ts       # Svelte stores
│       └── components/     # Reusable components
└── build/                  # Built output (gitignored)
```

## Setup & Running

### Prerequisites

- Python 3.12+ with the project installed (`uv pip install -e ".[web]"` or install fastapi/uvicorn/python-multipart)
- Node.js 18+ and npm

### Build Frontend

```bash
npm install --prefix frontend          # First time only
npm run build --prefix frontend        # Build SPA
cp -r frontend/build depensage/web/static  # Copy to backend
```

### Configure

Add to `.secrets/config.json`:
```json
{
  "web_password": "your-strong-password-here"
}
```

### Generate SSL Certificate (optional, for HTTPS)

```bash
openssl req -x509 -newkey rsa:2048 \
  -keyout .secrets/key.pem -out .secrets/cert.pem \
  -days 365 -nodes -subj "/CN=depensage"
```

### Run

```bash
# HTTPS (production)
python -m depensage.web.main

# HTTP (local dev)
python -m depensage.web.main --no-ssl

# Custom host/port
python -m depensage.web.main --host 0.0.0.0 --port 9000
```

Access from any device on the network: `https://<desktop-ip>:8000`

### Frontend Development

For live-reload during frontend development:

```bash
# Terminal 1: backend
python -m depensage.web.main --no-ssl

# Terminal 2: frontend dev server (auto-proxies /api to backend)
npm run dev --prefix frontend
```

Access the dev server at `http://localhost:5173` (Vite dev server with hot reload).

## API Endpoints

### Authentication
- `POST /api/system/auth` — Login (password → session cookie)
- `GET /api/system/health` — Health check (public)
- `GET /api/system/config` — Spreadsheet config

### Pipeline
- `POST /api/pipeline/upload` — Upload XLSX files → session_id
- `POST /api/pipeline/{id}/run` — Start pipeline (SSE progress)
- `GET /api/pipeline/{id}/progress` — Server-Sent Events stream
- `GET /api/pipeline/{id}/result` — Staged result summary
- `GET /api/pipeline/{id}/months/{month}/{year}` — Month detail
- `POST /api/pipeline/{id}/commit` — Write to Google Sheets
- `DELETE /api/pipeline/{id}` — Discard session

### Planned (not yet implemented)
- `/api/staging/*` — Classification & editing (Phase 2)
- `/api/lookups/*` — Lookup table CRUD (Phase 3)
- `/api/categories` — Category management (Phase 3)
- `/api/months/*` — Month data viewing/editing (Phase 3)
- `/api/history` — Run history (Phase 4)
- `/api/stats` — Statistics (Phase 5)

## Security

- **Auth**: Simple password in config.json + session cookie + rate limiting (5 attempts/min)
- **HTTPS**: Self-signed cert for local network. Swap to Let's Encrypt for cloud.
- **Primary boundary**: WiFi password (WPA2/3). Web app password is a second layer.
- **No sensitive data stored**: Service account credentials are on the server filesystem. The app doesn't store CC numbers or bank login credentials.

## Design Decisions

- **On-demand deployment**: No systemd service. Start when needed, stop when done.
- **Trivially redeployable**: No hardcoded localhost, config-driven paths. Same command works on a VPS or behind Tailscale.
- **English UI, Hebrew data**: Navigation and labels in English. Table cell contents in Hebrew with RTL styling.
- **Mobile-responsive**: Designed for phone access from the start.
- **Run history**: Persisted to `.artifacts/run_history.json` (last 50 runs).
- **No file retention**: Uploaded statements deleted when session expires. Re-upload if needed.
