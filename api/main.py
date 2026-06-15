"""
RWITC Horse Racing API
----------------------
Run locally:
    uvicorn main:app --reload --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
OpenAPI JSON: http://localhost:8000/openapi.json
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys

from config import settings
from api.routes import venues, meetings, races, runners, horses, jockeys, trainers, schema
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    # Swagger UI lives at /docs
    docs_url="/docs",
    # ReDoc lives at /redoc
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Venues",
            "description": "Race venues / tracks.",
        },
        {
            "name": "Meetings",
            "description": "Race meetings — one meeting = one day at one venue.",
        },
        {
            "name": "Races",
            "description": "Individual races within a meeting, including dividends and exotic pools.",
        },
        {
            "name": "Runners",
            "description": "Per-horse per-race data: acceptance, declaration, equipment, result.",
        },
        {
            "name": "Horses",
            "description": "Horse registry, alias history, rating history, and medical records.",
        },
        {
            "name": "Jockeys",
            "description": "Jockey registry, rides history, and win statistics.",
        },
        {
            "name": "Trainers",
            "description": "Trainer registry, runner history, and win statistics.",
        },
        {
            "name": "Schema & Stats",
            "description": "Database introspection: tables, columns, row counts, recent activity.",
        },
    ],
)

# ---------------------------------------------------------------------------
# CORS — adjust origins for production
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(venues.router)
app.include_router(meetings.router)
app.include_router(races.router)
app.include_router(runners.router)
app.include_router(horses.router)
app.include_router(jockeys.router)
app.include_router(trainers.router)
app.include_router(schema.router)


# ---------------------------------------------------------------------------
# Root health check
# ---------------------------------------------------------------------------

@app.get("/", tags=["Schema & Stats"], summary="Health check")
def root():
    return {
        "status": "ok",
        "api": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
    }
