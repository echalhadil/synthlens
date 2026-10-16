from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from ..core.detector import detect

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="synthlens", version="0.1.0",
              description="Detect AI-generated images with no-ML forensic analysis.")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MAX_BYTES = 25 * 1024 * 1024


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse(request, "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/detect")
async def api_detect(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="file too large (max 25MB)")
    try:
        result = detect(data, include_viz=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not process image: {exc}")
    payload = result.to_dict(include_viz=True)
    payload["filename"] = file.filename
    return JSONResponse(payload)