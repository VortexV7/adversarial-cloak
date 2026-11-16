"""FastAPI backend: WebSocket streaming + REST config / enroll / benchmark.

WebSocket ``/ws/stream`` protocol
---------------------------------
Client -> server: binary JPEG frames (browser webcam), or a JSON text frame
``{"image": "<base64 jpeg>"}``.
Server -> client: one JSON text frame per input frame::

    {"type": "frame", "original": "<b64 jpeg>", "protected": "<b64 jpeg>",
     "metrics": {...}, "boxes": [{"x":..,"y":..,"w":..,"h":..,"track_id":..}]}

REST
----
GET  /health            liveness + device + model status
GET  /config            current CloakConfig
POST /config            partial update (epsilon, steps, update_interval, ...)
POST /enroll            multipart image -> store reference embedding
GET  /enroll            enrolled identity (embedding hash only, never the vector)
DELETE /enroll          clear enrolment
POST /benchmark         run the eval on the bundled sample pairs
POST /export-report     CSV (and PDF when reportlab is installed) summary
POST /rtsp/start        start an RTSP/IP-camera source  {"url": "..."}
POST /rtsp/stop         stop it
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import CONFIG, CloakConfig
from .embedder import embedding_from_bytes, get_embedder
from .metrics import cosine_similarity
from .pipeline import CloakPipeline

app = FastAPI(title="Adversarial Cloak", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_pipeline: CloakPipeline | None = None
_enrolled_name: str | None = None
_enrolled_at: float | None = None
_rtsp_thread: threading.Thread | None = None
_rtsp_stop = threading.Event()
_rtsp_url: str | None = None
_ENROLL_FILE = Path(__file__).parent / "enrolled.json"


def get_pipeline() -> CloakPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CloakPipeline(CONFIG)
    return _pipeline


def _jpeg_b64(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, CONFIG.jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _decode_frame(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode JPEG frame")
    return frame


# ------------------------------------------------------------------ models
class ConfigUpdate(BaseModel):
    epsilon: float | None = None
    steps: int | None = None
    alpha: float | None = None
    update_interval: int | None = None
    ema_alpha: float | None = None
    surrogate: str | None = None
    evaluator: str | None = None
    use_eot: bool | None = None
    use_mask: bool | None = None
    enabled: bool | None = None
    enroll_threshold: float | None = None


class RtspRequest(BaseModel):
    url: str


# ------------------------------------------------------------------ REST
@app.get("/health")
def health():
    pipe = _pipeline
    return {
        "status": "ok",
        "device": CONFIG.resolved_device(),
        "surrogate": CONFIG.surrogate,
        "evaluator": CONFIG.evaluator,
        "detector": pipe.detector.backend if pipe else "not-started",
        "surrogate_fallback": pipe.surrogate.is_fallback if pipe else None,
        "evaluator_fallback": pipe.evaluator.is_fallback if pipe else None,
        "enrolled": _enrolled_name is not None,
        "rtsp_url": _rtsp_url,
    }


@app.get("/config")
def get_config():
    return CONFIG.__dict__


@app.post("/config")
def update_config(update: ConfigUpdate):
    changes = update.model_dump(exclude_unset=True)
    for key, value in changes.items():
        if key in ("surrogate", "evaluator") and value not in ("facenet", "arcface"):
            continue
        setattr(CONFIG, key, value)
    # A surrogate change needs a fresh pipeline (new embedder instance).
    if "surrogate" in changes or "evaluator" in changes:
        global _pipeline
        _pipeline = None
    return {"updated": changes, "config": CONFIG.__dict__}


@app.post("/enroll")
async def enroll(name: str = "user", file: UploadFile = File(...)):
    data = await file.read()
    pipe = get_pipeline()
    emb = embedding_from_bytes(data, pipe.evaluator)
    pipe.enrolled = emb
    global _enrolled_name, _enrolled_at
    _enrolled_name, _enrolled_at = name, time.time()
    _ENROLL_FILE.write_text(json.dumps(
        {"name": name, "at": _enrolled_at, "embedding": emb.tolist()}))
    return {"name": name, "embedding_dim": len(emb), "enrolled_at": _enrolled_at}


@app.get("/enroll")
def get_enrollment():
    if _enrolled_name is None and _ENROLL_FILE.exists():
        try:
            saved = json.loads(_ENROLL_FILE.read_text())
            return {"name": saved["name"], "enrolled_at": saved["at"], "embedding_dim": len(saved["embedding"])}
        except Exception:
            pass
    if _enrolled_name is None:
        return {"name": None, "enrolled": False}
    return {"name": _enrolled_name, "enrolled_at": _enrolled_at, "enrolled": True}


@app.delete("/enroll")
def clear_enrollment():
    pipe = get_pipeline()
    pipe.enrolled = None
    global _enrolled_name, _enrolled_at
    _enrolled_name, _enrolled_at = None, None
    if _ENROLL_FILE.exists():
        _ENROLL_FILE.unlink()
    return {"enrolled": False}


@app.post("/benchmark")
def benchmark(pairs: int = 20):
    """Run the evaluation harness on the bundled sample (or LFW if present)."""
    import subprocess
    import sys

    root = Path(__file__).parent.parent
    out = root / "eval" / "results.csv"
    cmd = [sys.executable, str(root / "eval" / "run_eval.py"), "--pairs", str(pairs),
           "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
        return {"status": "error", "stderr": proc.stderr[-2000:]}
    import csv as _csv

    rows = list(_csv.DictReader(open(out)))
    return {"status": "ok", "rows": rows, "stdout_tail": proc.stdout[-1000:]}


@app.post("/export-report")
def export_report(fmt: str = "csv"):
    pipe = get_pipeline()
    m = pipe._last_metrics or {}
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["metric", "value"])
    for k, v in m.items():
        if not isinstance(v, (list, dict)):
            w.writerow([k, v])
    if fmt == "pdf":
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas

            pdf = io.BytesIO()
            c = canvas.Canvas(pdf, pagesize=A4)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(72, 780, "Adversarial Cloak — Session Report")
            c.setFont("Helvetica", 11)
            y = 750
            for k, v in m.items():
                if isinstance(v, (list, dict)):
                    continue
                c.drawString(72, y, f"{k}: {v}")
                y -= 18
            c.save()
            pdf.seek(0)
            return _bytes_response(pdf.read(), "application/pdf", "cloak-report.pdf")
        except ImportError:
            # 501, not 200: the frontend only downloads the body when the
            # response is OK, so a 200 here used to save the JSON error as
            # "cloak-report.pdf" and the browser then reported
            # "Failed to load PDF document".
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=501,
                content={"error": "reportlab not installed; use fmt=csv"},
            )
    return _bytes_response(buf.getvalue().encode(), "text/csv", "cloak-report.csv")


def _bytes_response(data: bytes, media_type: str, filename: str):
    from fastapi.responses import Response

    return Response(content=data, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.post("/rtsp/start")
def rtsp_start(req: RtspRequest):
    global _rtsp_thread, _rtsp_url
    if _rtsp_thread and _rtsp_thread.is_alive():
        return {"status": "already-running", "url": _rtsp_url}
    _rtsp_stop.clear()
    _rtsp_url = req.url

    def _loop():
        pipe = get_pipeline()
        cap = cv2.VideoCapture(req.url)
        while not _rtsp_stop.is_set() and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            pipe.process_frame(frame)  # latest metrics available via /config+WS
        cap.release()

    _rtsp_thread = threading.Thread(target=_loop, daemon=True)
    _rtsp_thread.start()
    return {"status": "started", "url": req.url}


@app.post("/rtsp/stop")
def rtsp_stop():
    global _rtsp_url
    _rtsp_stop.set()
    _rtsp_url = None
    return {"status": "stopped"}


# ------------------------------------------------------------------ WebSocket
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    pipe = get_pipeline()
    try:
        while True:
            msg = await ws.receive()
            frame = None
            if "bytes" in msg and msg["bytes"]:
                frame = _decode_frame(msg["bytes"])
            elif "text" in msg and msg["text"]:
                try:
                    payload = json.loads(msg["text"])
                    if "image" in payload:
                        frame = _decode_frame(base64.b64decode(payload["image"]))
                except Exception:
                    await ws.send_json({"type": "error", "message": "bad frame payload"})
                    continue
            if frame is None:
                continue
            result = pipe.process_frame(frame)
            await ws.send_json({
                "type": "frame",
                "original": _jpeg_b64(frame),
                "protected": _jpeg_b64(result.protected_bgr),
                "metrics": result.metrics,
                "boxes": [{"x": b.x, "y": b.y, "w": b.w, "h": b.h,
                           "track_id": b.track_id, "score": round(b.score, 2)}
                          for b in result.boxes],
            })
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # never kill the socket on one bad frame
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
