"""Device modes and the /cloak-image endpoint."""
import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.config import CloakConfig
from backend.pipeline import CloakPipeline
from backend.tests.test_pipeline import _synthetic_frame


def _jpeg_bytes(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def test_device_modes_resolve():
    # On this VM there is no CUDA, so every GPU-asking mode must fall
    # back to CPU instead of crashing.
    for mode in ("auto", "cpu", "cuda", "both"):
        cfg = CloakConfig(device=mode)
        assert cfg.resolved_device() in ("cpu", "cuda")
        status = cfg.device_status()
        assert status["mode"] == mode
        assert status["compute_device"] == cfg.resolved_device()
        assert status["detector_device"] == "cpu"


def test_device_status_endpoint():
    c = TestClient(app)
    r = c.get("/device")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] in ("auto", "cpu", "cuda", "both")
    assert isinstance(body["cuda_available"], bool)
    assert body["compute_device"] in ("cpu", "cuda")


def test_config_device_change_rebuilds_pipeline():
    c = TestClient(app)
    r = c.post("/config", json={"device": "cpu"})
    assert r.status_code == 200
    assert r.json()["config"]["device"] == "cpu"
    r = c.post("/config", json={"device": "bogus"})
    assert r.status_code == 200
    assert r.json()["config"]["device"] == "cpu"  # invalid values are ignored
    c.post("/config", json={"device": "auto"})


def test_cloak_image_cloaks_faces():
    pipe = CloakPipeline(CloakConfig())
    try:
        result = pipe.cloak_image(_synthetic_frame())
    finally:
        pipe.close()
    assert result.metrics["faces"] >= 1
    assert result.metrics["similarity_after"] < result.metrics["similarity_before"]
    assert result.metrics["ssim"] > 0.8
    assert result.protected_bgr.shape[:2] == _synthetic_frame().shape[:2]


def test_cloak_image_endpoint():
    c = TestClient(app)
    files = {"file": ("face.jpg", _jpeg_bytes(_synthetic_frame()), "image/jpeg")}
    r = c.post("/cloak-image", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["faces"] >= 1
    assert body["similarity_after"] < body["similarity_before"]
    assert body["ssim"] > 0.8
    assert body["image"].startswith("/9j/")  # JPEG base64
    assert body["original"].startswith("/9j/")


def test_cloak_image_no_face_is_422():
    c = TestClient(app)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    files = {"file": ("blank.jpg", _jpeg_bytes(blank), "image/jpeg")}
    r = c.post("/cloak-image", files=files)
    assert r.status_code == 422
    assert "no face" in r.json()["error"]


def test_cloak_image_bad_file_is_400():
    c = TestClient(app)
    files = {"file": ("x.jpg", b"this is not an image", "image/jpeg")}
    r = c.post("/cloak-image", files=files)
    assert r.status_code == 400
