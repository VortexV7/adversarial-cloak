import cv2
import numpy as np

from backend.config import CloakConfig
from backend.pipeline import CloakPipeline


def _synthetic_frame():
    """480x640 frame with a drawn face. Shaded eye sockets, eyebrows, nose
    and mouth give Haar enough contrast to detect it even after JPEG."""
    frame = np.full((480, 640, 3), 210, dtype=np.uint8)
    cv2.ellipse(frame, (320, 170), (110, 80), 0, 180, 360, (40, 30, 20), -1)
    cv2.ellipse(frame, (320, 250), (95, 125), 0, 0, 360, (190, 160, 130), -1)
    cv2.ellipse(frame, (320, 190), (60, 35), 0, 0, 360, (205, 175, 145), -1)
    cv2.ellipse(frame, (282, 225), (28, 16), 0, 0, 360, (120, 95, 75), -1)
    cv2.ellipse(frame, (358, 225), (28, 16), 0, 0, 360, (120, 95, 75), -1)
    cv2.circle(frame, (282, 225), 9, (25, 25, 25), -1)
    cv2.circle(frame, (358, 225), 9, (25, 25, 25), -1)
    cv2.ellipse(frame, (282, 200), (26, 7), 0, 180, 360, (30, 20, 15), -1)
    cv2.ellipse(frame, (358, 200), (26, 7), 0, 180, 360, (30, 20, 15), -1)
    cv2.line(frame, (320, 235), (312, 285), (150, 120, 95), 4)
    cv2.line(frame, (312, 285), (328, 285), (150, 120, 95), 4)
    cv2.ellipse(frame, (320, 320), (38, 14), 0, 0, 180, (60, 40, 40), -1)
    cv2.ellipse(frame, (320, 315), (38, 10), 0, 0, 360, (170, 120, 110), -1)
    return frame


def test_pipeline_returns_frame_and_metrics():
    pipe = CloakPipeline(CloakConfig(steps=3, update_interval=5))
    result = pipe.process_frame(_synthetic_frame())
    assert result.protected_bgr.shape == (480, 640, 3)
    assert "similarity_after" in result.metrics
    assert "fps" in result.metrics and "ssim" in result.metrics


def test_pipeline_disabled_passes_frame_through():
    pipe = CloakPipeline(CloakConfig(enabled=False))
    frame = _synthetic_frame()
    result = pipe.process_frame(frame)
    assert np.array_equal(result.protected_bgr, frame)


def test_pipeline_video_loop_runs():
    pipe = CloakPipeline(CloakConfig(steps=2, update_interval=2))
    frames = [_synthetic_frame() for _ in range(6)]
    results = [pipe.process_frame(f) for f in frames]
    assert len(results) == 6
    assert all(r.metrics["latency_ms"] >= 0 for r in results)
