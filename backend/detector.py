"""Face detection with a margin, returning crops and coordinates.

Primary detector: MediaPipe **Tasks API** FaceDetector
(``blaze_face_short_range.tflite``, bundled in ``backend/models/``).

MediaPipe 1.x removed the legacy ``mp.solutions.face_detection`` API, so
this module uses the Tasks API and never touches ``mp.solutions``.
If MediaPipe or the model file is unavailable, it falls back to the
OpenCV Haar cascade, and finally to no detector at all -- in which case
:meth:`FaceDetector.detect` returns ``[]`` and the pipeline passes the
frame through unchanged (never crashes).

Coordinates are always in the original frame's pixel space.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_MODEL_PATH = Path(__file__).parent / "models" / "blaze_face_short_range.tflite"


@dataclass
class FaceBox:
    """One detected face."""

    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    track_id: int = -1  # filled in by temporal.Tracker

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    def iou(self, other: "FaceBox") -> float:
        """Intersection-over-union with another box (0..1)."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0


def _ensure_model() -> Path | None:
    """Return the bundled model path, downloading it once if missing."""
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    try:
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        return _MODEL_PATH
    except Exception:
        return None


class FaceDetector:
    """Detect faces; :meth:`detect` never raises on a bad frame."""

    def __init__(self, margin: float = 0.20, min_confidence: float = 0.5) -> None:
        self.margin = margin
        self.min_confidence = min_confidence
        self.backend = "none"
        self._mp_detector = None
        self._haar = None
        model_path = _ensure_model()
        if model_path is not None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                options = vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                    min_detection_confidence=min_confidence,
                    running_mode=vision.RunningMode.IMAGE,
                )
                self._mp_detector = vision.FaceDetector.create_from_options(options)
                self._mp = mp  # kept for mp.Image construction
                self.backend = "mediapipe-tasks"
            except Exception:
                self._mp_detector = None
        if self._mp_detector is None:
            try:
                path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self._haar = cascade
                    self.backend = "haar"
            except Exception:
                self._haar = None

    # -- public API ------------------------------------------------------
    def close(self) -> None:
        """Release the MediaPipe detector (avoids a GC-time TypeError)."""
        if self._mp_detector is not None:
            try:
                self._mp_detector.close()
            except Exception:
                pass
            self._mp_detector = None

    def __del__(self) -> None:  # best effort; close() is the real path
        try:
            self.close()
        except Exception:
            pass

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        """Return face boxes (with margin, clipped to the frame)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        boxes: list[FaceBox] = []
        if self._mp_detector is not None:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                          data=rgb)
                result = self._mp_detector.detect(mp_image)
                for det in result.detections:
                    bb = det.bounding_box
                    score = float(det.categories[0].score) if det.categories else 1.0
                    boxes.append(self._with_margin(
                        FaceBox(int(bb.origin_x), int(bb.origin_y),
                                int(bb.width), int(bb.height), score), w, h))
            except Exception:
                boxes = []  # fall through to Haar on a Tasks failure
        if not boxes and self._haar is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            for (x, y, bw, bh) in self._haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            ):
                boxes.append(self._with_margin(FaceBox(int(x), int(y), int(bw), int(bh), 0.9), w, h))
        return boxes

    def crop(self, frame_bgr: np.ndarray, box: FaceBox) -> np.ndarray:
        """Crop the (margined) face region from a BGR frame."""
        x1, y1, x2, y2 = box.xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
        return frame_bgr[y1:y2, x1:x2].copy()

    # -- internals -------------------------------------------------------
    def _with_margin(self, box: FaceBox, img_w: int, img_h: int) -> FaceBox:
        mx = int(box.w * self.margin)
        my = int(box.h * self.margin)
        x = max(0, box.x - mx)
        y = max(0, box.y - my)
        x2 = min(img_w, box.x + box.w + mx)
        y2 = min(img_h, box.y + box.h + my)
        return FaceBox(x, y, max(1, x2 - x), max(1, y2 - y), box.score)
