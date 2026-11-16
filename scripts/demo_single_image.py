"""Demo script (Step 1): cloak a single image and save original/protected.

Usage:
    python scripts/demo_single_image.py --input path/to/face.jpg --output out/
    python scripts/demo_single_image.py            # uses a synthetic face

Prints similarity before/after (FaceNet + ArcFace), SSIM/PSNR/LPIPS and
latency, and writes ``original.png`` / ``protected.png`` / ``side_by_side.png``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import CloakConfig  # noqa: E402
from backend.pipeline import CloakPipeline  # noqa: E402


def synthetic_face() -> np.ndarray:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default="demo_out")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=8 / 255)
    args = ap.parse_args()

    if args.input:
        frame = cv2.imread(args.input)
        if frame is None:
            raise SystemExit(f"could not read {args.input}")
    else:
        frame = synthetic_face()
        print("no --input given: using a synthetic face (detector may find 0 faces)")

    pipe = CloakPipeline(CloakConfig(steps=args.steps, epsilon=args.epsilon))
    t0 = time.perf_counter()
    result = pipe.process_frame(frame)
    print(f"latency: {(time.perf_counter() - t0) * 1000:.1f} ms")
    print(f"faces: {result.metrics['faces']}  detector: {pipe.detector.backend}")
    print(f"similarity: {result.metrics['similarity_before']} -> {result.metrics['similarity_after']}")
    print(f"confidence: {result.metrics['confidence_before']} -> {result.metrics['confidence_after']}")
    print(f"SSIM {result.metrics['ssim']}  PSNR {result.metrics['psnr']}  LPIPS {result.metrics['lpips']}")
    print(f"attack success: {result.metrics['attack_success']}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "original.png"), frame)
    cv2.imwrite(str(out / "protected.png"), result.protected_bgr)
    cv2.imwrite(str(out / "side_by_side.png"),
                np.hstack([frame, result.protected_bgr]))
    print(f"wrote {out}/original.png, protected.png, side_by_side.png")


if __name__ == "__main__":
    main()
