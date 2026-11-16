"""End-to-end pipeline: capture -> detect -> attack -> composite -> metrics.

One :class:`CloakPipeline` owns the detector, both embedders, the temporal
cache and the latency tracker. ``process_frame`` is the realtime path used
by the WebSocket handler; ``process_image`` / ``process_video`` are the
offline paths used by the demo script and the evaluation harness.

Compositing: the cloaked crop is blended back into the frame with a
feathered (Gaussian-blurred) alpha mask, so there is no visible rectangle
edge around the face.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .attack import cloak_attack, fgsm, pgd
from .config import CloakConfig
from .detector import FaceBox, FaceDetector
from .embedder import BaseEmbedder, get_embedder
from .metrics import LatencyTracker, cosine_similarity, lpips_score, match_confidence, psnr, ssim
from .temporal import PerturbationCache, Tracker


@dataclass
class FrameResult:
    protected_bgr: np.ndarray
    boxes: list[FaceBox] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    frame_idx: int = 0


class CloakPipeline:
    def __init__(self, cfg: CloakConfig | None = None,
                 surrogate: BaseEmbedder | None = None,
                 evaluator: BaseEmbedder | None = None,
                 detector: FaceDetector | None = None) -> None:
        self.cfg = cfg or CloakConfig()
        self.device = self.cfg.resolved_device()
        self.surrogate = surrogate or get_embedder(self.cfg.surrogate, self.device)
        self.evaluator = evaluator or get_embedder(self.cfg.evaluator, self.device)
        self.detector = detector or FaceDetector()
        self.tracker = Tracker(self.cfg.iou_threshold)
        self.cache = PerturbationCache(self.cfg)
        self.latency = LatencyTracker()
        self.enrolled: np.ndarray | None = None  # reference embedding (if enrolled)
        self.frame_idx = 0
        self._closed = False
        self._sim_history: list[float] = []
        # Per-track cached similarities: the evaluator forward is the most
        # expensive per-frame op, so when the perturbation is reused (the
        # common case) we reuse the last measured similarities too.
        self._sim_cache: dict[int, tuple[float, float]] = {}
        self._last_metrics: dict = {}

    def close(self) -> None:
        """Release the detector (call on server shutdown)."""
        if not self._closed:
            self._closed = True
            self.detector.close()

    def __del__(self) -> None:  # best effort; close() is the real path
        try:
            self.close()
        except Exception:
            pass

    # -- realtime path ---------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        start = time.perf_counter()
        self.frame_idx += 1
        frame = self._resize(frame_bgr)
        protected = frame.copy()
        boxes = self.detector.detect(frame)
        boxes = self.tracker.update(boxes)
        sims_before, sims_after = [], []
        ssim_vals, psnr_vals, lpips_vals = [], [], []

        if self.cfg.enabled:
            for box in boxes:
                crop = self.detector.crop(frame, box)
                if crop.size == 0 or min(crop.shape[:2]) < 32:
                    continue
                orig_t = self._crop_to_tensor(crop)
                ref = self._reference_embedding(orig_t)
                recompute = self.cache.should_recompute(
                    box.track_id, box, crop, self.frame_idx) \
                    if self.cache.get(box.track_id) is not None else True
                delta = self._delta_for(box, crop, orig_t)
                adv_t = (orig_t + delta).clamp(0, 1)
                adv_crop = self._tensor_to_crop(adv_t, crop.shape)
                if recompute or box.track_id not in self._sim_cache:
                    import torch.nn.functional as _F

                    with torch.no_grad():
                        # Both similarities in the evaluator's space.
                        orig_eval = _F.interpolate(
                            orig_t, size=(self.evaluator.image_size,
                                         self.evaluator.image_size),
                            mode="bilinear", align_corners=False)
                        sim_before = cosine_similarity(
                            ref, self.evaluator.embed(orig_eval).cpu().numpy()[0])
                        sim_after = cosine_similarity(
                            ref, self.evaluator.embed(
                                self.evaluator.preprocess(
                                    cv2.cvtColor(adv_crop, cv2.COLOR_BGR2RGB))).cpu().numpy()[0])
                    self._sim_cache[box.track_id] = (sim_before, sim_after)
                else:
                    sim_before, sim_after = self._sim_cache[box.track_id]
                sims_before.append(sim_before)
                sims_after.append(sim_after)
                ssim_vals.append(ssim(crop, adv_crop))
                psnr_vals.append(psnr(crop, adv_crop))
                lpips_vals.append(lpips_score(crop, adv_crop)[0])
                protected = self._composite(protected, adv_crop, box)

        self.cache.prune({b.track_id for b in boxes})
        for tid in list(self._sim_cache):
            if tid not in {b.track_id for b in boxes}:
                del self._sim_cache[tid]
        latency_ms = (time.perf_counter() - start) * 1000
        self.latency.record(latency_ms)
        metrics = self._summarise(sims_before, sims_after, ssim_vals, psnr_vals,
                                  lpips_vals, latency_ms, len(boxes))
        self._last_metrics = metrics
        return FrameResult(protected_bgr=protected, boxes=boxes,
                           metrics=metrics, frame_idx=self.frame_idx)

    # -- offline paths ---------------------------------------------------
    def process_image(self, image_bgr: np.ndarray) -> FrameResult:
        return self.process_frame(image_bgr)

    def process_video(self, source: str | int, max_frames: int = 0,
                      on_frame=None):
        """Yield FrameResults from a file / webcam index / RTSP URL."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"could not open video source: {source!r}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = self.process_frame(frame)
                if on_frame is not None:
                    on_frame(result)
                yield result
                if max_frames and self.frame_idx >= max_frames:
                    break
        finally:
            cap.release()

    # -- internals -------------------------------------------------------
    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w > self.cfg.max_frame_width:
            scale = self.cfg.max_frame_width / w
            return cv2.resize(frame, (self.cfg.max_frame_width, int(h * scale)))
        return frame

    def _crop_to_tensor(self, crop_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return self.surrogate.preprocess(rgb)

    def _tensor_to_crop(self, tensor: torch.Tensor, shape) -> np.ndarray:
        import torch.nn.functional as F

        h, w = shape[:2]
        t = F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
        arr = (t[0].detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _reference_embedding(self, orig_t: torch.Tensor) -> np.ndarray:
        """Reference for similarity: enrolled embedding if present, else the
        original crop's own embedding (dodging = move away from yourself).

        Always in the *evaluator's* embedding space: sim_before and
        sim_after are both computed with the evaluator, so the reported
        drop is the attack's effect, not a difference between two models.
        (Comparing a surrogate embedding with an evaluator embedding
        would give ~0 even for an uncloaked frame.)"""
        if self.enrolled is not None:
            return self.enrolled
        import torch.nn.functional as F

        t = F.interpolate(orig_t, size=(self.evaluator.image_size,
                                        self.evaluator.image_size),
                          mode="bilinear", align_corners=False)
        with torch.no_grad():
            return self.evaluator.embed(t).cpu().numpy()[0]

    def _delta_for(self, box: FaceBox, crop: np.ndarray, orig_t: torch.Tensor) -> torch.Tensor:
        """Cached / EMA-smoothed perturbation, recomputed only when due."""
        cached = self.cache.get(box.track_id)
        if cached is not None and not self.cache.should_recompute(
                box.track_id, box, crop, self.frame_idx):
            self.cache.touch(box.track_id, box, self.frame_idx)
            return cached.delta.to(orig_t.device)
        adv = cloak_attack(self.surrogate, orig_t, self.cfg, seed=self.frame_idx)
        delta = (adv - orig_t).detach()
        return self.cache.put(box.track_id, delta, box, crop, self.frame_idx).to(orig_t.device)

    @staticmethod
    def _composite(frame: np.ndarray, adv_crop: np.ndarray, box: FaceBox) -> np.ndarray:
        """Feathered blend of the cloaked crop back into the frame."""
        x1, y1, x2, y2 = box.xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
            return frame
        adv = cv2.resize(adv_crop, (w, h))
        alpha = np.ones((h, w), dtype=np.float32)
        feather = max(3, min(h, w) // 12 | 1)
        alpha = cv2.GaussianBlur(alpha, (feather, feather), 0)[..., None]
        out = frame.astype(np.float32)
        out[y1:y2, x1:x2] = adv.astype(np.float32) * alpha + out[y1:y2, x1:x2] * (1 - alpha)
        return out.astype(np.uint8)

    def _summarise(self, before, after, ssim_v, psnr_v, lpips_v, latency_ms, n_faces):
        def _mean(xs, default=0.0):
            return float(np.mean(xs)) if xs else default

        sim_b, sim_a = _mean(before, 1.0), _mean(after, 1.0)
        self._sim_history.append(sim_a)
        return {
            "faces": n_faces,
            "similarity_before": round(sim_b, 4),
            "similarity_after": round(sim_a, 4),
            "confidence_before": round(match_confidence(sim_b, self.cfg.enroll_threshold), 4),
            "confidence_after": round(match_confidence(sim_a, self.cfg.enroll_threshold), 4),
            "attack_success": bool(n_faces and sim_b >= self.cfg.enroll_threshold
                                   and sim_a < self.cfg.enroll_threshold),
            "ssim": round(_mean(ssim_v, 1.0), 4),
            "psnr": round(_mean(psnr_v, 99.0), 2),
            "lpips": round(_mean(lpips_v, 0.0), 4),
            "latency_ms": round(latency_ms, 1),
            "fps": round(self.latency.fps, 1),
            "similarity_history": [round(float(v), 3) for v in self._sim_history[-120:]],
            "enabled": self.cfg.enabled,
            "epsilon": round(self.cfg.epsilon, 4),
            "steps": self.cfg.steps,
            "update_interval": self.cfg.update_interval,
            "surrogate": self.surrogate.name,
            "evaluator": self.evaluator.name,
            "surrogate_fallback": self.surrogate.is_fallback,
            "evaluator_fallback": self.evaluator.is_fallback,
        }


# Convenience wrappers used by the evaluation script (single-image attacks).
def attack_crop(embedder: BaseEmbedder, crop_rgb: np.ndarray, method: str,
                cfg: CloakConfig) -> np.ndarray:
    """Attack one RGB crop; returns the cloaked RGB crop (uint8)."""
    import torch.nn.functional as F

    orig = embedder.preprocess(crop_rgb)
    if method == "fgsm":
        adv = fgsm(embedder, orig, cfg.epsilon)
    elif method == "pgd":
        adv = pgd(embedder, orig, cfg.epsilon, cfg.steps, cfg.resolved_alpha())
    elif method == "ours":
        adv = cloak_attack(embedder, orig, cfg)
    elif method == "ours_no_eot":
        cfg2 = CloakConfig(**{**cfg.__dict__, "use_eot": False})
        adv = cloak_attack(embedder, orig, cfg2)
    else:
        raise ValueError(f"unknown method {method!r}")
    h, w = crop_rgb.shape[:2]
    t = F.interpolate(adv, size=(h, w), mode="bilinear", align_corners=False)
    return (t[0].detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
