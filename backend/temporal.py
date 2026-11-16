"""Temporal smoothing: cache, EMA, and re-optimisation triggers.

Recomputing a PGD perturbation every frame is too slow for realtime and
produces visible flicker (each optimisation lands in a different corner of
the L-inf ball). Instead:

* compute the perturbation every ``update_interval`` frames (default 5);
* reuse the cached perturbation between recomputations;
* smooth each new perturbation into the cached one with an EMA, so the
  on-screen noise drifts instead of jumping;
* recompute early when the face moves (IoU drop) or the crop content
  changes (mean absolute difference above a threshold) -- e.g. a head turn.

Faces are tracked across frames by greedy IoU assignment in :class:`Tracker`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .config import CloakConfig
from .detector import FaceBox


@dataclass
class CachedPerturbation:
    delta: torch.Tensor                 # (1, 3, H, W), in [-eps, eps]
    bbox: FaceBox
    last_frame: int
    last_crop: np.ndarray               # BGR crop, for change detection
    ema_delta: torch.Tensor | None = None

    def smoothed(self, new_delta: torch.Tensor, ema_alpha: float) -> torch.Tensor:
        """EMA-smooth ``new_delta`` into the cache; return the smoothed delta."""
        base = self.ema_delta if self.ema_delta is not None else self.delta
        if base.shape != new_delta.shape:
            base = torch.nn.functional.interpolate(
                base, size=new_delta.shape[2:], mode="bilinear", align_corners=False
            )
        smoothed = ema_alpha * new_delta + (1 - ema_alpha) * base
        self.ema_delta = smoothed.detach().clone()
        self.delta = smoothed.detach().clone()
        return smoothed.detach().clone()


class Tracker:
    """Greedy IoU tracker: assigns stable integer ids to face boxes."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self._next_id = 0
        self._tracks: dict[int, FaceBox] = {}

    def update(self, boxes: list[FaceBox]) -> list[FaceBox]:
        assigned: dict[int, FaceBox] = {}
        used_ids: set[int] = set()
        for box in sorted(boxes, key=lambda b: -b.score):
            best_id, best_iou = -1, self.iou_threshold
            for tid, prev in self._tracks.items():
                if tid in used_ids:
                    continue
                iou = box.iou(prev)
                if iou > best_iou:
                    best_id, best_iou = tid, iou
            if best_id < 0:
                best_id = self._next_id
                self._next_id += 1
            used_ids.add(best_id)
            box.track_id = best_id
            assigned[best_id] = box
        self._tracks = assigned
        return boxes


class PerturbationCache:
    """Per-track perturbation cache with the recompute policy."""

    def __init__(self, cfg: CloakConfig) -> None:
        self.cfg = cfg
        self._cache: dict[int, CachedPerturbation] = {}

    def should_recompute(self, track_id: int, bbox: FaceBox, crop: np.ndarray,
                         frame_idx: int) -> bool:
        entry = self._cache.get(track_id)
        if entry is None:
            return True
        if frame_idx - entry.last_frame >= self.cfg.update_interval:
            return True
        if bbox.iou(entry.bbox) < self.cfg.iou_threshold:
            return True
        if self._crop_changed(entry.last_crop, crop):
            return True
        return False

    def get(self, track_id: int) -> CachedPerturbation | None:
        return self._cache.get(track_id)

    def put(self, track_id: int, delta: torch.Tensor, bbox: FaceBox,
            crop: np.ndarray, frame_idx: int) -> torch.Tensor:
        """Store a fresh perturbation (EMA-smoothed against the old one)."""
        entry = self._cache.get(track_id)
        if entry is not None:
            delta = entry.smoothed(delta, self.cfg.ema_alpha)
        self._cache[track_id] = CachedPerturbation(
            delta=delta.detach().clone(), bbox=bbox, last_frame=frame_idx,
            last_crop=crop.copy(),
            ema_delta=delta.detach().clone(),
        )
        return delta

    def touch(self, track_id: int, bbox: FaceBox, frame_idx: int) -> None:
        if track_id in self._cache:
            self._cache[track_id].bbox = bbox
            self._cache[track_id].last_frame = frame_idx

    def prune(self, live_ids: set[int]) -> None:
        for tid in list(self._cache):
            if tid not in live_ids:
                del self._cache[tid]

    # -- internals -------------------------------------------------------
    def _crop_changed(self, old: np.ndarray, new: np.ndarray) -> bool:
        if old.shape != new.shape:
            return True
        import cv2

        a = cv2.resize(old, (64, 64)).astype(np.float32) / 255.0
        b = cv2.resize(new, (64, 64)).astype(np.float32) / 255.0
        return float(np.mean(np.abs(a - b))) > self.cfg.change_threshold
