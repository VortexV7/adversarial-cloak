import numpy as np
import torch

from backend.config import CloakConfig
from backend.detector import FaceBox
from backend.temporal import PerturbationCache, Tracker


def test_tracker_assigns_stable_ids():
    tr = Tracker(iou_threshold=0.5)
    a = tr.update([FaceBox(10, 10, 100, 100, 0.9)])[0]
    b = tr.update([FaceBox(12, 12, 100, 100, 0.9)])[0]
    assert a.track_id == b.track_id
    c = tr.update([FaceBox(300, 300, 100, 100, 0.9)])[0]
    assert c.track_id != a.track_id


def test_cache_recompute_policy():
    cfg = CloakConfig(update_interval=5, change_threshold=0.08)
    cache = PerturbationCache(cfg)
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    box = FaceBox(10, 10, 100, 100, 0.9, track_id=0)
    assert cache.should_recompute(0, box, crop, frame_idx=0)
    delta = torch.zeros(1, 3, 160, 160)
    cache.put(0, delta, box, crop, frame_idx=0)
    # Same crop, 1 frame later: reuse.
    assert not cache.should_recompute(0, box, crop, frame_idx=1)
    # 5 frames later: recompute.
    assert cache.should_recompute(0, box, crop, frame_idx=5)
    # Face moved far: recompute even before the interval.
    moved = FaceBox(300, 300, 100, 100, 0.9, track_id=0)
    assert cache.should_recompute(0, moved, crop, frame_idx=1)
    # Crop content changed a lot: recompute.
    changed = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert cache.should_recompute(0, box, changed, frame_idx=1)


def test_ema_smoothing_moves_toward_new_delta():
    cfg = CloakConfig(ema_alpha=0.7)
    cache = PerturbationCache(cfg)
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    box = FaceBox(10, 10, 100, 100, 0.9, track_id=0)
    cache.put(0, torch.zeros(1, 3, 160, 160), box, crop, 0)
    out = cache.put(0, torch.ones(1, 3, 160, 160), box, crop, 1)
    assert abs(float(out.mean()) - 0.7) < 1e-5
