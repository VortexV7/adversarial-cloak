"""Evaluation metrics: recognition, image quality, and latency.

Recognition
-----------
* ``cosine_similarity`` -- between two (already normalised) embeddings.
* ``match_confidence`` -- maps similarity to a 0..1 confidence with a
  logistic centred on the operating threshold (default 0.60 for FaceNet /
  VGGFace2; ArcFace typically uses ~0.30-0.40 -- pass the threshold in).
* ``attack_success_rate`` -- fraction of originally-matching pairs whose
  similarity falls *below* the threshold after cloaking.

Image quality -- ``ssim``, ``psnr``, ``lpips_score`` (LPIPS if the
``lpips`` package is installed, otherwise a calibrated MSE proxy, flagged
by ``lpips_is_proxy`` so reports never mislabel it).

Latency -- :class:`LatencyTracker` keeps a rolling window of per-frame
times and reports mean latency and FPS.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim
except Exception:  # pragma: no cover
    _psnr = _ssim = None  # type: ignore[assignment]

_LPIPS_MODEL = None
_LPIPS_TRIED = False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def match_confidence(similarity: float, threshold: float = 0.60, slope: float = 18.0) -> float:
    """Logistic map: sim == threshold -> 0.5, well above -> ~1, well below -> ~0."""
    import math

    return float(1.0 / (1.0 + math.exp(-slope * (similarity - threshold))))


def attack_success_rate(orig_sims: list[float] | np.ndarray,
                        adv_sims: list[float] | np.ndarray,
                        threshold: float = 0.60) -> float:
    """Fraction of pairs that matched before and no longer match after."""
    orig = np.asarray(orig_sims, dtype=float)
    adv = np.asarray(adv_sims, dtype=float)
    matched = orig >= threshold
    if matched.sum() == 0:
        return 0.0
    return float(((adv < threshold) & matched).sum() / matched.sum())


def rank1_drop(orig_rank1: float, adv_rank1: float) -> float:
    return float(orig_rank1 - adv_rank1)


def ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """SSIM in [0, 1]; 1.0 = identical. Works on uint8 HWC images."""
    if _ssim is None:
        return _ssim_fallback(img_a, img_b)
    a, b = _to_gray_float(img_a), _to_gray_float(img_b)
    return float(_ssim(a, b, data_range=1.0))


def psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    if _psnr is None:
        mse = float(np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2))
        return float("inf") if mse == 0 else float(20 * np.log10(255.0) - 10 * np.log10(mse))
    return float(_psnr(img_a, img_b, data_range=255))


def lpips_score(img_a: np.ndarray, img_b: np.ndarray) -> tuple[float, bool]:
    """Return (score, is_proxy). Lower is better; identical images score 0."""
    global _LPIPS_MODEL, _LPIPS_TRIED
    if not _LPIPS_TRIED:
        _LPIPS_TRIED = True
        try:
            import lpips
            import torch

            _LPIPS_MODEL = lpips.LPIPS(net="alex")
            _LPIPS_MODEL.eval()
            if torch.cuda.is_available():
                _LPIPS_MODEL.cuda()
        except Exception:
            _LPIPS_MODEL = None
    if _LPIPS_MODEL is not None:
        import torch

        def _to_t(x: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            return (t * 2 - 1).to(next(_LPIPS_MODEL.parameters()).device)  # type: ignore[union-attr]
        with torch.no_grad():
            return float(_LPIPS_MODEL(_to_t(img_a), _to_t(img_b)).item()), False
    # Proxy: normalised MSE, scaled to roughly LPIPS magnitude for eps=8/255.
    mse = float(np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2))
    return float(min(1.0, mse / (255.0 ** 2) * 18.0)), True


def _to_gray_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]) / 255.0
    return img.astype(np.float64) / 255.0


def _ssim_fallback(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _to_gray_float(a), _to_gray_float(b)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((2 * mx * my + c1) * (2 * cov + c2) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))


class LatencyTracker:
    """Rolling mean latency / FPS over the last ``window`` frames."""

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self._latencies: deque[float] = deque(maxlen=window)
        self._last_tick: float | None = None

    def record(self, latency_ms: float) -> None:
        self._latencies.append(float(latency_ms))

    def tick(self) -> float:
        """Call once per processed frame; returns instantaneous FPS."""
        now = time.perf_counter()
        fps = 0.0
        if self._last_tick is not None:
            dt = now - self._last_tick
            fps = 1.0 / dt if dt > 0 else 0.0
        self._last_tick = now
        return fps

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self._latencies)) if self._latencies else 0.0

    @property
    def fps(self) -> float:
        if not self._latencies:
            return 0.0
        return 1000.0 / self.mean_latency_ms if self.mean_latency_ms > 0 else 0.0

    def snapshot(self) -> dict[str, float]:
        return {"latency_ms": round(self.mean_latency_ms, 1), "fps": round(self.fps, 1)}
