"""Adversarial attacks on face embeddings.

Goal (untargeted, dodging): maximise the distance between the embedding of
the cloaked crop and the embedding of the original crop -- equivalently
minimise their cosine similarity -- subject to ``||delta||_inf <= epsilon``.

Methods
-------
* ``fgsm``      -- one-step baseline.
* ``pgd``       -- iterated FGSM with projection onto the L-inf ball.
* ``cloak_attack`` -- PGD + EOT (random JPEG / resize / Gaussian blur) +
  a facial-region mask that concentrates the perturbation on the
  eyes / nose / mouth, where recognisers look, and keeps it light on
  cheeks / forehead / background, where humans notice it.

EOT JPEG is a *differentiable* DCT quantisation (straight-through rounding),
so gradients flow through it. Resize and blur are plain torch ops.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import CloakConfig
from .embedder import BaseEmbedder

# Standard JPEG luminance quantisation table.
_LUMA_Q = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)


def _dct_matrix(n: int = 8) -> torch.Tensor:
    m = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            m[k, i] = math.cos((2 * i + 1) * k * math.pi / (2 * n))
    m[0] *= math.sqrt(1 / n)
    m[1:] *= math.sqrt(2 / n)
    return m


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-through rounding: forward rounds, backward passes gradient."""
    return x + (torch.round(x) - x).detach()


def jpeg_compress(x: torch.Tensor, quality: float) -> torch.Tensor:
    """Differentiable JPEG approximation (luminance DCT quantisation).

    ``x``: (N, 3, H, W) in [0, 1]; H, W are padded to a multiple of 8.
    Chroma is left untouched -- recognisers are far more sensitive to
    luminance, and this keeps the transform cheap enough for realtime use.
    """
    if quality >= 99:
        return x
    # Work in 0-255 space: the quantisation table is defined for 8-bit
    # coefficients. Running it on [0, 1] values would quantise the DC
    # coefficient with a step larger than the signal itself.
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    q = torch.clamp(torch.round((_LUMA_Q * scale + 50) / 100), 1, 255).to(x.device)
    dct = _dct_matrix().to(x.device)
    n, c, h, w = x.shape
    ph, pw = (-h) % 8, (-w) % 8
    xp = F.pad(x * 255.0, (0, pw, 0, ph), mode="reflect")
    nh, nw = xp.shape[2] // 8, xp.shape[3] // 8
    blocks = xp.view(n, c, nh, 8, nw, 8).permute(0, 1, 2, 4, 3, 5)
    coeff = torch.einsum("ij,...jk,lk->...il", dct, blocks, dct)
    coeff = _ste_round(coeff / q) * q
    rec = torch.einsum("ij,...jk,lk->...il", dct.t(), coeff, dct.t())
    out = rec.permute(0, 1, 2, 4, 3, 5).reshape(n, c, nh * 8, nw * 8)
    return (out[:, :, :h, :w] / 255.0).clamp(0, 1)


def gaussian_blur(x: torch.Tensor, kernel: int = 3, sigma: float = 0.8) -> torch.Tensor:
    """Depthwise Gaussian blur (differentiable)."""
    if kernel <= 1:
        return x
    ax = torch.arange(kernel, device=x.device, dtype=torch.float32) - kernel // 2
    k1 = torch.exp(-0.5 * (ax / sigma) ** 2)
    k1 = k1 / k1.sum()
    k2 = torch.outer(k1, k1).expand(x.shape[1], 1, kernel, kernel)
    return F.conv2d(F.pad(x, (kernel // 2,) * 4, mode="reflect"), k2, groups=x.shape[1])


def eot_transform(x: torch.Tensor, rng: torch.Generator, device: torch.device) -> torch.Tensor:
    """Sample one EOT transform: JPEG (q 50-95) *or* resize *or* blur.

    Exactly one transform per sample. Applying all three at once every
    step stacks JPEG + downscale + blur, which low-passes the
    perturbation away faster than 10 PGD steps can rebuild it, and the
    combined gradient is mostly noise. One transform per sample keeps
    the EOT expectation honest while remaining optimisable.
    """
    choice = int(torch.randint(0, 3, (1,), generator=rng).item())
    if choice == 0:
        q = float(torch.randint(50, 96, (1,), generator=rng).item())
        return jpeg_compress(x, q)
    if choice == 1:
        s = 0.7 + 0.3 * float(torch.rand((), generator=rng).item())
        h, w = x.shape[2], x.shape[3]
        small = F.interpolate(x, size=(max(8, int(h * s)), max(8, int(w * s))),
                              mode="bilinear", align_corners=False)
        return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
    return gaussian_blur(x, kernel=3,
                         sigma=0.5 + 0.7 * float(torch.rand((), generator=rng).item())).clamp(0, 1)


def facial_mask(h: int, w: int, floor: float = 0.35, device: str = "cpu") -> torch.Tensor:
    """Weight map (1, 1, H, W): 1.0 on eyes/nose/mouth, ``floor`` elsewhere.

    Regions are relative to the (margined) face crop, which is the standard
    approximation when no landmarks are available. Smoothed so the mask
    itself introduces no visible edge.
    """
    mask = torch.full((1, 1, h, w), floor, dtype=torch.float32)
    # eyes:   y 0.28-0.46, x 0.12-0.88   nose: y 0.44-0.66, x 0.38-0.62
    # mouth:  y 0.66-0.86, x 0.28-0.72
    for (y0, y1, x0, x1) in [(0.28, 0.46, 0.12, 0.88), (0.44, 0.66, 0.38, 0.62),
                             (0.66, 0.86, 0.28, 0.72)]:
        mask[:, :, int(h * y0): int(h * y1), int(w * x0): int(w * x1)] = 1.0
    blurred = gaussian_blur(mask.to(device), kernel=9, sigma=2.0).to(device)
    return blurred.clamp(floor, 1.0)  # blur overshoot must not exceed [floor, 1]


def _loss(orig_emb: torch.Tensor, adv_emb: torch.Tensor) -> torch.Tensor:
    """Cosine similarity to minimise (mean over the batch)."""
    return F.cosine_similarity(orig_emb, adv_emb, dim=1).mean()


def _project(delta: torch.Tensor, eps: float) -> torch.Tensor:
    return delta.clamp(-eps, eps)


def _rand_like(rng: torch.Generator, ref: torch.Tensor) -> torch.Tensor:
    """Uniform [0, 1) tensor shaped like ``ref``.

    ``torch.rand_like(..., generator=...)`` only exists from torch 2.3;
    the real-model env runs torch 2.2, so build it from ``torch.rand``,
    which has accepted ``generator`` for much longer.
    """
    return torch.rand(ref.shape, generator=rng,
                      device=ref.device, dtype=ref.dtype)


def fgsm(embedder: BaseEmbedder, orig: torch.Tensor, eps: float,
         mask: torch.Tensor | None = None, seed: int = 0) -> torch.Tensor:
    """One-step FGSM dodging attack.

    Cosine similarity is *stationary* at the original (its gradient is
    exactly zero there -- 1.0 is a maximum), so a naive one-step FGSM
    from the original does nothing. Like PGD, FGSM therefore starts
    from a small random offset inside the ball and takes its one step
    from there.
    """
    orig = orig.detach()
    rng = torch.Generator(device="cpu").manual_seed(seed)
    start = orig + (_rand_like(rng, orig) * 2 - 1) * (eps / 2)
    with torch.enable_grad():
        x = start.clone().requires_grad_(True)
        loss = _loss(embedder.embed(orig), embedder.embed(x))
        loss.backward()
    # We MINIMISE cosine similarity, so we step *against* the gradient
    # (gradient descent on the similarity). Stepping with the gradient
    # would climb back toward the original embedding.
    delta = -eps * x.grad.sign()
    if mask is not None:
        delta = delta * mask
    return (orig + _project(delta, eps)).clamp(0, 1).detach()


def pgd(embedder: BaseEmbedder, orig: torch.Tensor, eps: float, steps: int = 10,
        alpha: float | None = None, mask: torch.Tensor | None = None,
        use_eot: bool = False, eot_samples: int = 1,
        seed: int = 0) -> torch.Tensor:
    """Iterated PGD dodging attack, optionally with EOT."""
    alpha = alpha if alpha and alpha > 0 else eps / max(1, steps)
    orig = orig.detach()
    rng = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        orig_emb = embedder.embed(orig)
    # Random start inside the ball (standard PGD), then iterate.
    delta = (_rand_like(rng, orig) * 2 - 1) * eps
    delta = _project(delta, eps)
    for _ in range(steps):
        adv = (orig + delta).clamp(0, 1)
        adv.requires_grad_(True)
        with torch.enable_grad():
            total = torch.zeros((), device=orig.device)
            for _s in range(max(1, eot_samples)):
                view = eot_transform(adv, rng, orig.device) if use_eot else adv
                total = total + _loss(orig_emb, embedder.embed(view))
            (total / max(1, eot_samples)).backward()
        with torch.no_grad():
            step = -alpha * adv.grad.sign()  # descent on cosine similarity
            delta = _project(delta + step, eps)
    if mask is not None:
        # Apply the facial mask ONCE, to the finished delta. Masking
        # every iteration compounds (0.6 * (0.6 * ...)) and shrinks the
        # off-mask budget to ~0.35*eps, which guts the attack for
        # little visual gain. One application gives exactly
        # floor*eps off-mask and full eps on eyes/nose/mouth.
        delta = _project(delta * mask, eps)
    return (orig + delta).clamp(0, 1).detach()


def cloak_attack(embedder: BaseEmbedder, orig: torch.Tensor,
                 cfg: CloakConfig, seed: int = 0) -> torch.Tensor:
    """The full Cloak attack: PGD + EOT + facial mask, per the brief."""
    mask = None
    if cfg.use_mask:
        mask = facial_mask(orig.shape[2], orig.shape[3], cfg.mask_floor, str(orig.device))
    return pgd(embedder, orig, eps=cfg.epsilon, steps=cfg.steps,
               alpha=cfg.resolved_alpha(), mask=mask, use_eot=cfg.use_eot,
               eot_samples=cfg.eot_samples, seed=seed)
