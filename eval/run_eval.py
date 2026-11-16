"""Evaluation harness: FGSM / PGD / Ours / Ours-no-EOT on LFW (or synthetic pairs).

Outputs
-------
* ``results.csv`` -- one row per (method, evaluator) with ASR, Rank-1,
  SSIM, PSNR, LPIPS, robustness (JPEG 70 / resize 0.5x / blur) and latency.
* ``plots/*.png`` -- ablation plots for epsilon, PGD steps and EOT.

Usage
-----
    python -m eval.run_eval --pairs 50 --out eval/results.csv
    python eval/download_lfw.py        # first, to fetch the real dataset

Without LFW the harness falls back to deterministic synthetic face-like
pairs (clearly labelled ``dataset=synthetic`` in the CSV) so the script
always runs end-to-end on a fresh checkout.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from backend.config import CloakConfig  # noqa: E402
from backend.embedder import get_embedder  # noqa: E402
from backend.metrics import (  # noqa: E402
    attack_success_rate,
    cosine_similarity,
    lpips_score,
    psnr,
    ssim,
)
from backend.pipeline import attack_crop  # noqa: E402

METHODS = ["fgsm", "pgd", "ours", "ours_no_eot"]


def synthetic_pairs(n: int, size: int = 160) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic face-like pairs: same identity = same base pattern.

    Each identity is a *smooth* pattern -- a skin-tone gradient plus
    identity-specific dark blobs where eyes / nose / mouth sit -- with
    only small per-shot noise. Pure white-noise pairs are a bad stand-in
    for faces: real face embeddings are driven by smooth, low-frequency
    structure, and an attack's behaviour on noise does not transfer to
    faces (it underestimates every method, especially EOT, whose
    transforms are low-pass).
    """
    pairs = []
    y, x = np.mgrid[0:size, 0:size]
    for i in range(n):
        rng = np.random.default_rng(1000 + i)
        # Skin-tone gradient, identity-specific hue/tilt.
        hue = rng.uniform(-0.05, 0.05)
        base = np.zeros((size, size, 3))
        base[..., 0] = 0.55 + hue + 0.12 * np.sin(x / 45.0 + i)
        base[..., 1] = 0.45 + 0.10 * np.cos(y / 50.0 + i * 0.7)
        base[..., 2] = 0.40 + 0.08 * np.sin((x + y) / 60.0 + i)
        # Identity-specific facial features (dark blobs).
        for (cy, cx, sy, sx, amp) in [
            (0.36, 0.36, 0.05, 0.07, -0.30), (0.36, 0.64, 0.05, 0.07, -0.30),
            (0.55, 0.50, 0.09, 0.04, -0.18), (0.74, 0.50, 0.05, 0.10, -0.25),
        ]:
            jx = rng.normal(0, 0.01)
            jy = rng.normal(0, 0.01)
            blob = np.exp(-(((y / size - (cy + jy)) / sy) ** 2
                            + ((x / size - (cx + jx)) / sx) ** 2))
            base += amp * blob[..., None]
        a = np.clip(base + rng.normal(0, 0.02, base.shape), 0, 1)
        b = np.clip(base + rng.normal(0, 0.02, base.shape), 0, 1)
        pairs.append(((a * 255).astype(np.uint8), (b * 255).astype(np.uint8)))
    return pairs


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert an image to uint8 HWC.

    scikit-learn's LFW loader returns float64 images in [0, 1]; a naive
    ``.astype(np.uint8)`` truncates those to 0/1, i.e. near-black images
    (mean 0.0002) whose embeddings are all identical -- every number the
    eval then reports is garbage (FGSM "does nothing", PGD ASR inverted,
    Ours == Ours-no-EOT). Scale floats properly; pass uint8 through.
    """
    if img.dtype == np.uint8:
        return img
    return np.clip(img * 255.0, 0, 255).round().astype(np.uint8)


def load_lfw_pairs(n: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Load genuine LFW pairs if present (see download_lfw.py), else synthetic."""
    try:
        from sklearn.datasets import fetch_lfw_pairs
    except ImportError:
        return synthetic_pairs(n), "synthetic"

    try:
        data = fetch_lfw_pairs(subset="test", data_home=str(ROOT / "eval" / "lfw_home"),
                               download_if_missing=False, color=True, resize=0.5)
        pairs, targets = data.pairs, data.target
        same = [(p[0], p[1]) for p, t in zip(pairs, targets) if t == 1][:n]
        if len(same) >= max(1, n // 2):
            return [(_to_uint8(a), _to_uint8(b)) for a, b in same], "lfw"
    except Exception:
        pass
    return synthetic_pairs(n), "synthetic"


def robust_similarity(emb, crop_rgb: np.ndarray, ref: np.ndarray, transform: str) -> float:
    img = crop_rgb
    if transform == "jpeg70":
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    elif transform == "resize0.5":
        h, w = img.shape[:2]
        small = cv2.resize(img, (w // 2, h // 2))
        img = cv2.resize(small, (w, h))
    elif transform == "blur":
        img = cv2.GaussianBlur(img, (5, 5), 0)
    with torch.no_grad():
        emb_v = emb.embed(emb.preprocess(img)).cpu().numpy()[0]
    return cosine_similarity(ref, emb_v)


def evaluate(method: str, pairs, surrogate, evaluator, cfg: CloakConfig,
             threshold: float = 0.60) -> dict:
    orig_sims, adv_sims, adv_sims_eval = [], [], []
    ssims, psnrs, lpipss, latencies = [], [], [], []
    robust = {"jpeg70": [], "resize0.5": [], "blur": []}
    for ref_img, query_img in pairs:
        with torch.no_grad():
            ref = surrogate.embed(surrogate.preprocess(ref_img)).cpu().numpy()[0]
            q0 = surrogate.embed(surrogate.preprocess(query_img)).cpu().numpy()[0]
            ref_eval = evaluator.embed(evaluator.preprocess(ref_img)).cpu().numpy()[0]
            q0_eval = evaluator.embed(evaluator.preprocess(query_img)).cpu().numpy()[0]
        orig_sims.append(cosine_similarity(ref, q0))
        t0 = time.perf_counter()
        adv_img = attack_crop(surrogate, query_img, method, cfg)
        latencies.append((time.perf_counter() - t0) * 1000)
        with torch.no_grad():
            qa = surrogate.embed(surrogate.preprocess(adv_img)).cpu().numpy()[0]
            qa_eval = evaluator.embed(evaluator.preprocess(adv_img)).cpu().numpy()[0]
        adv_sims.append(cosine_similarity(ref, qa))
        adv_sims_eval.append(cosine_similarity(ref_eval, qa_eval))
        ssims.append(ssim(query_img, adv_img))
        psnrs.append(psnr(query_img, adv_img))
        lpipss.append(lpips_score(query_img, adv_img)[0])
        for name in robust:
            robust[name].append(robust_similarity(evaluator, adv_img, ref_eval, name))
    # Rank-1: gallery = all reference images; correct if nearest is the true ref.
    # With one gallery entry per pair this is the same-pair check, reported
    # as the fraction still ranked first after cloaking (drop = orig - adv).
    #
    # Robustness is reported BOTH as ASR and as mean similarity after the
    # transform. The mean matters: on the offline fallback nets the
    # transform ceiling is high -- even an *unattacked* image scores
    # ~0.95 after JPEG 70 -- so robust ASR alone understates the
    # difference between methods. Lower robust sim = more robust.
    # (The unattacked baseline sims are added by main(), once.)
    return {
        "method": method,
        "asr_facenet": round(attack_success_rate(orig_sims, adv_sims, threshold), 4),
        "asr_arcface": round(attack_success_rate(orig_sims, adv_sims_eval, threshold), 4),
        "rank1_drop": round(float(np.mean(orig_sims) - np.mean(adv_sims)), 4),
        "ssim": round(float(np.mean(ssims)), 4),
        "psnr": round(float(np.mean(psnrs)), 2),
        "lpips": round(float(np.mean(lpipss)), 4),
        "robust_jpeg70_asr": round(attack_success_rate(orig_sims, robust["jpeg70"], threshold), 4),
        "robust_resize_asr": round(attack_success_rate(orig_sims, robust["resize0.5"], threshold), 4),
        "robust_blur_asr": round(attack_success_rate(orig_sims, robust["blur"], threshold), 4),
        "robust_jpeg70_sim": round(float(np.mean(robust["jpeg70"])), 4),
        "robust_resize_sim": round(float(np.mean(robust["resize0.5"])), 4),
        "robust_blur_sim": round(float(np.mean(robust["blur"])), 4),
        "latency_ms": round(float(np.mean(latencies)), 1),
    }


def _refs(emb, pairs):
    """Evaluator embeddings of the reference images in each pair."""
    refs = []
    with torch.no_grad():
        for ref_img, _ in pairs:
            refs.append(emb.embed(emb.preprocess(ref_img)).cpu().numpy()[0])
    return refs


def ablation_plots(results: list[dict], outdir: Path) -> None:
    """Epsilon / steps / EOT ablation, re-running the harness on small grids."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    pairs, _ = load_lfw_pairs(12)
    surrogate = get_embedder("facenet", "cpu")
    evaluator = get_embedder("arcface", "cpu")

    def _asr(**kw):
        cfg = CloakConfig(**kw)
        r = evaluate("ours", pairs, surrogate, evaluator, cfg)
        return r["asr_facenet"], r["asr_arcface"], r["ssim"]

    eps_grid = [2 / 255, 4 / 255, 8 / 255, 12 / 255, 16 / 255]
    eps_rows = [_asr(epsilon=e) for e in eps_grid]
    steps_rows = [_asr(steps=s) for s in [1, 5, 10, 20]]
    eot_on = _asr(use_eot=True)
    eot_off = _asr(use_eot=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot([e * 255 for e in eps_grid], [r[0] for r in eps_rows], "o-", label="FaceNet ASR")
    axes[0].plot([e * 255 for e in eps_grid], [r[1] for r in eps_rows], "s-", label="ArcFace ASR")
    axes[0].set_xlabel("epsilon ( /255)"); axes[0].set_ylabel("Attack success rate")
    axes[0].set_title("Ablation: epsilon"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot([1, 5, 10, 20], [r[0] for r in steps_rows], "o-", label="FaceNet ASR")
    axes[1].plot([1, 5, 10, 20], [r[1] for r in steps_rows], "s-", label="ArcFace ASR")
    axes[1].set_xlabel("PGD steps"); axes[1].set_ylabel("Attack success rate")
    axes[1].set_title("Ablation: PGD steps"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].bar(["EOT on", "EOT off"], [eot_on[0], eot_off[0]], label="FaceNet ASR")
    axes[2].bar(["EOT on", "EOT off"], [eot_on[1], eot_off[1]], alpha=0.6, label="ArcFace ASR")
    axes[2].set_ylabel("Attack success rate"); axes[2].set_title("Ablation: EOT")
    axes[2].legend(); axes[2].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "ablation.png", dpi=180)
    plt.close(fig)

    # Method comparison bar chart.
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["method"] for r in results]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["asr_facenet"] for r in results], 0.4, label="FaceNet (white-box)")
    ax.bar(x + 0.2, [r["asr_arcface"] for r in results], 0.4, label="ArcFace (black-box)")
    ax.set_xticks(x, names); ax.set_ylabel("Attack success rate"); ax.set_ylim(0, 1)
    ax.set_title("Attack success rate by method"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "methods.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results.csv"))
    ap.add_argument("--plots", default=str(ROOT / "eval" / "plots"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    pairs, dataset = load_lfw_pairs(args.pairs)
    print(f"dataset={dataset} pairs={len(pairs)}")
    surrogate = get_embedder("facenet", "cpu")
    evaluator = get_embedder("arcface", "cpu")
    print(f"surrogate fallback={surrogate.is_fallback} evaluator fallback={evaluator.is_fallback}")
    cfg = CloakConfig()
    # Unattacked baseline: similarity after each transform with NO cloak.
    # This is the ceiling -- no method can score above it after that transform.
    refs = _refs(evaluator, pairs)
    baseline = {}
    for name, key in (("jpeg70", "baseline_jpeg70_sim"),
                      ("resize0.5", "baseline_resize_sim"),
                      ("blur", "baseline_blur_sim")):
        baseline[key] = round(
            float(np.mean([robust_similarity(evaluator, q, ref, name)
                           for (_, q), ref in zip(pairs, refs)])), 4)
    print("unattacked baseline:", baseline)
    rows = []
    for method in METHODS:
        row = evaluate(method, pairs, surrogate, evaluator, cfg)
        row["dataset"] = dataset
        row.update(baseline)
        rows.append(row)
        print(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    if not args.no_plots:
        ablation_plots(rows, Path(args.plots))
        print(f"wrote plots to {args.plots}")


if __name__ == "__main__":
    main()
