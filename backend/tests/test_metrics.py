import numpy as np

from backend.metrics import (
    attack_success_rate,
    cosine_similarity,
    lpips_score,
    match_confidence,
    psnr,
    ssim,
)


def _img(seed=0, h=64, w=64):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_cosine_similarity_identical_is_one():
    a = np.random.default_rng(0).normal(size=512)
    assert cosine_similarity(a, a) == 1.0


def test_cosine_similarity_orthogonal_is_zero():
    a = np.zeros(4); a[0] = 1.0
    b = np.zeros(4); b[1] = 1.0
    assert cosine_similarity(a, b) == 0.0


def test_match_confidence_monotonic_and_centred():
    assert match_confidence(0.60, threshold=0.60) == 0.5
    assert match_confidence(0.9) > match_confidence(0.3)


def test_attack_success_rate():
    # 3 of 4 originally matched; 2 of those 3 fall below threshold after.
    assert attack_success_rate([0.9, 0.8, 0.7, 0.2], [0.1, 0.75, 0.2, 0.1]) == 2 / 3
    assert attack_success_rate([0.1, 0.2], [0.0, 0.0]) == 0.0  # nobody matched before


def test_ssim_psnr_identical():
    img = _img()
    assert ssim(img, img) == 1.0
    assert psnr(img, img) == float("inf")


def test_ssim_psnr_degrade_with_noise():
    img = _img()
    noisy = np.clip(img.astype(int) + 30, 0, 255).astype(np.uint8)
    assert 0.0 < ssim(img, noisy) < 1.0
    assert 0.0 < psnr(img, noisy) < 60.0


def test_lpips_identical_is_zero():
    img = _img()
    score, is_proxy = lpips_score(img, img)
    assert score == 0.0
    assert isinstance(is_proxy, bool)
