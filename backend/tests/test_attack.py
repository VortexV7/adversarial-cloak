import numpy as np
import torch

from backend.attack import cloak_attack, facial_mask, fgsm, jpeg_compress, pgd
from backend.config import CloakConfig
from backend.embedder import FaceNetEmbedder, get_embedder
from backend.metrics import cosine_similarity


def _face_like_tensor():
    """Deterministic face-like 160x160 RGB tensor (gradient + blobs)."""
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:160, 0:160]
    img = np.stack([
        0.55 + 0.2 * np.sin(x / 25.0) + rng.normal(0, 0.03, (160, 160)),
        0.45 + 0.15 * np.cos(y / 30.0) + rng.normal(0, 0.03, (160, 160)),
        0.40 + 0.1 * np.sin((x + y) / 40.0) + rng.normal(0, 0.03, (160, 160)),
    ]).astype(np.float32)
    return torch.from_numpy(np.clip(img, 0, 1)).unsqueeze(0)


def _sim(embedder, a, b):
    with torch.no_grad():
        ea = embedder.embed(a).cpu().numpy()[0]
        eb = embedder.embed(b).cpu().numpy()[0]
    return cosine_similarity(ea, eb)


def test_facial_mask_shape_and_range():
    m = facial_mask(160, 160, floor=0.35)
    assert m.shape == (1, 1, 160, 160)
    assert 0.3 <= float(m.min()) <= float(m.max()) <= 1.0
    # Eyes region is weighted more than a corner.
    assert float(m[0, 0, 60, 80]) > float(m[0, 0, 5, 5])


def test_jpeg_compress_is_identity_at_high_quality_and_bounded():
    x = _face_like_tensor()
    out = jpeg_compress(x, 99)
    assert torch.allclose(out, x)
    out = jpeg_compress(x, 50)
    assert out.shape == x.shape and float(out.min()) >= 0 and float(out.max()) <= 1


def test_fgsm_respects_epsilon():
    emb = get_embedder("facenet", "cpu")
    x = _face_like_tensor()
    adv = fgsm(emb, x, eps=8 / 255)
    assert adv.shape == x.shape
    assert float((adv - x).abs().max()) <= 8 / 255 + 1e-6


def test_pgd_lowes_similarity_and_respects_epsilon():
    emb = get_embedder("facenet", "cpu")
    x = _face_like_tensor()
    adv = pgd(emb, x, eps=8 / 255, steps=10)
    assert float((adv - x).abs().max()) <= 8 / 255 + 1e-6
    assert _sim(emb, x, adv) < _sim(emb, x, x)  # strictly less similar than itself


def test_cloak_attack_with_eot_and_mask():
    emb = get_embedder("facenet", "cpu")
    cfg = CloakConfig(epsilon=8 / 255, steps=10, use_eot=True, use_mask=True)
    x = _face_like_tensor()
    adv = cloak_attack(emb, x, cfg, seed=0)
    assert adv.shape == x.shape
    assert float((adv - x).abs().max()) <= 8 / 255 + 1e-6
    assert _sim(emb, x, adv) < 1.0
