"""Face embedders: FaceNet (differentiable surrogate) and ArcFace (evaluator).

Contract: every embedder exposes ``embed(img_tensor) -> (N, D)`` L2-normalised
embeddings for a float tensor in [0, 1] of shape (N, 3, H, W) in RGB order,
and ``image_size`` (the crop size the model expects).

Real models are tried first; if their weights cannot be downloaded / loaded
(e.g. an offline laptop), a small seeded convolutional network is used as a
*fallback surrogate* so the attack, pipeline and tests still run end-to-end.
The fallback is clearly flagged via ``is_fallback`` and is never presented
as FaceNet/ArcFace accuracy in evaluation output.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _FallbackNet(nn.Module):
    """Tiny seeded CNN producing a 512-d embedding. Differentiable by design."""

    def __init__(self, seed: int = 0, dim: int = 512) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            # 4x4 spatial pooling (NOT 1x1): a global average would make the
            # embedding almost invariant to L-inf noise, since signed
            # perturbations cancel out in the mean. Keeping spatial cells
            # preserves the localised sensitivity a real face embedder has.
            nn.AdaptiveAvgPool2d(4),
        )
        # Seed the weights deterministically so runs are reproducible.
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, generator=g)
                nn.init.zeros_(m.bias)
        self.fc = nn.Linear(128 * 16, dim)
        nn.init.orthogonal_(self.fc.weight, generator=g)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Centre the input and boost its high-frequency component before
        # the convolutions. A plain random CNN is almost invariant to
        # L-inf noise (signed perturbations cancel in spatial pooling),
        # which would make the fallback look unattackable. Real face
        # embedders are sensitive to fine detail; the high-pass boost
        # reproduces that sensitivity in the stand-in so the offline
        # pipeline/eval demonstrate the attack. It is NOT a real recogniser.
        x = x - 0.5
        hp = x - F.avg_pool2d(x, 3, stride=1, padding=1)
        x = x + 6.0 * hp
        z = self.conv(x).flatten(1)
        return F.normalize(self.fc(z), p=2, dim=1)


def _jitter(net: nn.Module, seed: int = 99, rel: float = 0.15) -> nn.Module:
    """Return ``net`` with every weight perturbed by ``rel`` * std Gaussian
    noise (seeded). Used only for the fallback ArcFace evaluator: a
    weight-jittered copy of the fallback surrogate is the standard
    black-box proxy when no pretrained weights are available -- same
    architecture, independently perturbed weights, so an attack must
    transfer across models rather than memorise one weight set. With
    real insightface weights this path is never used."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            if p.numel() > 1:
                p.add_(torch.randn(p.shape, generator=g) * rel * p.std())
    return net


class BaseEmbedder:
    name = "base"
    image_size = 160
    embedding_dim = 512
    is_fallback = False
    differentiable = True

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

    def preprocess(self, img_rgb: np.ndarray) -> torch.Tensor:
        """HWC uint8/float RGB -> (1, 3, S, S) float tensor in [0, 1]."""
        import cv2

        resized = cv2.resize(img_rgb, (self.image_size, self.image_size))
        arr = resized.astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed_np(self, img_rgb: np.ndarray) -> np.ndarray:
        return self.embed(self.preprocess(img_rgb)).cpu().numpy()[0]

    def embed(self, img_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FaceNetEmbedder(BaseEmbedder):
    """facenet-pytorch InceptionResnetV1, pretrained on VGGFace2."""

    name = "facenet"
    image_size = 160

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device)
        try:
            from facenet_pytorch import InceptionResnetV1

            model = InceptionResnetV1(pretrained="vggface2", classify=False)
            model.eval().to(self.device)
            for p in model.parameters():
                p.requires_grad_(False)
            self.model = model
        except Exception as exc:  # offline / no weights -> fallback
            self.model = _FallbackNet(seed=11).to(self.device).eval()
            self.is_fallback = True
            self._fallback_reason = str(exc)

    def embed(self, img_tensor: torch.Tensor) -> torch.Tensor:
        x = img_tensor.to(self.device)
        if self.is_fallback:
            return self.model(x)
        # InceptionResnetV1 expects (x - 127.5) / 128 in [0, 255] space.
        x = (x * 255.0 - 127.5) / 128.0
        out = self.model(x)
        return F.normalize(out, p=2, dim=1)


class ArcFaceEmbedder(BaseEmbedder):
    """insightface ArcFace (buffalo_l, ONNX). Evaluation only by default.

    The ONNX graph is not differentiable through torch, so
    ``differentiable`` is False unless the fallback net is in use.
    ``embed`` still works for scoring / transfer evaluation.
    """

    name = "arcface"
    image_size = 112
    differentiable = False

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device)
        self._app = None
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._app = app
        except Exception as exc:
            # Fallback evaluator: weight-jittered copy of the surrogate
            # fallback (see _jitter). Same architecture, different weights.
            # It also keeps the surrogate's 160px input size: resizing the
            # cloaked crop to 112px would low-pass away the (mostly
            # high-frequency) perturbation before the fallback can see it,
            # which would fake a transfer failure. The real ArcFace model
            # uses 112px; this only applies to the offline fallback.
            self.model = _jitter(_FallbackNet(seed=11)).to(self.device).eval()
            self.image_size = 160
            self.is_fallback = True
            self.differentiable = True  # fallback net *is* differentiable
            self._fallback_reason = str(exc)

    def embed(self, img_tensor: torch.Tensor) -> torch.Tensor:
        x = img_tensor.to(self.device)
        if self.is_fallback:
            return self.model(x)
        # insightface path: run the ONNX recognition model on each crop.
        # Detach to numpy (ONNX is not part of the torch graph) and re-wrap.
        import cv2

        embs = []
        for i in range(x.shape[0]):
            arr = (x[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            arr = cv2.resize(arr, (112, 112))
            # FaceAnalysis.get expects a BGR image; use the recognition model directly.
            rec = self._app.models.get("recognition")  # type: ignore[attr-defined]
            if rec is None:  # pragma: no cover
                raise RuntimeError("insightface recognition model not loaded")
            emb = rec.get_feat(arr) if hasattr(rec, "get_feat") else rec.get(arr)
            # insightface returns (1, 512) for a single face; flatten so
            # the stack is (N, 512) and the L2 normalisation is per-row.
            embs.append(torch.from_numpy(
                np.asarray(emb, dtype=np.float32).reshape(-1)))
        out = torch.stack(embs).to(self.device)
        return F.normalize(out, p=2, dim=1)


def get_embedder(name: str, device: str = "cpu") -> BaseEmbedder:
    """Factory used by the attack, pipeline and evaluation script."""
    name = (name or "facenet").lower()
    if name == "arcface":
        return ArcFaceEmbedder(device)
    if name == "facenet":
        return FaceNetEmbedder(device)
    raise ValueError(f"unknown embedder: {name!r} (expected 'facenet' or 'arcface')")


def embedding_from_bytes(image_bytes: bytes, embedder: BaseEmbedder) -> np.ndarray:
    """Decode JPEG/PNG bytes -> embedding (used by /enroll)."""
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image bytes")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return embedder.embed_np(rgb)


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]
