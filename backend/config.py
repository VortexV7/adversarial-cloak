"""Shared configuration for Adversarial Cloak.

Single source of truth for the attack / temporal / streaming knobs.
The API (`api.py`) mutates a single global instance of this dataclass,
so the frontend control panel and the pipeline always agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CloakConfig:
    """All tunable parameters. Defaults are the ones in the project brief."""

    # --- attack ---------------------------------------------------------
    epsilon: float = 8 / 255          # L-inf budget (in [0, 1] pixel space)
    steps: int = 10                   # PGD steps per (re-)optimisation
    alpha: float = 2 / 255            # PGD step size; None -> epsilon / steps
    surrogate: str = "facenet"        # differentiable surrogate: facenet | arcface*
    evaluator: str = "arcface"        # held-out black-box evaluator
    use_eot: bool = True              # Expectation-over-Transformation
    use_mask: bool = True             # concentrate perturbation on eyes/nose/mouth
    mask_floor: float = 0.6           # minimum perturbation weight outside the mask
    eot_samples: int = 2              # transforms sampled per PGD step (averaged)

    # --- temporal -------------------------------------------------------
    update_interval: int = 5          # recompute perturbation every N frames
    ema_alpha: float = 0.7            # EMA weight for the *new* perturbation
    iou_threshold: float = 0.5        # below this IoU a face is a new track
    change_threshold: float = 0.08    # mean |crop diff| that forces recomputation
    enabled: bool = True              # protection ON/OFF master switch

    # --- runtime --------------------------------------------------------
    # Device mode (Settings -> Performance):
    #   auto  - GPU if one is available, otherwise CPU
    #   cpu   - everything on the CPU
    #   cuda  - the AI models (FaceNet / ArcFace) on the GPU
    #   both  - GPU runs the AI models, CPU runs detection + video
    #           handling in parallel: the fastest option on a machine
    #           with a GPU. (MediaPipe detection is CPU-only by design,
    #           so it always stays on the CPU.)
    # A mode that asks for the GPU on a machine without one falls back
    # to CPU and reports that in device_status() -- it never crashes.
    device: str = "auto"              # auto | cpu | cuda | both
    jpeg_quality: int = 85            # JPEG quality for WebSocket frames
    max_frame_width: int = 960        # downscale wider frames before processing
    enroll_threshold: float = 0.60    # cosine-sim threshold for "same person"
    # * ArcFace is only a differentiable surrogate when its ONNX graph is
    #   loaded through a torch wrapper; by default it is evaluation-only.

    def resolved_alpha(self) -> float:
        return self.alpha if self.alpha > 0 else self.epsilon / max(1, self.steps)

    @staticmethod
    def cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def cuda_name() -> str | None:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except Exception:
            pass
        return None

    def resolved_device(self) -> str:
        """Torch device for the neural models: 'cuda' or 'cpu'."""
        if self.device in ("cuda", "both"):
            return "cuda" if self.cuda_available() else "cpu"
        if self.device == "cpu":
            return "cpu"
        return "cuda" if self.cuda_available() else "cpu"

    def device_status(self) -> dict:
        """Everything the Settings page needs to render the device picker."""
        cuda = self.cuda_available()
        wants_gpu = self.device in ("cuda", "both", "auto")
        return {
            "mode": self.device,
            "cuda_available": cuda,
            "cuda_name": self.cuda_name(),
            "compute_device": self.resolved_device(),
            "detector_device": "cpu",  # MediaPipe detection is CPU-only
            "gpu_requested": self.device in ("cuda", "both"),
            "using_gpu": self.resolved_device() == "cuda",
            "fallback_warning": bool(wants_gpu and self.device != "auto"
                                     and not cuda),
        }


# Global mutable config (owned by api.py).
CONFIG = CloakConfig()
