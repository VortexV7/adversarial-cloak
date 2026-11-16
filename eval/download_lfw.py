"""Download LFW (Labeled Faces in the Wild) for the evaluation harness.

Uses scikit-learn's ``fetch_lfw_pairs`` (test split, colour, resized).
Run once: ``python eval/download_lfw.py`` -- data lands in ``eval/lfw_home/``.
If the download fails (offline), ``run_eval.py`` falls back to synthetic
pairs and labels the CSV ``dataset=synthetic``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    from sklearn.datasets import fetch_lfw_pairs

    data = fetch_lfw_pairs(subset="test", data_home=str(ROOT / "eval" / "lfw_home"),
                           download_if_missing=True, color=True, resize=0.5)
    print(f"LFW downloaded: {len(data.pairs)} pairs, targets={data.target[:10].tolist()}")


if __name__ == "__main__":
    main()
