# Adversarial Cloak — Real-time Face Privacy Filter

A defensive privacy tool and final-year project: it adds **imperceptible
adversarial perturbations to face regions in live video**, so face
recognition models fail to match the identity while the video looks
unchanged to humans.

**Method:** PGD (ε = 8/255, 10 steps) on the face crop only, with
Expectation-over-Transformation (differentiable JPEG q50–95, resize,
Gaussian blur) and a facial-region mask (eyes / nose / mouth). The
perturbation is recomputed every N frames (default 5), reused between
frames, and EMA-smoothed to prevent flicker. FaceNet is the differentiable
surrogate; ArcFace is the held-out black-box evaluator.

## Quick start

```bash
make dev          # backend on :8000, frontend on :5173
```

Then open http://localhost:5173, press **Start camera**, and flip the
protection switch. Or run the pieces separately:

```bash
make backend      # uvicorn backend.api:app --port 8000
make frontend     # vite dev server on :5173
make test         # 25 backend unit tests
make eval         # evaluation -> eval/results.csv + eval/plots/*.png
make demo         # cloak one image -> demo_out/side_by_side.png
docker compose up --build   # containerised run (not verified on the build VM: no Docker there)
```

Python 3.10+; the Makefile creates `.venv` and installs CPU PyTorch
first (the wheel comes from the PyTorch index, not PyPI). Copy
`.env.sample` to `.env` to change host/port/device.

### Real models (report-grade numbers)

The default env runs with fallback embedders when the real packages are
absent. For real FaceNet / ArcFace / MediaPipe, use a **separate venv** —
do not install them into `.venv`:

```bash
make install-real   # creates .venv-real (torch 2.2.2 line, facenet-pytorch 2.6.0)
make test-real      # tests against the real models
make eval-real      # evaluation with real FaceNet + real ArcFace
```

Why separate: facenet-pytorch 2.6.0 pins `torch>=2.2,<2.3`,
`numpy<2`, `Pillow<10.3` — installing it into the main env would
downgrade torch 2.15 → 2.2 and numpy 2.5 → 1.26. The real stack is
pinned in `requirements-real.txt`; it also needs the system library
`libEGL.so.1` for MediaPipe 1.2.2 (`apt install libegl1 libgl1`) and
downloads FaceNet VGGFace2 weights (~107 MB) plus the insightface
`buffalo_l` pack (~275 MB) on first use. Versions that matter:
insightface is **2.1** (not 0.7.3) and mediapipe is **1.2.2**, where the
legacy `mp.solutions` API no longer exists — `detector.py` uses the
Tasks API (`FaceDetector` + the bundled `blaze_face_short_range.tflite`).
insightface models are non-commercial research licence only — fine for
a final-year project.

### Device modes (GPU / CPU / Both) — v1.1.0

Settings → **Performance — device** picks where the heavy AI work runs.
The backend config field is `device` (`auto | cpu | cuda | both`, default
`auto`); `GET /device` reports GPU availability and the effective device,
and changing the mode rebuilds the pipeline on the new device.

- **Auto** — GPU if one is available, otherwise CPU.
- **CPU only** — everything on the CPU; works everywhere, slowest.
- **GPU only** — FaceNet / ArcFace run on the GPU (cuDNN benchmarking on).
  On a machine without CUDA it falls back to CPU and the UI says so.
- **Both** — GPU runs the AI models, CPU handles detection (MediaPipe is
  CPU-only by design) and video work, with all CPU cores given to the
  CPU-side work: the fastest option on a machine with a GPU.

The header shows a live GPU/CPU chip. ~3 FPS on CPU with the real models;
15 FPS is a laptop-GPU target.

### Image Cloak — v1.1.0

The **Image Cloak** page cloaks a still photo instead of a live feed:
upload a picture (drag-and-drop or file picker), press **Cloak this
photo**, and every face in it gets the same PGD+EOT+mask attack, baked
into the image — the photo looks unchanged to people, but face-recognition
AI can't match the people in it. The page shows the original and cloaked
photos side by side with similarity before → after, match confidence,
SSIM and a **Download cloaked photo** button. Backend: `POST /cloak-image`
(multipart photo → base64 cloaked JPEG + metrics; 422 when no face is
found). A photo has no "previous frame", so each face is cloaked fresh
with no temporal cache reuse (`CloakPipeline.cloak_image`).

### UI notes — v1.1.0

- Every setting that needs explaining has a small **ⓘ button** next to it
  (epsilon, PGD steps, update interval, EOT, mask, surrogate, device,
  EMA, match threshold) — one short plain-English line each, nothing more.
- The control panel is grouped into Strength / Speed / Robustness, the
  sidebar carries a one-line plain-English summary of what the tool does,
  and the app background is a quiet teal-tinted wash (light) / deep navy
  gradient (dark) with glass cards.

## Repository layout

```
backend/
  detector.py   MediaPipe Tasks API face detection (Haar fallback), margined crops
  embedder.py   FaceNet (differentiable surrogate) + ArcFace (evaluator, ONNX)
  attack.py     FGSM, PGD, Cloak (PGD+EOT+mask), differentiable JPEG (DCT)
  temporal.py   IoU tracker, perturbation cache, EMA smoothing, recompute policy
  metrics.py    cosine sim, match confidence, ASR, SSIM/PSNR/LPIPS, latency/FPS
  pipeline.py   detect -> attack -> feathered composite -> metrics
  api.py        FastAPI: WS /ws/stream, /config, /enroll, /benchmark, /export-report, /rtsp/*
  main.py       uvicorn entry point
frontend/       React + Vite + TypeScript + Tailwind + Recharts + Framer Motion
eval/
  run_eval.py       FGSM / PGD / Ours / Ours-no-EOT on LFW (or synthetic pairs)
  download_lfw.py   fetches LFW once via scikit-learn
docs/
  THREAT_MODEL.md   what it protects against — and what it does not
  DEMO_SCRIPT.md    3-minute demo + screenshot list for the report
  SIMPLE_EXPLAINER.md  plain-language explainer + real LFW results + likely viva Q&A
scripts/demo_single_image.py
```

## How it works (pipeline)

```
webcam / RTSP --JPEG--> WS /ws/stream --> detect (MediaPipe)
  -> track by IoU -> recompute? (every N frames, or face moved/changed)
  -> PGD+EOT+mask on the crop (FaceNet surrogate)
  -> EMA-smooth the perturbation -> feathered composite back into frame
  -> metrics vs ArcFace evaluator -> WS reply {original, protected, metrics}
```

## Measured results (real models, LFW, 20 same-person pairs)

Run with `make eval-real` on this VM (CPU, no GPU): real FaceNet
VGGFace2 as the surrogate, real ArcFace `buffalo_l` as the held-out
evaluator, MediaPipe Tasks detection. ASR threshold 0.60. LPIPS here is
the flagged MSE proxy (the `lpips` package is not installed in the
real-model venv). Full CSV: `eval/results_real.csv`.

| method | ASR FaceNet | ASR ArcFace | Rank-1 drop | SSIM | PSNR | LPIPS* | latency |
|--------|-------------|-------------|-------------|------|------|--------|---------|
| FGSM | 0.74 | 1.00 | 0.29 | 0.925 | 31.4 | 0.0131 | 1027 ms |
| PGD | 1.00 | 0.95 | 0.66 | 0.964 | 34.0 | 0.0071 | 3442 ms |
| **Ours (PGD+EOT+mask)** | **1.00** | **0.95** | **0.63** | **0.976** | **35.6** | **0.0051** | **8156 ms** |
| Ours, no EOT | 0.84 | 0.89 | 0.35 | 0.980 | 36.2 | 0.0044 | 6202 ms |

Robustness (mean ArcFace similarity after transform; lower = more
robust; unattacked baseline in brackets — note the baseline itself is
already low, 0.42–0.49, because these are tiny 62×47 LFW crops and the
transforms destroy much of ArcFace's signal, so robust ASR is ~0.9 for
every method including the baseline and the *similarity* is the
informative column):

| method | JPEG 70 | resize 0.5× | blur |
|--------|---------|-------------|------|
| PGD | 0.427 (0.486) | 0.393 (0.425) | 0.395 (0.425) |
| **Ours** | **0.434** | **0.393** | **0.396** |
| Ours, no EOT | 0.444 | 0.402 | 0.403 |

Reading the table honestly: on real weights, EOT's win is **clean
attack success and invisibility, not robustness**. Ours matches PGD's
ASR (1.00 / 0.95) with the best SSIM/PSNR/LPIPS of the strong methods,
and EOT clearly helps vs Ours-no-EOT (FaceNet ASR 0.84 → 1.00, ArcFace
0.89 → 0.95, Rank-1 drop 0.35 → 0.63). But after JPEG/resize/blur, PGD
and Ours survive about equally (both below the unattacked baseline) —
the fallback nets had overstated EOT's robustness edge, because their
attackable features were almost entirely high-frequency. n = 20 pairs,
so treat ±0.05 differences as noise.

Single-image demo (synthetic face, MediaPipe Tasks detection, first
frame incl. model warm-up): similarity 1.00 → 0.67, confidence
0.9993 → 0.7857, SSIM 0.894, PSNR 30.2, LPIPS* 0.017. The loose
synthetic crop (background + hair) dilutes the face; tight face crops
(the eval pairs) score better.

**Performance, measured on this VM (CPU, real models):** steady-state
~300–390 ms/frame at 960 px (≈ 3 FPS) for reused perturbations;
recompute frames ~1 s, first frame ~8.7 s (FaceNet + ArcFace warm-up).
The 15 FPS target assumes a laptop GPU, or fewer PGD steps / a larger
update interval N on CPU. For scale: facenet-pytorch's own benchmark
is ~13 FPS at 1080p for MTCNN detection alone on a P100 GPU — PGD with
10 steps + EOT on CPU will be far below 15 FPS, so treat 15 FPS as a
laptop-GPU target only.

### Earlier fallback numbers (superseded — do not cite)

The first eval ran on the offline fallback embedders (small seeded
CNNs) because the real weights were not yet installed: PGD ASR
1.00/1.00, Ours 0.42/0.08, SSIM 0.915. Those numbers demonstrated the
mechanism only. A fifth bug found during the real-model re-verification:
`eval/run_eval.py` loaded LFW with `.astype(np.uint8)` on sklearn's
float [0, 1] images, turning them into near-black 0/1 images — the
first real-model eval run was garbage for that reason and was discarded.
The fix (`_to_uint8`, scale floats to 0–255) is in the code; the table
above is the corrected run.

## API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | liveness, device, model/fallback status |
| `GET /device` | device modes + GPU availability (Settings picker) |
| `GET/POST /config` | epsilon, steps, update interval, model, EOT/mask toggles, protection on/off, device mode |
| `POST /cloak-image` (multipart) | upload a photo → cloaked photo (base64) + metrics; 422 when no face is found |
| `POST /enroll` (multipart) | store a reference embedding (512-d only, never the photo) |
| `GET/DELETE /enroll` | enrolled identity / clear it |
| `POST /benchmark` | run `eval/run_eval.py`, returns the results table |
| `POST /export-report?fmt=csv\|pdf` | session metrics summary |
| `POST /rtsp/start` `{"url": ...}` / `POST /rtsp/stop` | RTSP/IP-camera source |
| `WS /ws/stream` | binary JPEG in → `{original, protected, metrics, boxes}` JSON out |

## Threat model & limitations

Short version: protects against **passive scraping and off-the-shelf
recognisers**, not adaptive attackers or adversarially trained models;
effectiveness varies by model and compression. Full discussion in
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

## Bugs found by the evaluation (and fixed)

1. **FGSM/PGD stepped the wrong way.** The loss is cosine similarity,
   which we *minimise* — the original code stepped *with* the gradient,
   climbing back toward the original embedding. PGD only looked like it
   worked because of its random start.
2. **FGSM from the original does nothing.** Cosine similarity is
   stationary at the original (gradient exactly zero — 1.0 is a maximum),
   so FGSM now starts from a small random offset, like PGD.
3. **The differentiable JPEG was broken.** The quantisation table is
   defined for 8-bit coefficients but was applied to [0, 1] values, so
   q=70 destroyed the image (mean error 0.40). It now runs in 0–255 space.
4. **Pipeline compared embeddings across models.** `similarity_after`
   was computed with the evaluator against a surrogate reference — ~0
   even uncloaked. Both similarities are now computed in the evaluator's
   space.
5. **LFW images were truncated to 0/1.** scikit-learn returns LFW as
   float [0, 1]; `.astype(np.uint8)` made near-black images, so the
   first real-model eval was garbage (FGSM "did nothing", PGD's ASR
   inverted, Ours == Ours-no-EOT). Fixed with `_to_uint8` in
   `eval/run_eval.py`.
6. **Real-model integration fixes.** insightface's `get_feat` returns
   (1, 512), which made the ArcFace embedding unnormalised (norm 22.6)
   — now flattened before stacking. `torch.rand_like(..., generator=...)`
   does not exist in torch 2.2 — replaced with a `torch.rand` helper.
   MediaPipe 1.2.2 removed `mp.solutions` — the detector now uses the
   Tasks API and needs `libEGL.so.1` on the system (`apt install
   libegl1`).

## Stretch goal

The lightweight generator network was skipped, per the brief: the
optimisation-based cloak is complete and the generator would only add
value after real-weight evaluation.
