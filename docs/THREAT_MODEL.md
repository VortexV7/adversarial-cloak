# Threat Model & Limitations

## What Adversarial Cloak protects against

**Scenario.** You are on a video call, a livestream, or a recorded lecture, and
frames of your face are scraped (screenshots, stream rips, social-media
re-uploads) and run through an off-the-shelf face recogniser to identify you
or to build a gallery of your face.

**Adversary assumed here (passive scraper):**
- Has the video frames, in the form they were streamed (JPEG-compressed,
  possibly resized or lightly blurred by the platform).
- Runs an off-the-shelf recogniser (FaceNet, ArcFace, or a commercial API
  built on similar embeddings) with a fixed operating threshold.
- Does **not** know the cloak is present, does not adapt the recogniser to
  it, and does not re-train anything on cloaked images.

Against that adversary, the cloak works by moving the face crop's embedding
away from the true identity (untargeted dodging) with an L-inf perturbation
of ε = 8/255, optimised with EOT so it survives the platform's JPEG / resize /
blur pipeline. The evaluation script measures exactly this: attack success
rate at a fixed false-accept rate, against FaceNet (white-box) and ArcFace
(black-box transfer), plus robustness after JPEG 70, 0.5× resize, and blur.

## What it does NOT protect against

- **Adaptive attackers.** An attacker who knows a cloak is in use can
  re-train or fine-tune the recogniser on cloaked images, apply a
  denoiser / purification step before recognition, or average embeddings
  over many frames (temporal averaging partly defeats per-frame noise;
  the EMA smoothing in this project deliberately trades some of that
  away for flicker-free video — see below).
- **Adversarially trained models.** Recognisers trained with adversarial
  examples are substantially more robust to this class of attack.
- **Other biometrics.** Voice, gait, clothing, background, and metadata
  are untouched. This tool cloaks the face only.
- **Human identification.** A human watching the video can still recognise
  you. The perturbation is designed to be imperceptible, not to disguise
  you from people.
- **Strong compression / re-encoding.** Heavy re-encoding (very low
  bitrate, aggressive cropping, face alignment + re-crop) can remove or
  weaken the perturbation. EOT covers JPEG q50–95, 0.7–1.0× resize and
  light blur; beyond that, effectiveness drops — the eval's robustness
  columns quantify it.

## Effectiveness varies

- **By model.** Transfer from the FaceNet surrogate to ArcFace is never
  as strong as the white-box result. Numbers in `eval/results.csv` are
  for the exact models installed; swapping the evaluator changes them.
- **By compression.** See the `robust_*` columns in the eval output.
- **By face size / motion.** Small faces (few pixels) leave little room
  for a perturbation; fast head turns trigger re-optimisation (IoU /
  change threshold) and briefly cost FPS.

## Privacy notes (for the user, not the attacker)

- The browser webcam stream goes to the backend over WebSocket on your
  own machine / LAN. Nothing is uploaded to any third party. Enrolment
  stores only a 512-d embedding (`backend/enrolled.json`), never the photo.
- Do not point this at other people without consent. Cloaking someone
  else's face to evade identification can itself be misuse; the tool is
  built for *your own* video.

## Known limitations of this implementation

- The offline **fallback embedders** (used when `facenet-pytorch` /
  `insightface` weights cannot be downloaded) are small seeded CNNs, not
  real recognisers. The fallback ArcFace is a weight-jittered copy of the
  fallback FaceNet (see `backend/embedder.py`). Offline numbers therefore
  demonstrate the mechanism, not FaceNet/ArcFace accuracy. Real weights
  are available in the separate `.venv-real` env (`make install-real`,
  `make eval-real`; see README) — use those numbers in the report.
  insightface models are non-commercial research licence only.
- No generator network (stretch goal) is implemented; the cloak is
  optimisation-based (PGD), which is why recomputation is every N frames.
- CPU performance is modest (see README). The 15 FPS target assumes a
  laptop GPU or a smaller frame / fewer PGD steps on CPU.
