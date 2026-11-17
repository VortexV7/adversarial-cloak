# Adversarial Cloak — explained in simple words

*For Ved — read this before the demo / viva. Every number here is from the
real-model LFW test (`eval/results_real.csv`, 20 same-person pairs).*

## The one-line answer

**It hides your face from face-recognition AI, without changing how you
look to a human.**

You are on a video call. People watching see you normally — same face,
same expressions, nothing weird. But the program adds a tiny invisible
change to just your face. Face-recognition software looking at that video
can no longer tell who you are. Humans see you. Machines can't recognise
you.

## How it works — 4 steps

1. **Find the face.** MediaPipe finds your face in each video frame.
2. **Add an invisible change.** It changes the face pixels by a tiny amount
   (at most 8 out of 255 brightness levels — too small for your eyes to see).
   The change is carefully chosen so a face-recognition AI gets confused.
3. **Don't redo it every frame.** Recomputing is slow, so it reuses the
   same change for 5 frames, then recomputes. It also smooths the change
   over time so the video doesn't flicker.
4. **Check with a second AI.** One AI (FaceNet) is used to *make* the
   change. A different AI (ArcFace) is used to *test* it — so the test is
   honest, not rigged in our favour.

## The words, in simple language

- **FaceNet / ArcFace** — two face-recognition AIs. Each turns a face into
  a list of 512 numbers (an "embedding"). Same person = similar numbers.
- **PGD** — the method that finds the confusing change. It takes 10 small
  steps, each step pushing the face's numbers further away from your real
  face. Our method is PGD + two extras.
- **EOT** — "expectation over transformation". While finding the change, it
  also tests JPEG compression, resizing and blur, so the change still works
  after a video app compresses the frame.
- **Mask** — the change is stronger on eyes, nose and mouth (where the AI
  looks most) and weaker elsewhere, so the picture stays clean.
- **Epsilon (8/255)** — how big the change is allowed to be. Bigger = more
  confusing to AI, but more visible to humans. 8/255 is the balance.
- **ASR (attack success rate)** — out of 100 tries, how often the AI fails
  to recognise the person. Higher is better for us.
- **SSIM / PSNR** — how similar the protected picture looks to the
  original. SSIM near 1.0 = looks the same. Ours is 0.976 — nearly identical.

## The results to quote (real models, LFW, 20 pairs)

| Method | Fails FaceNet | Fails ArcFace | Looks the same (SSIM) |
|---|---|---|---|
| FGSM (one big step) | 74% | 100% | 0.925 |
| PGD (10 steps) | 100% | 95% | 0.964 |
| **Ours (PGD + EOT + mask)** | **100%** | **95%** | **0.976** |
| Ours without EOT | 84% | 89% | 0.980 |

In one sentence: **ours matches the strongest attack (PGD) and looks the
cleanest, and EOT is what lifts it from 84% to 100% on FaceNet.**
Honest caveat: after JPEG/resize/blur, ours and PGD survive about equally —
EOT's real win is clean success + invisibility, not extra robustness.
And 20 pairs is small, so treat ±5% as noise.

## Numbers you must NOT quote

The early "similarity 1.0 → -0.04, confidence 99.9% → 0%" numbers were
**not FaceNet** — they came from a small stand-in network because the real
models weren't installed yet. They only prove the pipeline works. Do not
put them in the report. The table above is the real one.

## Why it is slow on your laptop

Your laptop GPU is dead, so everything runs on the CPU: about **3 frames
per second** with the real models (first frame ~9 seconds to warm up).
15 FPS is a laptop-GPU target only. To make it faster right now, in the
app's control panel set **PGD steps 10 → 5** and **update interval 5 → 10**.
Protection gets slightly weaker, speed gets noticeably better.

## What it does NOT do (say this in the viva — it scores marks)

- It does **not** hide you from humans. A person watching still knows
  it's you.
- It does **not** beat an attacker who knows the trick and retrains their
  recogniser on cloaked faces, or who cleans the image first.
- It does **not** protect voice, clothes, background, or anything except
  the face.
- It works against passive scrapers and off-the-shelf recognisers — that
  is the threat model. Nothing more, nothing claimed.

## Likely viva questions, with short answers

- **"Why PGD and not FGSM?"** FGSM takes one big step; from the original
  face its gradient is zero, so it does almost nothing. PGD takes 10 small
  steps from a random start, so it actually moves the face away.
- **"Why two models?"** FaceNet makes the cloak (white-box — we can see
  inside it). ArcFace only tests it (black-box — we can't see inside).
  If it fools a model it never saw, the trick is real, not memorised.
- **"Why EOT?"** Video apps compress, resize and blur frames. A change
  that only works on the perfect original dies on upload. EOT trains the
  change to survive those steps.
- **"Why recompute only every 5 frames?"** PGD is slow. Your face barely
  moves in 5 frames, so reusing the change is nearly as good and much
  faster. It recomputes sooner if the face moves a lot.
- **"Why is it only on the face crop?"** Changing the whole frame would be
  slower and more visible. The AI only looks at the face, so only the face
  needs changing.
- **"What is the main limitation?"** CPU speed (~3 FPS), and an adaptive
  attacker who retrains on cloaked images can beat it.

## How to run it on your PC

Git Bash (not cmd — cmd has no `make`):
```bash
make dev
```
Then open http://localhost:5173, press **Start camera**, flip the
protection switch. PDF export: if it still says "Failed to load PDF
document", run `.venv\Scripts\pip install reportlab` and restart the
backend (that was the Nov 6 fix, in `adversarial-cloak-v1.0.1.zip`).
