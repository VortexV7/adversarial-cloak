FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# OpenCV / MediaPipe system deps (libegl1 is required by mediapipe 1.x's
# native Tasks library; without it the detector falls back to Haar).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU PyTorch first (large wheel), then the rest from PyPI.
# NOTE: this image runs the MAIN stack (torch 2.15, fallback embedders
# unless the real-model packages are added). For real FaceNet/ArcFace,
# build from requirements-real.txt in a separate venv (see README).
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY eval/ ./eval/
COPY scripts/ ./scripts/

EXPOSE 8000
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
