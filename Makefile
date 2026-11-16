# Adversarial Cloak — one-command workflows
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: dev backend frontend install install-real test eval demo docker-up docker-down clean

REALVENV ?= .venv-real
REALPY := $(REALVENV)/bin/python
REALPIP := $(REALVENV)/bin/pip

dev: install ## Run backend + frontend together (backend :8000, frontend :5173)
	@echo "Starting backend on :8000 and frontend on :5173 ..."
	@(trap 'kill 0' INT; $(PY) -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 & cd frontend && npm run dev)

backend: ## Run only the FastAPI backend
	$(PY) -m uvicorn backend.api:app --host 0.0.0.0 --port 8000

frontend: ## Run only the React frontend (expects backend on :8000)
	cd frontend && npm run dev

install: $(VENV)/bin/activate ## Create venv + install backend + frontend deps
$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt
	cd frontend && npm install --no-audit --no-fund

install-real: $(REALVENV)/bin/activate ## Separate venv with REAL FaceNet/ArcFace/MediaPipe (torch 2.2 line)
$(REALVENV)/bin/activate:
	python3 -m venv $(REALVENV)
	$(REALPIP) install --upgrade pip
	$(REALPIP) install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
	$(REALPIP) install --no-deps insightface==2.1 mediapipe==1.2.2
	$(REALPIP) install -r requirements-real.txt
	@echo "Real-model venv ready. First run downloads FaceNet (~107 MB) and"
	@echo "insightface buffalo_l (~275 MB) into ~/.torch and ~/.insightface."

test-real: ## Backend tests against the real-model venv
	$(REALPY) -m pytest backend/tests/ -q

eval-real: ## Evaluation with real FaceNet (surrogate) + real ArcFace (evaluator)
	$(REALPY) -m eval.run_eval --pairs 20

test: ## Backend unit tests
	$(PY) -m pytest backend/tests/ -q

eval: ## Run the LFW/synthetic evaluation (writes eval/results.csv + plots)
	$(PY) -m eval.run_eval --pairs 20

demo: ## Cloak a single image (synthetic face if --input is omitted)
	$(PY) scripts/demo_single_image.py --output demo_out

docker-up: ## Build + run with Docker Compose
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf $(VENV) frontend/node_modules frontend/dist demo_out eval/results.csv eval/plots __pycache__ backend/__pycache__
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
