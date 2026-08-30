.PHONY: help install data synth train test smoke profile api dashboard mlflow notebook \
        build up down logs lint clean release

PY ?= python

help:
	@echo ""
	@echo "FraudLens - command reference"
	@echo "==========================================================="
	@echo "  make install    Install dependencies"
	@echo "  make synth      Generate synthetic data (no Kaggle needed)"
	@echo "  make data       Download the real IEEE-CIS data from Kaggle"
	@echo "  make train      Train and export all serving artifacts"
	@echo "  make test       Run the test suite"
	@echo "  make smoke      Exercise the API in-process"
	@echo "  make profile    Break scoring latency down by stage"
	@echo "  make api        Serve the API locally on :8000"
	@echo "  make dashboard  Serve the Streamlit dashboard on :8501"
	@echo "  make mlflow     Open the MLflow UI on :5000"
	@echo "  make notebook   Open the analysis notebook"
	@echo "  make up         Start the full stack in Docker"
	@echo "  make down       Stop the stack"
	@echo "  make release    Train, then build an image with that model baked in"
	@echo ""
	@echo "  First run: make install && make synth && make train && make smoke"
	@echo ""

install:
	$(PY) -m pip install -r requirements.txt

synth:
	$(PY) scripts/make_synthetic_data.py

data:
	$(PY) scripts/download_data.py

train:
	$(PY) scripts/train.py

train-fast:
	$(PY) scripts/train.py --fast

test:
	$(PY) -m pytest tests/ -v

smoke:
	$(PY) scripts/smoke_test.py

profile:
	$(PY) scripts/profile_latency.py

api:
	$(PY) -m uvicorn api.main:app --reload --port 8000

dashboard:
	$(PY) -m streamlit run dashboard.py --server.port 8501

# Same store scripts/train.py logs to; see fraudlens/config.py.
mlflow:
	$(PY) -m mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db

notebook:
	$(PY) -m jupyter notebook FraudLens.ipynb

build:
	docker compose build

up:
	docker compose up -d
	@echo ""
	@echo "  API       -> http://localhost:8000"
	@echo "  API docs  -> http://localhost:8000/docs"
	@echo "  MLflow    -> http://localhost:5000"
	@echo "  Dashboard -> http://localhost:8501"

down:
	docker compose down

logs:
	docker compose logs -f

# The image tag records which model is inside it, so a rollback is just the
# previous tag rather than a retrain.
TAG ?= $(shell date +%Y%m%d-%H%M%S)

release: train
	docker build -t fraudlens:$(TAG) -t fraudlens:latest .
	@echo ""
	@echo "  Built fraudlens:$(TAG) with the model baked in."
	@echo "  Verify: docker run --rm -p 8000:8000 fraudlens:$(TAG)"

lint:
	$(PY) -m ruff check fraudlens api scripts tests

clean:
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PY) -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"
