SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff

.PHONY: help install doctor start stop logs lint test test-gpu e2e clean-jobs migrate

help:
	@echo "MyLeetGpu commands:"
	@echo "  make install     Install locked backend and frontend dependencies"
	@echo "  make doctor      Inspect WSL2, Docker, NVIDIA and CUDA end to end"
	@echo "  make start       Start web, API and the single GPU worker"
	@echo "  make stop        Stop services"
	@echo "  make lint        Run backend and frontend static checks"
	@echo "  make test        Run all non-GPU tests"
	@echo "  make test-gpu    Run opt-in real NVIDIA GPU acceptance tests"
	@echo "  make e2e         Run browser end-to-end tests"
	@echo "  make clean-jobs  Remove completed temporary job directories"

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

install: $(VENV)/bin/python
	$(PIP) install -r requirements.lock
	cd apps/web && npm ci

doctor:
	bash scripts/doctor.sh

start:
	@mkdir -p data/jobs
	@export MYLEETGPU_HOST_DATA_DIR="$$(pwd)/data"; docker compose up --build -d
	@echo "MyLeetGpu: http://localhost:3000"

stop:
	docker compose down --remove-orphans

logs:
	docker compose logs -f --tail=200

migrate: $(VENV)/bin/python
	MYLEETGPU_DATA_DIR="$$(pwd)/data" PYTHONPATH=backend $(VENV)/bin/alembic upgrade head

lint: $(VENV)/bin/python
	PYTHONPATH=backend $(RUFF) check backend tests scripts
	PYTHONPATH=backend $(RUFF) format --check backend tests scripts
	cd apps/web && npm run lint && npm run typecheck

test: $(VENV)/bin/python
	PYTHONPATH=backend $(PYTEST) -m "not gpu and not e2e"
	cd apps/web && npm test -- --run

test-gpu: $(VENV)/bin/python
	MYLEETGPU_RUN_GPU_TESTS=1 PYTHONPATH=backend $(PYTEST) -m gpu -v

e2e:
	cd apps/web && npm run e2e

clean-jobs: $(VENV)/bin/python
	PYTHONPATH=backend $(VENV)/bin/python -m myleetgpu.cli clean-jobs
