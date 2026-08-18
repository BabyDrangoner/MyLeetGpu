SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
PNPM ?= npx --yes pnpm@9.15.1
MYLEETGPU_HOST_DATA_DIR ?= $(CURDIR)/data
MYLEETGPU_HOST_UID ?= $(shell id -u)
MYLEETGPU_HOST_GID ?= $(shell id -g)
MYLEETGPU_DOCKER_GID ?= $(shell stat -c '%g' /var/run/docker.sock 2>/dev/null || echo 0)
MYLEETGPU_LAN_ADDRESS ?= $(shell bash scripts/lan-auth.sh address 2>/dev/null)
MYLEETGPU_LAN_PORT ?= 3000
export MYLEETGPU_HOST_DATA_DIR MYLEETGPU_HOST_UID MYLEETGPU_HOST_GID MYLEETGPU_DOCKER_GID
export MYLEETGPU_LAN_ADDRESS MYLEETGPU_LAN_PORT

.PHONY: help install doctor start start-lan stop stop-lan ps logs lint test test-gpu e2e clean-jobs recover-runner migrate lan-password lan-firewall lan-firewall-off lan-status

help:
	@echo "MyLeetGpu commands:"
	@echo "  make install     Install locked backend and frontend dependencies"
	@echo "  make doctor      Inspect WSL2, Docker, NVIDIA and CUDA end to end"
	@echo "  make start       Start web, API and the single GPU worker"
	@echo "  make start-lan   Start an authenticated Web endpoint on the LAN address"
	@echo "  make stop        Stop services"
	@echo "  make stop-lan    Stop services started with the LAN overlay"
	@echo "  make ps          Show service status"
	@echo "  make lint        Run backend and frontend static checks"
	@echo "  make test        Run all non-GPU tests"
	@echo "  make test-gpu    Run opt-in real NVIDIA GPU acceptance tests"
	@echo "  make e2e         Run browser end-to-end tests"
	@echo "  make clean-jobs  Remove completed temporary job directories"
	@echo "  make recover-runner  Re-probe GPU and clear the runner circuit breaker"
	@echo "  make lan-password   Set or rotate LAN Basic Auth credentials"
	@echo "  make lan-firewall   Allow LAN subnet TCP/3000 (requires elevated Windows shell)"
	@echo "  make lan-firewall-off  Remove the scoped LAN firewall rules"

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

install: $(VENV)/bin/python
	$(PIP) install -r requirements.lock
	cd apps/web && $(PNPM) install --frozen-lockfile
	cd apps/web && $(PNPM) exec playwright install chromium

doctor:
	bash scripts/doctor.sh

start:
	@mkdir -p data/jobs
	docker compose up --build -d
	@echo "MyLeetGpu: http://localhost:3000"

start-lan:
	@test -n "$(MYLEETGPU_LAN_ADDRESS)" || (echo "Cannot detect LAN IPv4; set MYLEETGPU_LAN_ADDRESS explicitly" >&2; exit 1)
	@mkdir -p data/jobs
	@bash scripts/lan-auth.sh ensure
	docker compose -f docker-compose.yml -f docker-compose.lan.yml up --build -d
	@echo "MyLeetGpu local: http://localhost:3000"
	@echo "MyLeetGpu LAN:   http://$(MYLEETGPU_LAN_ADDRESS):$(MYLEETGPU_LAN_PORT)"
	@echo "If LAN access is blocked, run 'make lan-firewall' from an elevated Windows/WSL terminal."

stop:
	docker compose down --remove-orphans

stop-lan:
	docker compose -f docker-compose.yml -f docker-compose.lan.yml down --remove-orphans

ps:
	docker compose ps -a

logs:
	docker compose logs -f --tail=200

migrate: $(VENV)/bin/python
	MYLEETGPU_DATA_DIR="$$(pwd)/data" PYTHONPATH=backend $(VENV)/bin/alembic upgrade head

lint: $(VENV)/bin/python
	PYTHONPATH=backend $(RUFF) check backend tests scripts
	PYTHONPATH=backend $(RUFF) format --check backend tests scripts
	cd apps/web && $(PNPM) lint && $(PNPM) typecheck

test: $(VENV)/bin/python
	PYTHONPATH=backend $(PYTEST) -m "not gpu and not e2e"
	cd apps/web && $(PNPM) test

test-gpu: $(VENV)/bin/python
	MYLEETGPU_RUN_GPU_TESTS=1 PYTHONPATH=backend $(PYTEST) -m gpu -v

e2e:
	cd apps/web && $(PNPM) e2e

clean-jobs: $(VENV)/bin/python
	MYLEETGPU_DATA_DIR="$$(pwd)/data" MYLEETGPU_HOST_DATA_DIR="$$(pwd)/data" PYTHONPATH=backend $(VENV)/bin/python -m myleetgpu.cli clean-jobs

recover-runner: $(VENV)/bin/python
	MYLEETGPU_DATA_DIR="$$(pwd)/data" MYLEETGPU_HOST_DATA_DIR="$$(pwd)/data" PYTHONPATH=backend $(VENV)/bin/python -m myleetgpu.cli recover-runner

lan-password:
	@bash scripts/lan-auth.sh reset

lan-firewall:
	powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$$(wslpath -w scripts/lan-firewall.ps1)" -Action Enable -Port "$(MYLEETGPU_LAN_PORT)" -ListenAddress "$(MYLEETGPU_LAN_ADDRESS)"

lan-firewall-off:
	powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$$(wslpath -w scripts/lan-firewall.ps1)" -Action Disable -Port "$(MYLEETGPU_LAN_PORT)"

lan-status:
	@echo "LAN URL: http://$(MYLEETGPU_LAN_ADDRESS):$(MYLEETGPU_LAN_PORT)"
	powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$$(wslpath -w scripts/lan-firewall.ps1)" -Action Status -Port "$(MYLEETGPU_LAN_PORT)"
	docker compose -f docker-compose.yml -f docker-compose.lan.yml ps -a
