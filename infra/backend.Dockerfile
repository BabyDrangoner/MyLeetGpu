FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock pyproject.toml /app/
RUN pip install --no-cache-dir -r requirements.lock
COPY backend /app/backend
COPY alembic.ini /app/alembic.ini
COPY alembic /app/alembic
COPY scripts /app/scripts

ENV PYTHONPATH=/app/backend
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/jobs \
    && chown -R appuser:appuser /data

USER appuser
