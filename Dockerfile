# Frontend build stage
FROM node:20-slim AS frontend-builder

WORKDIR /app/webapp/frontend

COPY webapp/frontend/package*.json ./
RUN npm ci

COPY webapp/frontend ./
RUN npm run build

# Python application stage
FROM python:3.13-slim

# uv installs into /root/.local/bin; copying the static binary avoids a
# second package manager layer.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first for Docker layer caching -- code changes shouldn't
# force a full reinstall. --no-install-project defers installing this repo's
# own package until the source is copied in below.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY webapp ./webapp
COPY README.md ./

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/webapp/frontend/dist ./webapp/static

RUN uv sync --frozen --no-dev

# Render sets $PORT at runtime; default given for local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn webapp.main:app --host 0.0.0.0 --port ${PORT}"]
