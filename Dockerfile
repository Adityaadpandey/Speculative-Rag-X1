FROM node:20-slim AS frontend-build
WORKDIR /app/webapp/frontend
COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci
COPY webapp/frontend ./
RUN npm run build

FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev
COPY src ./src
COPY webapp ./webapp
COPY --from=frontend-build /app/webapp/static ./webapp/static
COPY README.md ./
RUN uv sync --frozen --no-dev
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uv run uvicorn webapp.main:app --host 0.0.0.0 --port ${PORT}"]
