FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime
ARG SOURCE_REVISION=unknown
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
    SLINKY_DATA_DIR=/data SLINKY_STATIC_DIR=/app/static \
    SLINKY_SOURCE_REVISION=${SOURCE_REVISION} \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LABEL org.opencontainers.image.title="Slinky Lab" \
      org.opencontainers.image.description="三维彩虹圈动力学研究工作台" \
      org.opencontainers.image.source="https://github.com/FanYu-Nijika/slinky-lab" \
      org.opencontainers.image.revision=${SOURCE_REVISION}
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 slinky \
    && mkdir /data && chown slinky:slinky /data
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY pyproject.toml ./
COPY backend/ ./backend/
RUN pip install --no-cache-dir --no-deps .
COPY --from=frontend /build/frontend/dist /app/static
COPY docs/ /app/docs/
COPY scripts/ /app/scripts/
USER slinky
EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)" || exit 1
CMD ["slinky-lab", "serve", "--host", "0.0.0.0", "--port", "8000"]
