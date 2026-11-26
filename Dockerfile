# Vibration anomaly batcher — CPU inference image.

# ---------------------------------------------------------------------------
# Stage 1: generate the example ONNX model (kept out of the runtime image)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS modelgen

WORKDIR /build
RUN pip install --no-cache-dir "onnx>=1.15,<2.0" "numpy>=1.24,<3.0"
COPY scripts ./scripts
RUN python scripts/make_model.py /out/anomaly_model.onnx

# ---------------------------------------------------------------------------
# Stage 2: CPU inference service
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    ORT_INTRA_OP_THREADS=1

# onnxruntime CPU wheels dynamically link libstdc++ and may dlopen libgomp;
# install the runtime libraries explicitly so the slim image can import it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 libgcc-s1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY main.py server.py ./
COPY scripts ./scripts
COPY docker-entrypoint.sh ./docker-entrypoint.sh
COPY README.md ./
COPY --from=modelgen /out/anomaly_model.onnx ./models/anomaly_model.onnx

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app \
    && chmod +x /app/docker-entrypoint.sh
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2); sys.exit(0 if r.status == 200 else 1)"

ENTRYPOINT ["sh", "/app/docker-entrypoint.sh"]
