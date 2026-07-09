FROM python:3.11-slim

# System deps for h5py + tf2onnx
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user up-front so all subsequent layers honour it.
RUN groupadd --system kanx && useradd --system --gid kanx --uid 1000 --home-dir /app kanx

WORKDIR /app

# Install Python deps first to maximise Docker layer cache. We install the
# package via pyproject.toml's `api` extra so FastAPI / uvicorn / pydantic
# come in, without bleeding into the core `pip install kanx`.
COPY pyproject.toml ./
COPY README.md ./README.md
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[api,onnx,hub]

# Make the runtime directory writeable by the non-root user (for the
# /app/checkpoints volume mount, the model registry, etc.).
RUN mkdir -p /app/checkpoints && chown -R kanx:kanx /app
USER kanx

ENV PYTHONPATH=/app/src:/app \
    KANX_CONFIG=/app/configs/default.yaml \
    KANX_CHECKPOINT=/app/checkpoints/kanx_model.keras \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
