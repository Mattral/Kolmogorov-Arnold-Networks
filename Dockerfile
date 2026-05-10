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
    && pip install --no-cache-dir .[api,onnx]
