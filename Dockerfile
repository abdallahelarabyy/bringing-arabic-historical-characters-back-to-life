# Arabic Historical Characters - production image
#   docker compose up -d --build        then open http://localhost:8000
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    TOKENIZERS_PARALLELISM=false

# libgomp: OpenMP runtime used by PyTorch and FAISS
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Dependencies (cached unless requirements.txt changes). CPU-only PyTorch keeps the image small.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# 2. Bake the Arabic embedding model into the image, so the container starts without downloading it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('silma-ai/silma-embedding-matryoshka-v0.1')"

# 3. Application code, website and ready-built indexes
COPY backend/ backend/
COPY frontend/ frontend/
COPY data/faiss_indices/ data/faiss_indices/

# Run as a non-root user; only the runtime data folders are writable
RUN useradd --create-home --uid 1000 app \
    && mkdir -p data/conversations \
    && chown -R app:app data /opt/hf-cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if b'ready' in urllib.request.urlopen('http://127.0.0.1:8000/health').read() else 1)"

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
