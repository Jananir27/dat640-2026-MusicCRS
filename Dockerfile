FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/cache/huggingface \
    HF_DATASETS_CACHE=/app/cache/huggingface/datasets \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

# PyTorch's CPU wheels keep the default laptop setup independent of CUDA.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install -r /tmp/requirements.txt

COPY musiccrs ./musiccrs
COPY retrieval ./retrieval

# The saved G1 tuning artifact is required; do not build an image that would
# only fail later when the backend starts.
RUN test -s /app/retrieval/artifacts/final_retrieval.json

ENV MUSICCRS_HOST=0.0.0.0 \
    MUSICCRS_PORT=5000 \
    MUSICCRS_RETRIEVAL_ARTIFACT=/app/retrieval/artifacts/final_retrieval.json

EXPOSE 5000

CMD ["python", "-m", "musiccrs.musiccrs"]
