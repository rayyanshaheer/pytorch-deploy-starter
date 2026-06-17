# Slim CPU image keeps the starter small and free to run on most cloud tiers.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU-only torch/torchvision to avoid pulling large CUDA wheels.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt

COPY app ./app

EXPOSE 8000

# Warm the model cache at build time so the first request is fast.
RUN python -c "from app.model import build_model; build_model()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
