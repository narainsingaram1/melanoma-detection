# CPU inference service: FastAPI + ONNX Runtime (no PyTorch in the image).
#   docker build -t melanoma-api .
#   docker run -p 8000:8000 melanoma-api
# Models: either build with models/ present (python -m scripts.build_release ...),
# or set HF_MODEL_REPO=<user>/<repo> at runtime to download them from the HF Hub.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    ORT_THREADS=2 \
    PORT=8000

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/__init__.py src/constants.py src/inference.py src/
COPY api/ api/
COPY models/ models/

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
