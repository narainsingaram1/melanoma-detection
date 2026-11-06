"""
FastAPI inference service (ONNX Runtime, CPU).

  uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /health    liveness + loaded model
  GET  /model     model card: classes, calibration, OOD threshold, test metrics
  POST /predict   multipart image upload -> calibrated class probabilities,
                  risk group, and out-of-distribution flag

Not a medical device. Uploaded images are processed in memory and never stored.
"""
import io
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from src.inference import OnnxLesionClassifier, ensure_model_dir

MAX_UPLOAD_BYTES = int(os.environ.get('MAX_UPLOAD_MB', '10')) * 1024 * 1024
RATE_LIMIT_PER_MIN = int(os.environ.get('RATE_LIMIT_PER_MIN', '30'))
ALLOWED_TYPES = {'image/jpeg', 'image/png', 'image/webp'}
DISCLAIMER = ('Research demo, not a medical device. Do not use for diagnosis; '
              'consult a dermatologist about any concerning skin lesion.')

@asynccontextmanager
async def lifespan(_app):
    if os.environ.get('SKIP_WARMUP') != '1':
        classifier()  # load the model before accepting traffic
    yield


app = FastAPI(title='Melanoma Detection API', version='2.0.0', lifespan=lifespan,
              description='EfficientNet-B0 skin-lesion classifier (HAM10000, 7 classes) with '
                          'calibrated probabilities and out-of-distribution rejection.')
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
                   allow_methods=['GET', 'POST'], allow_headers=['*'])

_classifier = None
_hits = defaultdict(deque)


def classifier() -> OnnxLesionClassifier:
    global _classifier
    if _classifier is None:
        _classifier = OnnxLesionClassifier(ensure_model_dir())
    return _classifier


def _rate_limit(request: Request):
    """Simple in-memory sliding-window limiter (per client IP, per process)."""
    ip = request.headers.get('x-forwarded-for', request.client.host if request.client else 'unknown').split(',')[0]
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, 'rate limit exceeded, try again in a minute')
    q.append(now)


@app.get('/health')
def health():
    c = classifier()
    return {'status': 'ok', 'model': c.model_file, 'variant': c.post.info['variant']}


@app.get('/model')
def model_card():
    info = classifier().post.info
    return {k: info[k] for k in ('architecture', 'variant', 'seed', 'classes', 'calibration', 'ood',
                                 'test_metrics_uncalibrated', 'serving')} | {'disclaimer': DISCLAIMER}


@app.post('/predict')
async def predict(request: Request, file: UploadFile = File(...)):
    _rate_limit(request)
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f'unsupported content type {file.content_type}; use JPEG, PNG or WebP')
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f'file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB')
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        result = classifier().predict(img)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, 'could not decode image')

    if not result['in_distribution']:
        result['message'] = ('This does not look like a dermoscopic skin-lesion image, so no '
                             'prediction is shown. Upload a close-up dermoscopy photo.')
    result['disclaimer'] = DISCLAIMER
    return result
