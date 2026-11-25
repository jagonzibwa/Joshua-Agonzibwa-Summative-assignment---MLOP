# EngageWatch: API + web UI + retraining in one image.
#   docker build -t engagewatch .
#   docker run -p 8000:8000 engagewatch
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=app:app src ./src
COPY --chown=app:app api ./api
COPY --chown=app:app models ./models
COPY --chown=app:app data ./data
RUN chown app:app /app
USER app

# If the images were not part of the build context, download the dataset and
# rebuild data/train and data/test (needed for retraining and /evaluate).
RUN python -m src.preprocessing --acquire --if-missing && rm -rf data/raw \
    || echo "Dataset download skipped: retraining will use uploaded images only."
RUN test -f data/insights.json || python -m src.insights || true

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health',timeout=4)" || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
