# Dockerfile for a Hugging Face Space (Docker SDK).
# The Space only needs this file and the README.md next to it: the image clones
# the GitHub repository at build time, so there is nothing else to upload.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860
WORKDIR /home/user/app

# Change REPO_REF to force a rebuild after pushing new commits to GitHub.
ARG REPO_URL=https://github.com/jagonzibwa/Joshua-Agonzibwa-Summative-assignment---MLOP.git
ARG REPO_REF=main
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} .

RUN pip install --no-cache-dir --user -r requirements-serve.txt

# Fetch the dataset if the images are not in the repository.
RUN python -m src.preprocessing --acquire --if-missing && rm -rf data/raw \
    || echo "Dataset download skipped: retraining will use uploaded images only."
RUN test -f data/insights.json || python -m src.insights || true

EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
