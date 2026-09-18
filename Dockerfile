FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BALLFORM_REQUIRE_TOKEN=1 \
    BALLFORM_JOBS_DIR=/data/jobs \
    BALLFORM_MODELS_DIR=/data/models \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/ballform
COPY pyproject.toml ./
COPY app ./app
COPY web ./web
# CPU wheels avoid pulling CUDA into a CPU analysis worker.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \
    && useradd --create-home ballform \
    && mkdir -p /data/jobs /data/models \
    && chown -R ballform:ballform /data

USER ballform
EXPOSE 8000
# One process owns the job queue; do not add multiple Uvicorn workers.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log"]
