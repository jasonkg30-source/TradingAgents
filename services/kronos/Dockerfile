# Kronos forecast sidecar — CPU image, model weights pulled from HuggingFace at first request.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KRONOS_PATH=/opt/Kronos \
    KRONOS_MODEL=NeoQuasar/Kronos-small \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0+cpu" \
 && pip install -r /tmp/requirements.txt

# Upstream repo is not packaged; vendor it and put it on PYTHONPATH.
RUN git clone --depth 1 https://github.com/shiyu-coder/Kronos.git /opt/Kronos

WORKDIR /srv
COPY app.py /srv/app.py

ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 120
