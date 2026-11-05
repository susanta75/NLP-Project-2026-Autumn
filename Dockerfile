# Environment for Project 7 (HaluEval QA). Build and run with:  bash run_docker.sh
FROM python:3.12.3-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf-cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY code/ code/
# Download HaluEval and draw the 200-question sample now, so the container runs offline.
RUN cd code && python prepare_data.py && chmod -R a+rwX /app

WORKDIR /app/code
ENTRYPOINT ["bash", "run_all.sh"]
