FROM python:3.12.3-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY code/ code/
RUN cd code && python prepare_data.py && chmod -R a+rwX /app

WORKDIR /app/code
ENTRYPOINT ["bash", "run_all.sh"]
