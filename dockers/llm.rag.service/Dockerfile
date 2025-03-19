# syntax=docker/dockerfile:1
# Adapted from: https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM python:3.11-slim AS base-container

# Automatically set by buildx
ARG TARGETPLATFORM

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

RUN apt-get update && apt-get install -y \
  build-essential \
  ca-certificates \
  ccache \
  curl \
  libssl-dev ca-certificates make \
  git python3-pip && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /serveragllm

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY __init__.py .
COPY proxy_app.py .
COPY serveragllm.py .
COPY logging_config.py .
COPY serverragllm_csv_to_weaviate_local.py .
COPY common.py .
COPY pyproject.toml .

EXPOSE 8000

CMD ["python", "proxy_app.py"]
