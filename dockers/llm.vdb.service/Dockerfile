# syntax=docker/dockerfile:1
# Adapted from: https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM python:3.11-slim AS base-container

# Automatically set by buildx
ARG TARGETPLATFORM

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  ccache \
  curl \
  libmagic1 \
  file \
  libssl-dev ca-certificates make \
  git python3-pip && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /createvectordb

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies in separate layers
RUN pip3 install --no-cache-dir \
    "langchain" \
    "langchain_community" \
    "langchain_huggingface"

RUN pip3 install --no-cache-dir \
    "unstructured" \
    "sentence-transformers" \
    "faiss-cpu"

RUN pip3 install --no-cache-dir \
    "lxml" \
    "bs4"

RUN pip3 install --no-cache-dir \
    "python-magic" \
    "boto3"

COPY __init__.py .
COPY createvectordb.py .
COPY createvectordb_jira_csv_local.py .
COPY common.py .
COPY pyproject.toml .

CMD ["python", "createvectordb.py"]
