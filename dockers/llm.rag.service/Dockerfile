# syntax=docker/dockerfile:1
# Adapted from: https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM python:3.11-slim AS base-container
# syntax=docker/dockerfile-upstream:master
# FROM python:3.9-slim AS base-container

# Automatically set by buildx
ARG TARGETPLATFORM

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

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

# Install dependencies in separate layers
RUN pip3 install --no-cache-dir \
    "openai" \
    "langchain" \
    "langchain_community" \
    "langchain_huggingface" \
    "unstructured" \
    "sentence-transformers" \
    "faiss-cpu" \
    "fastapi" \
    "boto3" \
    "weaviate-client" \
    "langchain_weaviate" \
    "langchain-community" \
    "pandas" \
    "sqlalchemy"

RUN pip3 install --no-cache-dir \
    "langchain-openai"  \
    "opentelemetry-exporter-otlp" \
    "arize-phoenix==8.19.0" \
    "openinference-instrumentation-openai" \
    "uvicorn" \
    "tiktoken" \
    "nest-asyncio" \
    "joblib==1.4.2" \
    "pandas==2.2.3" \
    "scikit-learn==1.3.2" \
    "numpy==1.26.2"

COPY __init__.py .
COPY proxy_app.py .
COPY serveragllm.py .
COPY logging_config.py .
COPY serverragllm_csv_to_weaviate_local.py .
COPY serverragllm_jira_cvs_local.py .
COPY common.py .
COPY pyproject.toml .

RUN pip3 install --no-cache-dir \
    "httpx<0.28"

RUN pip3 install --no-cache-dir \
    "langchain-huggingface" \
    "openai>=1"

# Install the local package
RUN pip3 install -e .

EXPOSE 8000

CMD ["python", "proxy_app.py"]
