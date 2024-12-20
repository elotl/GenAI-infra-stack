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

<<<<<<< HEAD
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
    "uvicorn[standard]" \
    "weaviate-client" \
    "langchain_weaviate" \
    "langchain-community" \
    "pandas" \
    "sqlalchemy"  \
    "langchain-openai"  \
    "pandas"

COPY __init__.py .
COPY proxy_app.py .
COPY serveragllm.py .
COPY logging_config.py .
COPY serverragllm_csv_to_weaviate_local.py .
COPY common.py .
COPY pyproject.toml .
=======
# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    curl \
    libssl-dev \
    make \
    git \
    python3-pip \
    python3-dev \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /serveragllm

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies in separate layers
RUN pip3 install --no-cache-dir \
    "openai" \
    "fastapi" \
    "uvicorn[standard]" \
    "boto3"

RUN pip3 install --no-cache-dir \
    "langchain" \
    "langchain_community" \
    "langchain_huggingface"

RUN pip3 install --no-cache-dir \
    "sentence-transformers" \
    "unstructured" \
    "faiss-cpu"

RUN pip3 install --no-cache-dir \
    "arize-phoenix" \
    "openinference-instrumentation-langchain"

RUN pip3 install --no-cache-dir \
    "arize-phoenix[evals]" \
    "tiktoken" \
    "nest-asyncio"

RUN pip3 install --no-cache-dir \
    "httpx<0.28"

RUN pip3 install --no-cache-dir \
    "langchain-openai"

RUN pip3 install --no-cache-dir \
    "langchain-huggingface" \
    "opentelemetry-api" \
    "opentelemetry-instrumentation" \
    "opentelemetry-semantic-conventions" \
    "opentelemetry-exporter-otlp-proto-http" \
    "opentelemetry-sdk" \
    "opentelemetry-exporter-otlp" \
    "openai>=1"

# Copy application files
COPY pyproject.toml .
COPY __init__.py .
COPY common.py .
COPY serveragllm.py .
COPY serverragllm_jira_cvs_local.py .

# Install the local package
RUN pip3 install -e .
>>>>>>> 11be6f1 (Add LLM observability tool Phoenix to QA in a box stack)

EXPOSE 8000

CMD ["python", "proxy_app.py"]
