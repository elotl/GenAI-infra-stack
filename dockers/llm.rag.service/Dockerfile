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
  libssl-dev ca-certificates make \
  git python3-pip && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /serveragllm

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies in separate layers
RUN pip3 install --no-cache-dir \
    "openai" \
    "langchain" \
    "langchain_community" \
    "langchain_huggingface"

RUN pip3 install --no-cache-dir \
    "unstructured" \
    "sentence-transformers" \
    "faiss-cpu"

RUN pip3 install --no-cache-dir \
    "fastapi" \
    "boto3"

RUN pip3 install --no-cache-dir \
    "uvicorn[standard]" \
    "weaviate-client" \
    "langchain_weaviate"

RUN pip3 install --no-cache-dir \
    "langchain-community" \
    "pandas" \
    "sqlalchemy"  \
    "langchain-openai"  \
    "pandas"

COPY __init__.py .
COPY proxy_app.py .
COPY serveragllm.py .
COPY serverragllm_jira_cvs_local.py .
COPY serverragllm_csv_to_weaviate_local.py .
COPY common.py .
COPY pyproject.toml .

EXPOSE 8000

CMD ["python", "proxy_app.py"]
