# syntax=docker/dockerfile-upstream:master
# Adapted from: https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM python:3.11-slim as base-container

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

COPY serveragllm.py .
COPY pyproject.toml .

RUN --mount=type=cache,target=/root/.cache/pip \
  pip3 install -v --no-cache-dir \
  "openai" "langchain" "langchain_community" "langchain_huggingface" "unstructured" "sentence-transformers" "faiss-cpu" "uvicorn[standard]" "fastapi" "boto3" && \
  pip3 install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "serveragllm:app", "--host", "0.0.0.0", "--port", "8000"]
