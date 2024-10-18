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
  libmagic1 \
  file \
  libssl-dev ca-certificates make \
  git python3-pip && \
  rm -rf /var/lib/apt/lists/*

# Install Python dependencies (including python-magic)
RUN pip install python-magic

WORKDIR /createvectordb

COPY createvectordb.py .
COPY pyproject.toml .

RUN --mount=type=cache,target=/root/.cache/pip \
  pip3 install -v --no-cache-dir \
  "langchain" "langchain_community" "langchain_huggingface" "unstructured" "sentence-transformers" "faiss-cpu" "boto3" "lxml" "bs4" "python-magic" && \
  pip3 install --no-cache-dir -e .

CMD ["python", "createvectordb.py"]
