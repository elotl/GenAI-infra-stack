# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base-container

# Automatically set by buildx
ARG TARGETPLATFORM

ENV DEBIAN_FRONTEND=noninteractive

# Install only essential dependencies, clean up after install to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3-pip && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /simple_chat

# Copy application code and requirements.txt for dependency installation
COPY simple_chat.py .
COPY requirements.txt .

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
  pip3 install --upgrade pip && \
  pip3 install --no-cache-dir -r requirements.txt

# Expose the Gradio port
EXPOSE 7860

# Start the application
CMD ["python", "simple_chat.py"]