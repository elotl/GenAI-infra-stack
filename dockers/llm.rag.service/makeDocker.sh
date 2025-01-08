#!/usr/bin/env bash
set -e

#set -x

SERVE_RAG_LLM_REPO=$1
SERVE_RAG_LLM_TAG=$2

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo ""
echo "Building docker for rag+llm service"
docker buildx build --platform=linux/amd64 --load \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t ${SERVE_RAG_LLM_REPO}:${SERVE_RAG_LLM_TAG} \
  "${SCRIPT_DIR}"

# docker push ${SERVE_RAG_LLM_REPO}:${SERVE_RAG_LLM_TAG}
