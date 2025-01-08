#!/usr/bin/env bash
set -e

#set -x

SIMPLE_RAG_LLM_CHAT_REPO=$1
SIMPLE_RAG_LLM_CHAT_TAG=$2

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )"

echo ""
echo "Building docker for rag chat ui"
docker buildx build --platform=linux/amd64 --load \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t ${SIMPLE_RAG_LLM_CHAT_REPO}:${SIMPLE_RAG_LLM_CHAT_TAG} \
  "${SCRIPT_DIR}"

docker push ${SIMPLE_RAG_LLM_CHAT_REPO}:${SIMPLE_RAG_LLM_CHAT_TAG}
