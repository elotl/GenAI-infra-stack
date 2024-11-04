#!/usr/bin/env bash
set -e

#set -x

SERVE_RAG_LLM_REPO=$1
SERVE_RAG_LLM_TAG=$2

echo ""
echo "Building docker for rag+llm service"
docker buildx build --platform=linux/amd64 --load -f ./Dockerfile -t ${SERVE_RAG_LLM_REPO}:${SERVE_RAG_LLM_TAG} .
docker push ${SERVE_RAG_LLM_REPO}:${SERVE_RAG_LLM_TAG}
