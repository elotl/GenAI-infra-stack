#!/usr/bin/env bash
set -e

#set -x

SIMPLE_RAG_LLM_CHAT_REPO=$1
SIMPLE_RAG_LLM_CHAT_TAG=$2

echo ""
echo "Building docker for rag+llm service"
docker buildx build --platform=linux/amd64 --load -f ./Dockerfile -t ${SIMPLE_RAG_LLM_CHAT_REPO}:${SIMPLE_RAG_LLM_CHAT_TAG} .
docker push ${SIMPLE_RAG_LLM_CHAT_REPO}:${SIMPLE_RAG_LLM_CHAT_TAG}
