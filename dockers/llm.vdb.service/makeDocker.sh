#!/usr/bin/env bash
set -e

#set -x

CREATE_VECTOR_DB_REPO=$1
CREATE_VECTOR_DB_TAG=$2

echo ""
echo "Building docker for vectordb creation"
docker buildx build --platform=linux/amd64 --load -f ./Dockerfile -t ${CREATE_VECTOR_DB_REPO}:${CREATE_VECTOR_DB_TAG} .
docker push ${CREATE_VECTOR_DB_REPO}:${CREATE_VECTOR_DB_TAG}
