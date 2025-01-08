#!/usr/bin/env bash
set -e

#set -x

CREATE_VECTOR_DB_REPO=$1
CREATE_VECTOR_DB_TAG=$2

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo ""
echo "Building docker for vectordb creation"
docker buildx build --platform=linux/amd64 --load \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t ${CREATE_VECTOR_DB_REPO}:${CREATE_VECTOR_DB_TAG} \
  "${SCRIPT_DIR}"

docker push ${CREATE_VECTOR_DB_REPO}:${CREATE_VECTOR_DB_TAG}
