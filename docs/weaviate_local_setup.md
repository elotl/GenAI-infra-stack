# Running app locally with weaviate

## Run weaviate
```shell
 docker compose up weaviate
```

## Run create database
Create an .my-env file and put inside 
```shell
LOCAL_DIRECTORY="../../scripts/zendesk_dataprep_output"
OUTPUT_FILENAME=""

WEAVIATE_URI_WITH_PORT="localhost:8080"
WEAVIATE_GRPC_URI_WITH_PORT="localhost:50051"
WEAVIATE_INDEX_NAME="my_custom_index"

EMBEDDING_CHUNK_SIZE=4000
EMBEDDING_CHUNK_OVERLAP=100

EMBEDDING_MODEL_NAME=sentence-transformers/multi-qa-mpnet-base-dot-v1
```

run:
```shell
uv run createvectordb.py --env_file .my-env
```

## Run serve rag app
```shell
uv run serverragllm_csv_to_weaviate_local.py
```