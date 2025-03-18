# Run the RAG LLM module locally 

Note: This is a developer workflow and not an end-user or admin workflow.
This allows quick testing of new changes to the RAG LLM Service.

## Enable RAG-LLM service to access weaviate locally

Export these variables to locally access Weaviate running within a k8s cluster. An 
alternative to this would be to bring up Weaviate with docker compose to run it locally.

```bash
export WEAVIATE_URI_WITH_PORT="localhost:8080"
export WEAVIATE_GRPC_URI_WITH_PORT="localhost:50051"
```

## Enable RAG-LLM app to access LLM host locally

```bash
export MODEL_LLM_SERVER_URL=http://localhost:9000
```

## Enable RAG-LLM app to access SQL DB and Question router models locally:

```bash
export SQL_SEARCH_DB_AND_MODEL_PATH="/tmp/db/"
```

## Start the RAG LLM service

```bash
% uv run serverragllm_csv_to_weaviate_local.py                                                     
```
