# Process CSV to json and pass to RAG app

## Preparing the data
Go to `GenAI-infra-stack/scripts` create venv and install deps
```shell
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

set jira_url in `jira_config.ini` and run

```shell
uv run process_jira_tickets.py jira_elotl.csv jira_config.ini output_files
```

upload these files instead of the wiki docs

## Vector store creation
Run vector store creation with 
```shell
export VECTOR_DB_INPUT_ARG="jira-embedding-json"
```

## Rag app
Run rag service with this extra setting
```shell
export IS_JIRA_MODE="True"
```

## Chat UI app
Run chat UI with the same export MODEL_NAMESPACE=... as rag service:
```shell
envsubst < simple-chat.yaml | kubectl apply -f -
```

and port forward to use it:
```shell
kubectl port-forward svc/simple-chat-service 7860:7860
```
