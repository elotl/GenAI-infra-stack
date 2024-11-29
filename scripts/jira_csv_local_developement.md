# How to run the full process for jira csv locally

## Make sure You have 
- [UV installed](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama cli installed](https://ollama.com/download)

## Prepare data

Go to `GenAI-infra-stack/scripts` create venv and install deps
```shell
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
set jira_url in jira_config.ini

```shell
uv run process_jira_tickets.py jira_elotl.csv jira_config.ini output_files
```

## Create vector store
Go to `GenAI-infra-stack/dockers/llm.vdb.service`

```shell
uv run createvectordb_jira_csv_local.py ../../scripts/output_files pickled.obj
```

## Run rag app from local pickled.obj

Run Ollama Open Api compatible local model
https://ollama.com/blog/openai-compatibility

Go to `GenAI-infra-stack/dockers/llm.rag.service`

```shell
export FILE_PATH="../llm.vdb.service/pickled.obj"

uv run serverragllm_jira_cvs_local.py
```

## Test setup
```shell
curl "http://127.0.0.1:8000/answer/How%20to%20install%20luna%20?"
```

## Run UI
Go to `GenAI-infra-stack/dockers/llm.chatui.service` create venv and install requirements.

```shell
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

```shell
export RAG_LLM_QUERY_URL="http://127.0.0.1:8000"
 
uv run simple_chat.py
```