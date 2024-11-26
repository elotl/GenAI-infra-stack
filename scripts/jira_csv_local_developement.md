# How to run the full process for jira csv locally

## Prepare data

Go to `GenAI-infra-stack/scripts` create venv and install deps
```shell
uv run process_jira_tickets.py jira_elotl.csv jira_config.ini output.jsonl
```

## Create vector store
Go to `GenAI-infra-stack/dockers/llm.vdb.service`

```shell
uv run createvectordb_jira_csv_local.py ../../scripts pickled.obj
```

## Run rag app from local pickled.obj

Run Ollama Open Api compatible local model
https://ollama.com/blog/openai-compatibility

Go to `GenAI-infra-stack/dockers/llm.rag.service`

```shell
uv run serverragllm_jira_cvs_local.py --file-path ../llm.vdb.service/pickled.obj
```
