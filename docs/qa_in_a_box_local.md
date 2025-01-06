# Run Question-Answering-in-a-box on your local laptop 

# Overview
1. [Setup an OSS model](#ossmodel)
2. [Create the Vector DB](#createvectordb)
3. [Run RAG LLM serving app](#ragllm)
4. [Fourth Example](#fourth-examplehttpwwwfourthexamplecom)


The following steps apply to a MacOS laptop.

## Setup an OSS model <a name="ossmodel"></a>

We will use Ollama 2 model locally.
So export this env var:

```shell
export MODEL_ID="llama2"
```

Check that the Ollama model started up successfully:

Via your browser:
http://localhost:11434/

Via a commandline query:

```shell
% curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3.2",            
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "How many products has Nestle sold?"
            }
        ]
    }'

```



## Create vector DB <a name="createvectordb"></a>
uv run createvectordb_jira_csv_local.py /Users/selvik/stuff/dev/elotl-chat-in-a-box/GenAI-infra-stack/scripts/output_files rag-mini-vectorstore.txt



## Run RAG LLM serving app <a name="ragllm"></a>

```shell
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

```shell
uv run  serverragllm_jira_cvs_local.py
```



## Try a sample question <a name="samplequestion"></a>

% curl -X GET "http://localhost:8000/answer/what%20are%20the%20two%20types%20of%20elephants%20in%20Africa?"
{"question":"what are the two types of elephants in Africa","answer":{"answer":"\nThe most relevant ticket references for this question are:\n\n* ENG-175: QA Verified - Type: Bug (panic: interface conversion: interface {} is types.Image, not *types.Image)\n* ENG-93: To Do - Type: Bug (title: Luna: use cloudprovider.NodeType.Architecture with isArmInstanceType() functions)\n\nBased on the provided content, it seems that there are two types of elephants in Africa:\n\n1. African bush elephant (Loxodonta africana)","relevant_tickets":["ENG-175","ENG-93"],"sources":["https://elotl.atlassian.net/browse/ENG-175","https://elotl.atlassian.net/browse/ENG-93"]}}% 

## Start LLM obervability tool, Phoenix 

Option 1:

You can uncomment the two lines in the local Python script to start the RAG LLM serving script:
```shell
/GenAI-infra-stack/dockers/llm.rag.service/serverragllm_jira_cvs_local.py
```

```shell
# Uncomment the following 2 lines if you would like to bring
# up a local Phoenix app
# print("Starting LLM Ops tool, Phoenix locally")
# session = px.launch_app()
```

Please Ignore this error message, if the port 6006 is unused:

```shell
ERROR:    Application startup failed. Exiting.
💥 Phoenix failed to start. Please try again (making sure that port 6006 is not occupied by another process) or file an issue with us at https://github.com/Arize-ai/phoenix
```

Option 2: You can start Phoenix as a docker container:

```shell
docker run -p 6006:6006 -p 4317:4317 -i -t arizephoenix/phoenix:latest    
```

## Check LLM observability <a name="llmobserve"></a>

Access the [Phoenix UI](http://localhost:6006) to view traces for the question above. 

## Phoenix can be started on a K8s cluster along with the RAG LLM app, as follows:

In this case, start the Phoenix server like this:

```shell
% cd GenAI-infra-stack/dockers/llm.phoenix.service
% kubectl apply -f  phoenix.yaml  
```

Port-forward to allow the RAG LLM app to send trces to Phoenix

```shell
kubectl port-forward -n phoenix svc/phoenix 6006:6006
```


