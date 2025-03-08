# Text to SQL Setup

## Preparation

1. Convert structured data in CSV format to SQL DB and make it available in an S3 bucket within a prefix folder.

2. Use synthetic questions in CSV format to train a classification model. The result of 
this process are two pkl files. Save these files also in the same S3 bucket and prefix folder
as in Step 1.


## Setting up SQL + Vector search locally

1. Export local variables such as 

- LLM server url at 9000
- Setup location of SQL DB and question classification models

```sh	
source some_location/exports-local.sh
```

1. Run the LLM on k8s and portforward:

```sh	
 kubectl port-forward svc/llm-model-serve-serve-svc 9000:8000
```	

2. Run the local version of the SQL + hybrid search app:

```sh	
llm.rag.service % source .venv/bin/activate   
 uv run serverragllm_csv_to_weaviate_local.py
```	

Wait till applicaiton is loaded and you see this message:
```sh	
INFO:     Application startup complete.
```

3. Try a question about your user data: 

```sh	
cd /GenAI-infra-stack/scripts/query
```	

```sh	
% python  query_private_data.py
Type your query here: How many tickets are there? 
```
