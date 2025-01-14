import os
import pickle
import sys
from typing import Any, Dict, List, Union

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import FastAPI
from openai import OpenAI

from common import get_answer_with_settings

########
# Setup model name and query template parameters
MICROSOFT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MOSAICML_MODEL_ID = "mosaicml/mpt-7b-chat"
RELEVANT_DOCS_DEFAULT = 2
MAX_TOKENS_DEFAULT = 64
MODEL_TEMPERATURE_DEFAULT = 0.01
MODEL_ID_DEFAULT = MOSAICML_MODEL_ID
SYSTEM_PROMPT_DEFAULT = """You are a specialized support ticket assistant. Format your responses following these rules:
                1. Answer the provided question only using the provided context.
                2. Provide a clear, direct and factual answer
                3. Include relevant technical details when present
                4. If the information is outdated, mention when it was last updated
                """

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def str_to_int(value, name):
    try:
        # Convert the environment variable (or default) to an integer
        int_value = int(value)
    except ValueError:
        print(
            f"Error: Value {name} could not be converted to an integer value, please check."
        )
        sys.exit(1)
    return int_value


def str_to_float(value, name):
    try:
        # Convert the environment variable (or default) to an integer
        float_value = float(value)
    except ValueError:
        print(
            f"Error: Value {name} could not be converted to an float value, please check."
        )
        sys.exit(1)
    return float_value


########
# Fetch RAG context for question, form prompt from context and question, and call model
def get_answer(question: Union[str, None]):

    print("Received question: ", question)

    model_id = os.environ.get("MODEL_ID")
    if model_id == "" or model_id is None:
        model_id = MODEL_ID_DEFAULT
    print("Using Model ID: ", model_id)

    model_temperature = os.environ.get("MODEL_TEMPERATURE")
    if model_temperature == "" or model_temperature is None:
        model_temperature = MODEL_TEMPERATURE_DEFAULT
    else:
        model_temperature = str_to_float(model_temperature, "MODEL_TEMPERATURE")
    print("Using Model Temperature: ", model_temperature)

    max_tokens = os.environ.get("MAX_TOKENS")
    if max_tokens == "" or max_tokens is None:
        max_tokens = MAX_TOKENS_DEFAULT
    else:
        max_tokens = str_to_int(max_tokens, "MAX_TOKENS")
    print("Using Max Tokens: ", max_tokens)

    relevant_docs = os.environ.get("RELEVANT_DOCS")
    if relevant_docs == "" or relevant_docs is None:
        relevant_docs = RELEVANT_DOCS_DEFAULT
    else:
        relevant_docs = str_to_int(relevant_docs, "RELEVANT_DOCS")
    print("Using top-k search from Vector DB, k: ", relevant_docs)

    is_json_mode = os.environ.get("IS_JSON_MODE", "False") == "True"

    system_prompt = os.environ.get("SYSTEM_PROMPT")
    if system_prompt  == "" or system_prompt is None:
        system_prompt  = SYSTEM_PROMPT_DEFAULT
    print("Using System Prompt: ", system_prompt)

    # retrieve docs relevant to the input question
    docs = retriever.invoke(input=question)
    print(
        "Number of relevant documents retrieved and that will be used as context for query: ",
        len(docs),
    )

    if is_json_mode:
        return get_answer_with_settings(
            question,
            retriever,
            client,
            model_id,
            max_tokens,
            model_temperature,
            system_prompt,
        )
    else:
        print("Sending query to the LLM...")
        # concatenate relevant docs retrieved to be used as context
        allcontext = ""
        for i in range(len(docs)):
            allcontext += docs[i].page_content
        promptstr = template.format(context=allcontext, question=question)

        completions = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": promptstr,
                },
            ],
            max_tokens=max_tokens,
            temperature=model_temperature,
            stream=False,
        )

        answer = completions.choices[0].message.content
        print("Received answer: ", answer)
        return answer


########
# Get connection to LLM server
model_llm_server_url = os.environ.get("MODEL_LLM_SERVER_URL")
if model_llm_server_url is None:
    model_llm_server_url = (
        "http://llm-model-serve-serve-svc.default.svc.cluster.local:8000"
    )
    print(
        "Setting environment variable MODEL_LLM_SERVER_URL to default value: ",
        model_llm_server_url,
    )
llm_server_url = model_llm_server_url + "/v1"

print("Creating an OpenAI client to the hosted model at URL: ", llm_server_url)
try:
    client = OpenAI(base_url=llm_server_url, api_key="na")
except Exception as e:
    print("Error creating client:", e)
    sys.exit(1)

########
# Load vectorstore and get retriever for it

# get env vars needed to access Vector DB
vectordb_bucket = os.environ.get("VECTOR_DB_S3_BUCKET")
print("Using vector DB s3 bucket: ", vectordb_bucket)
if vectordb_bucket is None:
    print("Please set environment variable VECTOR_DB_S3_BUCKET")
    sys.exit(1)
print("Using Vector DB S3 bucket: ", vectordb_bucket)

vectordb_key = os.environ.get("VECTOR_DB_S3_FILE")
print("Using vector DB s3 file containing vector store: ", vectordb_key)
if vectordb_key is None:
    print("Please set environment variable VECTOR_DB_S3_FILE")
    sys.exit(1)
print("Using Vector DB S3 file: ", vectordb_key)

relevant_docs = os.environ.get("RELEVANT_DOCS")
if relevant_docs == "" or relevant_docs is None:
    relevant_docs = RELEVANT_DOCS_DEFAULT
else:
    relevant_docs = str_to_int(relevant_docs, "RELEVANT_DOCS")
print("Using top-k search from Vector DB, k: ", relevant_docs)

# Use s3 client to read in vector store
s3_client = boto3.client("s3")
response = None
try:
    response = s3_client.get_object(Bucket=vectordb_bucket, Key=vectordb_key)
except ClientError as e:
    print(
        f"Error accessing object, {vectordb_key} in bucket, {vectordb_bucket}, err: {e}"
    )
    sys.exit(1)
body = response["Body"].read()

print("Loading Vector DB...\n")
# needs prereq packages: sentence_transformers and faiss-cpu
vectorstore = pickle.loads(body)

# Retriever configuration parameters reference:
# https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.as_retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": relevant_docs})
print("Created Vector DB retriever successfully. \n")

# Uncomment to run a local test
#print("Testing with a sample question:")
#get_answer("What's a recent SSH issue customers had?")

########
# Start API service to answer questions
app = FastAPI()


@app.get("/answer/{question}")
def read_item(question: Union[str, None] = None):
    print(f"Received question: {question}")
    answer = get_answer(question)
    return {"question": question, "answer": answer}
