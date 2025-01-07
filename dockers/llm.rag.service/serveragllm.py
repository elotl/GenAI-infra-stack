import os
import pickle
import sys
from typing import Any, Dict, List, Union

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import FastAPI
from openai import OpenAI
from common import get_answer_with_settings

import phoenix as px
from phoenix.otel import register
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from openinference.instrumentation.langchain import LangChainInstrumentor

########
# Setup model name and query template parameters
MICROSOFT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MOSAICML_MODEL_ID = "mosaicml/mpt-7b-chat"
RELEVANT_DOCS_DEFAULT = 2
MAX_TOKENS_DEFAULT = 64
MODEL_TEMPERATURE_DEFAULT = 0.01
MODEL_ID_DEFAULT = MOSAICML_MODEL_ID

#template = """Answer the question based only on the following context:
#{context}
#
#Question: {question}
#"""
#os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

    # setup RAG LLM parameters
    # common.setup_rag_llm_config()

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
            True,
        )
    else:
        return get_answer_with_settings(
            question,
            retriever,
            client,
            model_id,
            max_tokens,
            model_temperature,
            False,
        )
########


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

# Setup Phoenix
phoenix_svc_url = "http://phoenix.phoenix.svc.cluster.local:6006"

print("Setting up Phoenix (LLM ops tool) tracer \n")
tracer_provider = register(
    project_name="default",
    endpoint=phoenix_svc_url,
)
LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)

print("Setting up Phoenix's configuration: \n")
queries_df = get_qa_with_reference(px.Client(endpoint=phoenix_svc_url))
retrieved_documents_df = get_retrieved_documents(px.Client(endpoint=phoenix_svc_url)) 

# Uncomment to run a local test
# print("Testing with a sample question:")
# get_answer("who are you?")

########
# Start API service to answer questions
app = FastAPI()


@app.get("/answer/{question}")
def read_item(question: Union[str, None] = None):
    print(f"Received question: {question}")
    answer = get_answer(question)
    return {"question": question, "answer": answer}
