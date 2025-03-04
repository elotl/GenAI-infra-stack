import os
import pickle
import sys
from typing import Any, Dict, List, Union
from enum import Enum

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import FastAPI
from openai import OpenAI

import logging
from logging.handlers import TimedRotatingFileHandler

from common import get_answer_with_settings
from common import get_sql_answer

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
                2. Do not add the provided context to the generated answer.
                3. Include relevant technical details when present or provide a summary of the comments in the ticket.
                4. Include the submitter, assignee and collaborator for a ticket when this info is available.
                5. If the question cannot be answered with the given context, please say so and do not attempt to provide an answer.
                6. Do not create new questions related to the given question, instead answer only the provided question.
                7. Provide a clear, direct and factual answer.
                """ 

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SearchType(Enum):
    SQL = 1
    VECTOR = 2

def str_to_int(value, name):
    try:
        # Convert the environment variable (or default) to an integer
        int_value = int(value)
    except ValueError:
        logging.error(
            f"Error: Value {name} could not be converted to an integer value, please check."
        )
        sys.exit(1)
    return int_value


def str_to_float(value, name):
    try:
        # Convert the environment variable (or default) to an integer
        float_value = float(value)
    except ValueError:
        logging.error(
            f"Error: Value {name} could not be converted to an float value, please check."
        )
        sys.exit(1)
    return float_value


########
# Fetch RAG context for question, form prompt from context and question, and call model
def get_answer(question: Union[str, None]):

    logging.info(f"In get_answer, received question: {question}")

    model_id = os.environ.get("MODEL_ID")
    if model_id == "" or model_id is None:
        model_id = MODEL_ID_DEFAULT
    logging.info(f"Using Model ID: {model_id}")

    model_temperature = os.environ.get("MODEL_TEMPERATURE")
    if model_temperature == "" or model_temperature is None:
        model_temperature = MODEL_TEMPERATURE_DEFAULT
    else:
        model_temperature = str_to_float(model_temperature, "MODEL_TEMPERATURE")
    logging.info(f"Using Model Temperature: {model_temperature}")

    max_tokens = os.environ.get("MAX_TOKENS")
    if max_tokens == "" or max_tokens is None:
        max_tokens = MAX_TOKENS_DEFAULT
    else:
        max_tokens = str_to_int(max_tokens, "MAX_TOKENS")
    logging.info(f"Using Max Tokens: {max_tokens}")

    relevant_docs = os.environ.get("RELEVANT_DOCS")
    if relevant_docs == "" or relevant_docs is None:
        relevant_docs = RELEVANT_DOCS_DEFAULT
    else:
        relevant_docs = str_to_int(relevant_docs, "RELEVANT_DOCS")
    logging.info(f"Using top-k search from Vector DB, k: {relevant_docs}")

    is_json_mode = os.environ.get("IS_JSON_MODE", "False") == "True"
    logging.info(f"Using is_json_mode: {is_json_mode}")

    system_prompt = os.environ.get("SYSTEM_PROMPT")
    if system_prompt  == "" or system_prompt is None:
        system_prompt  = SYSTEM_PROMPT_DEFAULT
    logging.info(f"Using System Prompt: {system_prompt}")


    # TODO: Add question classification block

    search_type_config = os.environ.get("SEARCH_TYPE", "SQL")
    logging.info(f"Using search type config: {search_type_config}")

    match search_type_config:
        case "SQL":
            search_type = SearchType.SQL            
        case "VECTOR":
            search_type = SearchType.VECTOR          

    logging.info(f"Using search type: {search_type}")

    if is_json_mode:
        logging.info("Sending query to the LLM (JSON mode)...")

        match search_type:
            case SearchType.SQL: 

                return get_sql_answer(
                    question,
                    model_id,
                    max_tokens,
                    model_temperature,
                    llm_server_url,
                )

            case SearchType.VECTOR: 

                logging.info("Retrieving docs relevant to the input question")
                docs = retriever.invoke(input=question)
                num_of_docs = len(docs)
                logging.info(f"Number of relevant documents retrieved and that will be used as context for query: {num_of_docs}")

                # Retriever configuration parameters reference:
                # https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.as_retriever
                retriever = vectorstore.as_retriever(search_kwargs={"k": relevant_docs})
                print("Created Vector DB retriever successfully. \n")

                print("Creating an OpenAI client to the hosted model at URL: ", llm_server_url)
                try:
                    client = OpenAI(base_url=llm_server_url, api_key="na")
                except Exception as e:
                    print("Error creating client:", e)
                    sys.exit(1)

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
        logging.info("Sending query to the LLM (non JSON mode)...")
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
        logging.info(f"Received answer (from non JSON processing): {answer}")
        return answer

########
# Setup logging

# When running locally: export RAGLLM_LOGS_PATH=logs/ragllm.log
log_file_path = os.getenv("RAGLLM_LOGS_PATH") or "/app/logs/ragllm.log"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure log directory exists
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
         # Log to file, rotate every 1H and store files from last 24 hrs * 7 days files == 168H data
        TimedRotatingFileHandler(log_file_path, when='h', interval=1, backupCount=168),
        logging.StreamHandler()             # Also log to console
    ]
)

########
# Get connection to LLM server
model_llm_server_url = os.environ.get("MODEL_LLM_SERVER_URL")
if model_llm_server_url is None:
    model_llm_server_url = (
        "http://llm-model-serve-serve-svc.default.svc.cluster.local:8000"
    )
    logging.info(
        f"Setting environment variable MODEL_LLM_SERVER_URL to default value: {model_llm_server_url}")
llm_server_url = model_llm_server_url + "/v1"

logging.info(f"Creating an OpenAI client to the hosted model at URL: {llm_server_url}")
try:
    client = OpenAI(base_url=llm_server_url, api_key="na")
except Exception as e:
    logging.error(f"Error creating client to self-hosted LLM: {e}")
    sys.exit(1)

########
# Load vectorstore and get retriever for it

# get env vars needed to access Vector DB
vectordb_bucket = os.environ.get("VECTOR_DB_S3_BUCKET")
if vectordb_bucket is None:
    logging.error("Please set environment variable VECTOR_DB_S3_BUCKET")
    sys.exit(1)
logging.info(f"Using Vector DB S3 bucket: {vectordb_bucket}")

vectordb_key = os.environ.get("VECTOR_DB_S3_FILE")
if vectordb_key is None:
    logging.error("Please set environment variable VECTOR_DB_S3_FILE")
    sys.exit(1)
logging.info(f"Using Vector DB S3 file: {vectordb_key}")

relevant_docs = os.environ.get("RELEVANT_DOCS")
if relevant_docs == "" or relevant_docs is None:
    relevant_docs = RELEVANT_DOCS_DEFAULT
else:
    relevant_docs = str_to_int(relevant_docs, "RELEVANT_DOCS")
logging.info(f"Using top-k search from Vector DB, {relevant_docs}")

# Use s3 client to read in vector store
s3_client = boto3.client("s3")
response = None
try:
    response = s3_client.get_object(Bucket=vectordb_bucket, Key=vectordb_key)
except ClientError as e:
    logging.error(
        f"Error accessing object, {vectordb_key} in bucket, {vectordb_bucket}, err: {e}"
    )
    sys.exit(1)
body = response["Body"].read()

logging.info("Loading Vector DB...")
# needs prereq packages: sentence_transformers and faiss-cpu
vectorstore = pickle.loads(body)

# Retriever configuration parameters reference:
# https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.as_retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": relevant_docs})
logging.info("Created Vector DB retriever successfully.")

# Uncomment to run a local test
#logging.info("Testing with a sample question:")
#get_answer("What's a recent SSH issue customers had?")

########
# Start API service to answer questions
app = FastAPI()


@app.get("/answer/{question}")
def read_item(question: Union[str, None] = None):
    logging.info(f"Received question: {question}")
    answer = get_answer(question)
    logging.info(f"Received answer: {answer}")
    return {"question": question, "answer": answer}
