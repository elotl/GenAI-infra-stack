import os
import pickle
import sys
from enum import Enum
from typing import Any, Dict, List, Union

from logging_config import logger

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import FastAPI
from openai import OpenAI
from common import get_answer_with_settings, get_sql_answer
from common import setup_rag_llm_config

import phoenix as px
from phoenix.otel import register
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from openinference.instrumentation.langchain import LangChainInstrumentor

########
# Setup model name and query template parameters
MICROSOFT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MOSAICML_MODEL_ID = "mosaicml/mpt-7b-chat"
RELEVANT_DOCS_DEFAULT = 2
MAX_TOKENS_DEFAULT = 256
MODEL_TEMPERATURE_DEFAULT = 0.01
MODEL_ID_DEFAULT = MOSAICML_MODEL_ID
SQL_SEARCH_DB_AND_MODEL_PATH_DEFAULT = "/app/db/"


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
        logger.error(
            f"Error: Value {name} could not be converted to an integer value, please check."
        )
        sys.exit(1)
    return int_value


def str_to_float(value, name):
    try:
        # Convert the environment variable (or default) to an integer
        float_value = float(value)
    except ValueError:
        logger.error(
            f"Error: Value {name} could not be converted to an float value, please check."
        )
        sys.exit(1)
    return float_value


########
# Fetch RAG context for question, form prompt from context and question, and call model
def get_answer(question: Union[str, None]):

    logger.info(f"In get_answer, received question: {question}")

    # setup RAG LLM parameters
    # common.setup_rag_llm_config()

    model_id = os.environ.get("MODEL_ID")
    if model_id == "" or model_id is None:
        model_id = MODEL_ID_DEFAULT
    logger.info(f"Using Model ID: {model_id}")

    model_temperature = os.environ.get("MODEL_TEMPERATURE")
    if model_temperature == "" or model_temperature is None:
        model_temperature = MODEL_TEMPERATURE_DEFAULT
    else:
        model_temperature = str_to_float(model_temperature, "MODEL_TEMPERATURE")
    logger.info(f"Using Model Temperature: {model_temperature}")

    max_tokens = os.environ.get("MAX_TOKENS")
    if max_tokens == "" or max_tokens is None:
        max_tokens = MAX_TOKENS_DEFAULT
    else:
        max_tokens = str_to_int(max_tokens, "MAX_TOKENS")
    logger.info(f"Using Max Tokens: {max_tokens}")

    relevant_docs = os.environ.get("RELEVANT_DOCS")
    if relevant_docs == "" or relevant_docs is None:
        relevant_docs = RELEVANT_DOCS_DEFAULT
    else:
        relevant_docs = str_to_int(relevant_docs, "RELEVANT_DOCS")
    logger.info(f"Using top-k search from Vector DB, k: {relevant_docs}")

    is_json_mode = os.environ.get("IS_JSON_MODE", "False") == "True"
    logger.info(f"Using is_json_mode: {is_json_mode}")

    system_prompt = os.environ.get("SYSTEM_PROMPT")
    if system_prompt == "" or system_prompt is None:
        system_prompt = SYSTEM_PROMPT_DEFAULT
    logger.info(f"Using System Prompt: {system_prompt}")

    # TODO: Add question classification block

    search_type_config = os.environ.get("SEARCH_TYPE", "SQL")
    logger.info(f"Using search type config: {search_type_config}")

    match search_type_config:
        case "SQL":
            search_type = SearchType.SQL
        case "VECTOR":
            search_type = SearchType.VECTOR

    logger.info(f"Using search type: {search_type}")

    sql_search_db_and_model_path = os.getenv( "SQL_SEARCH_DB_AND_MODEL_PATH", SQL_SEARCH_DB_AND_MODEL_PATH_DEFAULT)
    logger.info(f"Using sql db and model path: {sql_search_db_and_model_path}")

    if is_json_mode:
        logger.info("Sending query to the LLM (JSON mode)...")

        match search_type:
            case SearchType.SQL:
                logger.info("Handling search type: SQL")

                return get_sql_answer(
                    question,
                    model_id,
                    max_tokens,
                    model_temperature,
                    llm_server_url,
                    sql_search_db_and_model_path,
                )

            case SearchType.VECTOR:
                logger.info("Handling search type: VECTOR")

                logger.info("Retrieving docs relevant to the input question")
                docs = retriever.invoke(input=question)
                num_of_docs = len(docs)
                logger.info(
                    f"Number of relevant documents retrieved and that will be used as context for query: {num_of_docs}"
                )

                # Retriever configuration parameters reference:
                # https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.as_retriever
                retriever = vectorstore.as_retriever(search_kwargs={"k": relevant_docs})
                logger.info("Created Vector DB retriever successfully. \n")

                logger.info(
                    "Creating an OpenAI client to the hosted model at URL: ",
                    llm_server_url,
                )
                try:
                    client = OpenAI(base_url=llm_server_url, api_key="n/a")
                except Exception as e:
                    logger.error("Error creating client:", e)
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
        logger.info("Sending query to the LLM (non JSON mode)...")

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
        logger.info(f"Received answer (from non JSON processing): {answer}")
        return answer

# Get connection to LLM server
model_llm_server_url = os.environ.get("MODEL_LLM_SERVER_URL")
if model_llm_server_url is None:
    model_llm_server_url = (
        "http://llm-model-serve-serve-svc.default.svc.cluster.local:8000"
    )
    logger.info(
        f"Setting environment variable MODEL_LLM_SERVER_URL to default value: {model_llm_server_url}"
    )
llm_server_url = model_llm_server_url + "/v1"

logger.info(f"Creating an OpenAI client to the hosted model at URL: {llm_server_url}")
try:
    client = OpenAI(base_url=llm_server_url, api_key="na")
except Exception as e:
    logger.error(f"Error creating client to self-hosted LLM: {e}")
    sys.exit(1)

########
# Load vectorstore and get retriever for it

# get env vars needed to access Vector DB
vectordb_bucket = os.environ.get("VECTOR_DB_S3_BUCKET")
if vectordb_bucket is None:
    logger.error("Please set environment variable VECTOR_DB_S3_BUCKET")
    sys.exit(1)
logger.info(f"Using Vector DB S3 bucket: {vectordb_bucket}")

vectordb_key = os.environ.get("VECTOR_DB_S3_FILE")
if vectordb_key is None:
    logger.error("Please set environment variable VECTOR_DB_S3_FILE")
    sys.exit(1)
logger.info(f"Using Vector DB S3 file: {vectordb_key}")

relevant_docs = os.environ.get("RELEVANT_DOCS")
if relevant_docs == "" or relevant_docs is None:
    relevant_docs = RELEVANT_DOCS_DEFAULT
else:
    relevant_docs = str_to_int(relevant_docs, "RELEVANT_DOCS")
logger.info(f"Using top-k search from Vector DB, {relevant_docs}")

# Use s3 client to read in vector store
s3_client = boto3.client("s3")
response = None
try:
    response = s3_client.get_object(Bucket=vectordb_bucket, Key=vectordb_key)
except ClientError as e:
    logger.error(
        f"Error accessing object, {vectordb_key} in bucket, {vectordb_bucket}, err: {e}"
    )
    sys.exit(1)
body = response["Body"].read()

logger.info("Loading Vector DB...")
# needs prereq packages: sentence_transformers and faiss-cpu
vectorstore = pickle.loads(body)

# Retriever configuration parameters reference:
# https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.as_retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": relevant_docs})
logger.info("Created Vector DB retriever successfully.")

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
# logger.info("Testing with a sample question:")
# get_answer("What's a recent SSH issue customers had?")

########
# Start API service to answer questions
app = FastAPI()

@app.get("/answer/{question}")
def read_item(question: Union[str, None] = None):
    logger.info(f"Received question: {question}")
    answer = get_answer(question)
    logger.info(f"Received answer: {answer}")
    return {"question": question, "answer": answer}
