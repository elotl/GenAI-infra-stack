import os
import sys
from functools import partial
from typing import Union
from logging_config import logger

import click
import uvicorn
from fastapi import FastAPI

import phoenix as px
from phoenix.otel import register
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor

import weaviate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

from common import get_answer_with_settings_with_weaviate_filter

SYSTEM_PROMPT_DEFAULT = """You are a specialized support ticket assistant. Format your responses following these rules:
                1. Answer the provided question only using the provided context.
                2. Do not add the provided context to the generated answer.
                3. Include relevant technical details when present or provide a summary of the comments in the ticket.
                4. Include the submitter, assignee and collaborator for a ticket when this info is available.
                5. If the question cannot be answered with the given context, please say so and do not attempt to provide an answer.
                6. Do not create new questions related to the given question, instead answer only the provided question.
                7. Provide a clear, direct and factual answer.
                """

def setup_phoenix(phoenix_svc_url: str):

    logger.info("Setting up Phoenix (LLM ops tool) tracer \n")
    tracer_provider = register(
        project_name="qa-in-a-box",
        endpoint=phoenix_svc_url,
        #auto_instrument=True,
    )

    logger.info("Starting instrumentation of all OpenAI API calls\n")
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

def setup(
    relevant_docs: int,
    llm_server_url: str,
    model_id: str,
    max_tokens: int,
    model_temperature: float,
    weaviate_url: str,
    weaviate_grpc_url: str,
    weaviate_index: str,
    embedding_model_name: str,
    sql_search_db_and_model_path: str,
    alpha: float,
    max_context_length: int,
    sql_ticket_source: str,
):
    app = FastAPI()

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    weaviate_client = weaviate.connect_to_custom(
        http_host=weaviate_url.split(":")[0],
        http_port=int(weaviate_url.split(":")[1]),
        http_secure=False,
        grpc_host=weaviate_grpc_url.split(":")[0],
        grpc_port=int(weaviate_grpc_url.split(":")[1]),
        grpc_secure=False,
    )

    vectorstore = WeaviateVectorStore(
        client=weaviate_client,
        index_name=weaviate_index,
        text_key="text",
        embedding=embeddings,
    )

    if not llm_server_url.endswith("/v1"):
        llm_server_url = llm_server_url + "/v1"
    logger.info(
        f"Creating an OpenAI client to the hosted model at URL: {llm_server_url}"
    )
    try:
        client = OpenAI(base_url=llm_server_url, api_key="na")
    except Exception as e:
        logger.error(f"Error creating client: {e}")
        sys.exit(1)

    get_answer = partial(
        get_answer_with_settings_with_weaviate_filter,
        vectorstore=vectorstore,
        client=client,
        model_id=model_id,
        max_tokens=max_tokens,
        model_temperature=model_temperature,
        system_prompt=SYSTEM_PROMPT_DEFAULT,
        relevant_docs=relevant_docs,
        llm_server_url=llm_server_url,
        sql_search_db_and_model_path=sql_search_db_and_model_path,
        alpha=alpha,
        max_context_length=max_context_length,
        sql_ticket_source=sql_ticket_source,
    )

    @app.get("/answer/{question}")
    def read_item(question: Union[str, None] = None):
        logger.info(f"Received question: {question}")
        answer = get_answer(question)
        return {"question": question, "answer": answer}

    return app


MICROSOFT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MOSAICML_MODEL_ID = "mosaicml/mpt-7b-chat"
RELEVANT_DOCS_DEFAULT = 2
MAX_TOKENS_DEFAULT = 256
MODEL_TEMPERATURE_DEFAULT = 0.01
SQL_SEARCH_DB_AND_MODEL_PATH_DEFAULT = "/app/db/"
WEAVIATE_HYBRID_ALPHA_DEFAULT = 0.5
MODEL_MAX_CONTEXT_LEN = 8192

relevant_docs = int(os.getenv("RELEVANT_DOCS", RELEVANT_DOCS_DEFAULT))

# llm_server_url = os.getenv("MODEL_LLM_SERVER_URL", "http://localhost:11434/v1")
# model_id = os.getenv("MODEL_ID", "llama2")

llm_server_url = os.getenv("MODEL_LLM_SERVER_URL", "http://localhost:9000/v1")
# model_id = os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
model_id = os.getenv("MODEL_ID", "rubra-ai/Phi-3-mini-128k-instruct")

max_tokens = int(os.getenv("MAX_TOKENS", MAX_TOKENS_DEFAULT))
model_temperature = float(os.getenv("MODEL_TEMPERATURE", MODEL_TEMPERATURE_DEFAULT))

weaviate_url = os.getenv("WEAVIATE_URI_WITH_PORT", "localhost:8080")
weaviate_grpc_url = os.getenv("WEAVIATE_GRPC_URI_WITH_PORT", "localhost:50051")
weaviate_index = os.getenv("WEAVIATE_INDEX_NAME", "my_custom_index")
alpha = float(
    os.getenv("WEAVIATE_HYBRID_ALPHA", WEAVIATE_HYBRID_ALPHA_DEFAULT)
)

PHOENIX_SVC_URL_DEFAULT="http://phoenix.phoenix.svc.cluster.local:6006/v1/traces"
# When running this app locally (outside of k8s)
# setup PHOENIX_SVC_URL = http://localhost:6006/v1/traces 
phoenix_svc_url = os.getenv("PHOENIX_SVC_URL", PHOENIX_SVC_URL_DEFAULT)

embedding_model_name = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
# embedding_model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

sql_search_db_and_model_path = os.getenv(
    "SQL_SEARCH_DB_AND_MODEL_PATH", SQL_SEARCH_DB_AND_MODEL_PATH_DEFAULT
)

max_context_length = int(os.getenv("MODEL_MAX_CONTEXT_LEN", MODEL_MAX_CONTEXT_LEN))

sql_ticket_source = os.getenv("SQL_TICKET_SOURCE", "https://support.com/tickets/")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = setup(
    relevant_docs,
    llm_server_url,
    model_id,
    max_tokens,
    model_temperature,
    weaviate_url,
    weaviate_grpc_url,
    weaviate_index,
    embedding_model_name,
    sql_search_db_and_model_path,
    alpha,
    max_context_length,
    sql_ticket_source,
)

# setup observability stack
setup_phoenix(phoenix_svc_url)

@click.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host for the FastAPI server (default: 127.0.0.1)",
)
@click.option(
    "--port", type=int, default=8000, help="Port for the FastAPI server (default: 8000)"
)
def run(host, port):
    # Serve the app using Uvicorn
    uvicorn.run(
        "serverragllm_csv_to_weaviate_local:app", host=host, port=port, reload=True
    )


if __name__ == "__main__":
    run()
