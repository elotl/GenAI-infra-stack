# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "faiss-cpu",
#     "fastapi",
#     "langchain-community",
#     "langchain-huggingface",
#     "openai",
#     "uvicorn",
#     "weaviate-client",
#     "langchain_weaviate",
# ]
# ///

import os
import sys
from functools import partial
from typing import Union
from logging_config import logger

import click
import uvicorn
from fastapi import FastAPI
from openai import OpenAI

import weaviate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore


from common import get_answer_with_settings_with_weaviate_filter
from config import HybridSearchSettings, LlmSettings, WeaviateSettings


def setup(
    sql_search_db_and_model_path: str,
    max_context_length: int,
    sql_ticket_source: str,
):
    app = FastAPI()

    weaviate_settings = WeaviateSettings()
    hybrid_search_settings = HybridSearchSettings()
    llm_settings = LlmSettings()

    embeddings = HuggingFaceEmbeddings(model_name=weaviate_settings.embedding_model_name)

    weaviate_client = weaviate.connect_to_custom(
        http_host=weaviate_settings.get_weaviate_uri(),
        http_port=weaviate_settings.get_weaviate_port(),
        http_secure=False,
        grpc_host=weaviate_settings.get_weaviate_grpc_uri(),
        grpc_port=weaviate_settings.get_weaviate_grpc_port(),
        grpc_secure=False,
    )

    vectorstore = WeaviateVectorStore(
        client=weaviate_client,
        index_name=weaviate_settings.weaviate_index_name,
        text_key="text",
        embedding=embeddings,
    )

    logger.info(
        f"Creating an OpenAI client to the hosted model at URL: {llm_settings.llm_server_url}"
    )
    try:
        client = OpenAI(base_url=llm_settings.llm_server_url, api_key="na")
    except Exception as e:
        logger.error(f"Error creating client: {e}")
        sys.exit(1)

    get_answer = partial(
        get_answer_with_settings_with_weaviate_filter,
        vectorstore=vectorstore,
        client=client,
        system_prompt=hybrid_search_settings.system_prompt,
        relevant_docs=hybrid_search_settings.relevant_docs,
        sql_search_db_and_model_path=sql_search_db_and_model_path,
        alpha=hybrid_search_settings.alpha,
        max_context_length=max_context_length,
        sql_ticket_source=sql_ticket_source,
        llm_settings=llm_settings,
    )

    @app.get("/answer/{question}")
    def read_item(question: Union[str, None] = None):
        logger.info(f"Received question: {question}")
        answer = get_answer(question)
        return {"question": question, "answer": answer}

    return app


SQL_SEARCH_DB_AND_MODEL_PATH_DEFAULT = "/app/db/"
MODEL_MAX_CONTEXT_LEN = 8192

sql_search_db_and_model_path = os.getenv(
    "SQL_SEARCH_DB_AND_MODEL_PATH", SQL_SEARCH_DB_AND_MODEL_PATH_DEFAULT
)

max_context_length = int(os.getenv("MODEL_MAX_CONTEXT_LEN", MODEL_MAX_CONTEXT_LEN))

sql_ticket_source = os.getenv("SQL_TICKET_SOURCE", "https://zendesk.com/api/v2/tickets/")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = setup(
    sql_search_db_and_model_path,
    max_context_length,
    sql_ticket_source,
)


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
