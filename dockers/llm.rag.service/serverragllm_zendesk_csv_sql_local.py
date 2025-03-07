# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "faiss-cpu",
#     "fastapi",
#     "langchain-community",
#     "langchain-huggingface",
#     "openai",
#     "uvicorn",
# ]
# ///

import os
import pickle
import sys
import logging
import uvicorn

from functools import partial
from typing import Union

import click
from fastapi import FastAPI
from openai import OpenAI

from common import get_answer_with_settings
from common import get_sql_answer
from common import question_router
from common import SearchType

def setup(
        file_path: str,
        relevant_docs: int,
        llm_server_url:str,
        model_id: str,
        max_tokens: int,
        model_temperature: float,
):
    app = FastAPI()

    # TO DO: Add question classification block
    # search_type = question_router(question) 

    # For now, hard-coding question type to aggregation
    search_type = SearchType.SQL

    match search_type:
        case SearchType.SQL: 
            logging.info("Handling search type: SQL")

            get_answer = partial(
                get_sql_answer,
                model_id=model_id,
                max_tokens=max_tokens,
                model_temperature=model_temperature,
                llm_server_url=llm_server_url,
            )
        case SearchType.VECTOR: 
            logging.info("Handling search type: VECTOR")

            # Load the object from the pickle file
            with open(file_path, "rb") as file:
                print("Loading Vector DB...\n")
                vectorstore = pickle.load(file)

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

            jira_system_prompt = """You are a specialized support ticket assistant. Format your responses following these rules:
                        1. Answer the provided question only using the provided context.
                        2. Provide a clear, direct and factual answer
                        3. Include relevant technical details when present
                        4. If the information is outdated, mention when it was last updated
                        """

            get_answer = partial(
                get_answer_with_settings,
                retriever=retriever,
                client=client,
                model_id=model_id,
                max_tokens=max_tokens,
                model_temperature=model_temperature,
                system_prompt=jira_system_prompt,
            )


    @app.get("/answer/{question}")
    def read_item(question: Union[str, None] = None):
        print(f"Received question: {question}")
        answer = get_answer(question)
        return {"question": question, "answer": answer}

    return app


MICROSOFT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MOSAICML_MODEL_ID = "mosaicml/mpt-7b-chat"
RELEVANT_DOCS_DEFAULT = 2
MAX_TOKENS_DEFAULT = 350
MODEL_TEMPERATURE_DEFAULT = 0.00


file_path = os.getenv("FILE_PATH")
if not file_path:
    print("Please provide the DB file path")

logging.info("Setting LLM setup parameters like LLM server URL, model id, model temperature...")
relevant_docs = int(os.getenv("RELEVANT_DOCS", RELEVANT_DOCS_DEFAULT))

# LLM server URL if using ollama hosting locally
# llm_server_url = os.getenv("LLM_SERVER_URL", "http://localhost:11434/v1")
# model_id = os.getenv("MODEL_ID", "llama2")

# LLM server URL if using k8s elotl hosting + port-forwarding 
llm_server_url = os.getenv("LLM_SERVER_URL", "http://localhost:8080/v1")
model_id = os.getenv("MODEL_ID", "rubra-ai/Phi-3-mini-128k-instruct")

max_tokens = int(os.getenv("MAX_TOKENS", MAX_TOKENS_DEFAULT))
model_temperature = float(os.getenv("MODEL_TEMPERATURE", MODEL_TEMPERATURE_DEFAULT))

app = setup(file_path, relevant_docs, llm_server_url, model_id, max_tokens, model_temperature)

@click.command()
@click.option("--host", default="127.0.0.1", help="Host for the FastAPI     server (default: 127.0.0.1)")
@click.option("--port", type=int, default=8000, help="Port for the FastAPI server (default: 8000)")
def run(host, port):
    # Serve the app using Uvicorn
    uvicorn.run("serverragllm_zendesk_csv_sql_local:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    run()
