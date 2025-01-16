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

import asyncio
import os
import pickle
import sys
import uvicorn
import time

from functools import partial
from typing import Union

import click
from fastapi import FastAPI
from openai import AsyncOpenAI

from common import async_get_answer_with_settings


def setup(
        file_path: str,
        relevant_docs: int,
        llm_server_url:str,
        model_id: str,
        max_tokens: int,
        model_temperature: float,
):
    app = FastAPI()

    # Load the object from the pickle file
    with open(file_path, "rb") as file:
        print("Loading Vector DB...\n")
        vectorstore = pickle.load(file)

    # Retriever configuration parameters reference:
    # https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.as_retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": relevant_docs})
    print("Created Vector DB retriever successfully. \n")

    print("Creating an AsyncOpenAI client to the hosted model at URL: ", llm_server_url)
    try:
        client = AsyncOpenAI(base_url=llm_server_url, api_key="na")
    except Exception as e:
        print("Error creating client:", e)
        sys.exit(1)

    

    async_get_answer = partial(
        async_get_answer_with_settings,
        retriever=retriever,
        client=client,
        model_id=model_id,
        max_tokens=max_tokens,
        model_temperature=model_temperature,
    )

    @app.get("/answer/{question}")
    async def read_item(question: Union[str, None] = None):
        print(f"Received question: {question}")
        start_time = time.time()
        answer = await async_get_answer(question)
        print("--- Get answer: %s seconds ---" % (time.time() - start_time))
        return {"question": question, "answer": answer}

    return app


MICROSOFT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MOSAICML_MODEL_ID = "mosaicml/mpt-7b-chat"
RELEVANT_DOCS_DEFAULT = 2
MAX_TOKENS_DEFAULT = 64
MODEL_TEMPERATURE_DEFAULT = 0.01


file_path = os.getenv("FILE_PATH")
if not file_path:
    print("Please provide the pickeled vector store path")

relevant_docs = int(os.getenv("RELEVANT_DOCS", RELEVANT_DOCS_DEFAULT))
llm_server_url = os.getenv("LLM_SERVER_URL", "http://localhost:11434/v1")
model_id = os.getenv("MODEL_ID", "llama2")
max_tokens = int(os.getenv("MAX_TOKENS", MAX_TOKENS_DEFAULT))
model_temperature = float(os.getenv("MODEL_TEMPERATURE", MODEL_TEMPERATURE_DEFAULT))

app = setup(file_path, relevant_docs, llm_server_url, model_id, max_tokens, model_temperature)


@click.command()
@click.option("--host", default="127.0.0.1", help="Host for the FastAPI server (default: 127.0.0.1)")
@click.option("--port", type=int, default=8000, help="Port for the FastAPI server (default: 8000)")
def run(host, port):
    # Serve the app using Uvicorn
    uvicorn.run("serverragllm_jira_cvs_local:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    run()
