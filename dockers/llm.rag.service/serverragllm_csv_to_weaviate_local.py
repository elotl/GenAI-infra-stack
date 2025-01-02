# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "faiss-cpu",
#     "fastapi",
#     "langchain-community",
#     "langchain-huggingface",
#     "openai",
#     "uvicorn",
#     "weaviate",
#     "langchain_weaviate",
# ]
# ///

import os
import sys
import uvicorn

from functools import partial
from typing import Union

import click
from fastapi import FastAPI
from openai import OpenAI

from common import get_answer_with_settings


def setup(
        relevant_docs: int,
        llm_server_url:str,
        model_id: str,
        max_tokens: int,
        model_temperature: float,
):
    app = FastAPI()

    # TODO: move to imports
    import weaviate
    from langchain_weaviate.vectorstores import WeaviateVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings

    # TODO: pass through settings or params
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    weaviate_client = weaviate.connect_to_local()
    vectorstore  = WeaviateVectorStore(
       client=weaviate_client,
       index_name="my_custom_index",
       text_key="text",
       embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": relevant_docs,
            "alpha": 0.5,
        }
    )
    print("Created Vector DB retriever successfully. \n")

    print("Creating an OpenAI client to the hosted model at URL: ", llm_server_url)
    try:
        client = OpenAI(base_url=llm_server_url, api_key="na")
    except Exception as e:
        print("Error creating client:", e)
        sys.exit(1)

    get_answer = partial(
        get_answer_with_settings,
        retriever=retriever,
        client=client,
        model_id=model_id,
        max_tokens=max_tokens,
        model_temperature=model_temperature,
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
MAX_TOKENS_DEFAULT = 64
MODEL_TEMPERATURE_DEFAULT = 0.01

relevant_docs = os.getenv("RELEVANT_DOCS", RELEVANT_DOCS_DEFAULT)
llm_server_url = os.getenv("LLM_SERVER_URL", "http://localhost:11434/v1")
model_id = os.getenv("MODEL_ID", "llama2")
max_tokens = int(os.getenv("MAX_TOKENS", MAX_TOKENS_DEFAULT))
model_temperature = float(os.getenv("MODEL_TEMPERATURE", MODEL_TEMPERATURE_DEFAULT))

app = setup(relevant_docs, llm_server_url, model_id, max_tokens, model_temperature)


@click.command()
@click.option("--host", default="127.0.0.1", help="Host for the FastAPI server (default: 127.0.0.1)")
@click.option("--port", type=int, default=8000, help="Port for the FastAPI server (default: 8000)")
def run(host, port):
    # Serve the app using Uvicorn
    uvicorn.run("serverragllm_csv_to_weaviate_local:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    run()
