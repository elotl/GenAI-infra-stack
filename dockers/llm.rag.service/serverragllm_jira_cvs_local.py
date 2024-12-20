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

import json
import os
import pickle
import sys
import uvicorn
import click

from fastapi import FastAPI
from functools import partial
from typing import Union

from openai import OpenAI
#from langchain_openai import OpenAI

from openinference.instrumentation.langchain import LangChainInstrumentor
from common import get_answer_with_settings

import phoenix as px
from phoenix.otel import register
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents

def setup(
        vectore_store_path: str,
        relevant_docs: int,
        llm_server_url:str,
        model_id: str,
        max_tokens: int,
        model_temperature: float,
):
    app = FastAPI()

    # Load the object from the pickle file
    with open(vectore_store_path, "rb") as file:
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

    print("Setting up Phoenix (LLM ops tool) tracer \n")
    tracer_provider = register(
        project_name="default",
        endpoint="http://localhost:6006/v1/traces",
    )

    LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)

    print("Setting up Phoenix's configuration: \n")
    queries_df = get_qa_with_reference(px.Client())
    retrieved_documents_df = get_retrieved_documents(px.Client()) 

    @app.get("/answer/{question}")
    def read_item(question: Union[str, None] = None):
        print(f"Received question: {question}")
        answer = get_answer(question)
        return {"question": question, "answer": answer}

    return app

print("Setting up configuration for RAG LLM")
MICROSOFT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MOSAICML_MODEL_ID = "mosaicml/mpt-7b-chat"
RELEVANT_DOCS_DEFAULT = 2
MAX_TOKENS_DEFAULT = 64
MODEL_TEMPERATURE_DEFAULT = 0.01

vectore_store_path = os.getenv("VECTOR_STORE_PATH")
if not vectore_store_path:
    print("Please provide the pickled vector store path via env var, VECTORE_STORE_PATH")

relevant_docs = int(os.getenv("RELEVANT_DOCS", RELEVANT_DOCS_DEFAULT))
llm_server_url = os.getenv("LLM_SERVER_URL", "http://localhost:11434/v1")
model_id = os.getenv("MODEL_ID", "llama2")
max_tokens = int(os.getenv("MAX_TOKENS", MAX_TOKENS_DEFAULT))
model_temperature = float(os.getenv("MODEL_TEMPERATURE", MODEL_TEMPERATURE_DEFAULT))

# Uncomment the following 2 lines if you would like to bring
# up a local Phoenix app
#print("Starting LLM Ops tool, Phoenix locally")
#session = px.launch_app()

print("Setting up Fast API app \n")
app = setup(vectore_store_path, relevant_docs, llm_server_url, model_id, max_tokens, model_temperature)

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
    uvicorn.run("serverragllm_jira_cvs_local:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    run()
