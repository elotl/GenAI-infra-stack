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

import pickle
import sys
import uvicorn

from functools import partial
from typing import Any, Dict, List, Union

import click
from fastapi import FastAPI
from openai import OpenAI


def format_context(results: List[Dict[str, Any]]) -> str:
    """Format search results into context for the LLM"""
    context_parts = []

    for result in results:
        # TODO: make metadata keyes configurable
        ticket_metadata = result.metadata
        ticket_content = result.page_content

        context_parts.append(
            f"Key: {ticket_metadata['key']} | Status: {ticket_metadata['status']} - "
            f"Type: {ticket_metadata['type']}\n"
            f"Content: {ticket_content}...\n"
        )

    return "\n\n".join(context_parts)


def get_answer_with_settings(question, retriever, client, model_id, max_tokens, model_temperature):
    SYSTEM_PROMPT = """You are a specialized Jira ticket assistant. Format your responses following these rules:
                1. Start with the most relevant ticket references
                2. Provide a clear, direct answer
                3. Include relevant technical details when present
                4. Mention related tickets if they provide additional context
                5. If the information is outdated, mention when it was last updated
                """

    docs = retriever.invoke(input=question)
    print(
        "Number of relevant documents retrieved and that will be used as context for query: ",
        len(docs),
    )

    context = format_context(docs)

    print("Sending query to the LLM...")
    completions = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        max_tokens=max_tokens,
        temperature=model_temperature,
        stream=False,
    )

    answer = {
        "answer": completions.choices[0].message.content,
        "relevant_tickets": [r.metadata["key"] for r in docs],
    }
    print("Received answer: ", answer)
    return answer


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


@click.command()
@click.option("--file-path", required=True)
@click.option("--relevant-docs", default=RELEVANT_DOCS_DEFAULT)
@click.option("--llm-server-url", default="http://localhost:11434/v1")
@click.option("--model-id", default="llama2")
@click.option("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT)
@click.option("--model-temperature", type=float, default=MODEL_TEMPERATURE_DEFAULT)
@click.option("--host", default="127.0.0.1", help="Host for the FastAPI server (default: 127.0.0.1)")
@click.option("--port", type=int, default=8000, help="Port for the FastAPI server (default: 8000)")
def run(file_path, relevant_docs, llm_server_url, model_id, max_tokens, model_temperature, host, port):
    # Create the FastAPI app
    app = setup(file_path, relevant_docs, llm_server_url, model_id, max_tokens, model_temperature)

    breakpoint()
    # Serve the app using Uvicorn
    uvicorn.run(app, host=host, port=port, reload=True)


if __name__ == "__main__":
    run()
