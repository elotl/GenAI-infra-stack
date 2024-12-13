# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "langchain",
#     "langchain-community",
#     "langchain-huggingface",
#     "pgvector",
#     "psycopg2-binary",
# ]
# ///

# Source: dockers/llm.vdb.service/createvectordb.py

import click

from common import create_vectordb_pgvector


@click.command()
@click.argument("local_tmp_dir", type=click.Path(exists=True))
@click.argument(
    "embedding_model_name", default="sentence-transformers/all-MiniLM-L6-v2"
)
def run(local_tmp_dir: str, embedding_model_name: str):
    # TODO: pass through settings or params
    connection_string = "postgresql+psycopg2://testuser:testpwd@localhost:5555/vectordb"
    collection_name = "jira_tickets"

    db = create_vectordb_pgvector(
        local_tmp_dir,
        embedding_model_name,
        connection_string,
        collection_name,
    )

    print(f"Data saved to {db.collection_name}")


if __name__ == "__main__":
    run()
