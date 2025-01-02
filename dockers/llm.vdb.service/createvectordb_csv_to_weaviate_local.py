# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "langchain",
#     "langchain-community",
#     "langchain-huggingface",
#     "langchain_postgres",
#     "weaviate",
#     "langchain_weaviate",
# ]
# ///

# Source: dockers/llm.vdb.service/createvectordb.py

import click

from common import create_vectordb_local_weaviate


@click.command()
@click.argument("local_tmp_dir", type=click.Path(exists=True))
@click.argument(
    "embedding_model_name", default="sentence-transformers/all-MiniLM-L6-v2"
)
def run(local_tmp_dir: str, embedding_model_name: str):
    # TODO: pass through settings or params

    db = create_vectordb_local_weaviate(
        local_tmp_dir,
        embedding_model_name,
    )

    print(f"Data saved to local weaviate")


if __name__ == "__main__":
    run()