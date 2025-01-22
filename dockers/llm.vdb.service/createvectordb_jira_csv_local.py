# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "faiss-cpu",
#     "langchain-community",
#     "langchain-huggingface",
#     "pydantic_settings",
# ]
# ///

# Source: dockers/llm.vdb.service/createvectordb.py

import click
import pickle

from common import create_vectordb
from config import LocalSettings


@click.command()
@click.option("--env_file", type=click.Path(exists=True), help="Path to the environment file")
def run(env_file: str):
    if env_file:
        config = LocalSettings(_env_file=env_file)
    else:
        config = LocalSettings()

    vectorstore = create_vectordb(
        config.local_directory, 
        config.embedding_model_name,
        chunk_size=config.embedding_chunk_size,
        chunk_overlap=config.embedding_chunk_overlap,
    )

    pickle_byte_obj = pickle.dumps(vectorstore)

    with open(config.output_filename, "wb") as file:
        file.write(pickle_byte_obj)

    print(f"Pickle byte object saved to {config.output_filename}")


if __name__ == "__main__":
    run()
