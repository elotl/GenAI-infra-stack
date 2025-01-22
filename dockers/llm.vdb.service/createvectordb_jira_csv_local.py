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


@click.command()
@click.argument("local_tmp_dir", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.argument(
    "embedding_model_name", default="sentence-transformers/all-MiniLM-L6-v2"
)
def run(local_tmp_dir: str, output_file: str, embedding_model_name: str):
    vectorstore = create_vectordb(local_tmp_dir, embedding_model_name)

    pickle_byte_obj = pickle.dumps(vectorstore)

    with open(output_file, "wb") as file:
        file.write(pickle_byte_obj)

    print(f"Pickle byte object saved to {output_file}")


if __name__ == "__main__":
    run()
