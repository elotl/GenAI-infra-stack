import click
import os
import pickle
import sys

from common import create_vectordb
from config import try_load_settings
from s3_utils import download_files_from_s3, save_file_to_s3


@click.command()
@click.option("--env_file", type=click.Path(exists=True), help="Path to the environment file")
def run(env_file: str):
    s3_settings, _ = try_load_settings(env_file)

    if not s3_settings:
        raise "Missing s3 settings"
    
    config = s3_settings

    # Initialize vectorstore and create pickle representation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_tmp_dir = "/tmp/" + config.vectordb_name

    # create this temp dir if it does not already exist
    if not os.path.exists(local_tmp_dir):
        os.makedirs(local_tmp_dir)

    # download text docs from S3 bucket + folder (vectordb_s3_input_dir/arg) into this tmp local directory
    num_files = download_files_from_s3(
        config.s3_bucket_name, config.s3_dir_name, local_tmp_dir
    )
    print(
        f"Number of files downloaded is {num_files}, local tmp dir is {local_tmp_dir}"
    )

    vectorstore = create_vectordb(
        local_tmp_dir,
        config.embedding_model_name,
        config.embedding_chunk_size,
        config.embedding_chunk_overlap,
    )

    pickle_byte_obj = pickle.dumps(vectorstore)

    # Persist vectorstore to S3 bucket vectorstores
    save_file_to_s3(pickle_byte_obj, config.s3_bucket_name, config.vectordb_name)
    
    print("Uploaded vectordb to", config.s3_bucket_name, config.vectordb_name)
    sys.exit(0)


if __name__ == "__main__":
    run()
