import click
import logging
import sys

from config import (
    try_load_settings,
    try_load_weaviate_settings,
)
from service import (
    LocalDirDbCreationService,
    LocalDirWeaviateDbCreationService,
    S3WeaviateDbCreationService,
    S3VectorDbCreationService,
)


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


@click.command()
@click.option("--env_file", type=click.Path(exists=True), help="Path to the environment file")
def run(env_file: str):
    s3_settings, local_settings = try_load_settings(env_file)
    weaviate_settings = try_load_weaviate_settings(env_file)

    if s3_settings:
        if weaviate_settings.is_set():
            logging.info("---> S3WeaviateDbCreationService")
            service = S3WeaviateDbCreationService(s3_settings, weaviate_settings)
        else:
            logging.info("---> S3VectorDbCreationService")
            service = S3VectorDbCreationService(s3_settings)

        service.create()

    elif local_settings:
        if weaviate_settings.is_set():
            logging.info("---> LocalDirWeaviateDbCreationService")
            service = LocalDirWeaviateDbCreationService(local_settings, weaviate_settings)
        else:
            logging.info("---> S3VectorDbCreationService")
            service = S3VectorDbCreationService(local_settings)

        service.create()

    else:
        # TODO: not really needed, error will be thrown earlier
        raise "Missing config"

    sys.exit(0)


if __name__ == "__main__":
    run()
