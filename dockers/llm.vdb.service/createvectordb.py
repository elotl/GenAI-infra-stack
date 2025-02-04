import click
import sys

from config import try_load_settings
from service import LocalDirDbCreationService, S3VectorDbCreationService


@click.command()
@click.option("--env_file", type=click.Path(exists=True), help="Path to the environment file")
def run(env_file: str):
    s3_settings, local_settings = try_load_settings(env_file)

    if s3_settings:
        service = S3VectorDbCreationService(s3_settings)
        service.create()

    elif local_settings:
        service = LocalDirDbCreationService(local_settings)
        service.create()

    else:
        # TODO: not really needed, error will be thrown earlier
        raise "Missing config"

    sys.exit(0)


if __name__ == "__main__":
    run()
