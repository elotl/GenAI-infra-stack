import click
import os
import pytest
import s3fs
import subprocess

from botocore.session import Session
from moto.moto_server.threaded_moto_server import ThreadedMotoServer
from s3fs.core import S3FileSystem

from createvectordb import run


def test_create_faiss_vector_db_using_local_files():
    ctx = click.Context(run)
    try:
        ctx.forward(run, env_file="test_data/.env_local")
    except SystemExit as e:
        assert e.code == 0

    assert os.path.exists("test_data/output/output_pickled.obj")

    if os.path.exists("test_data/output/output_pickled.obj"):
        os.remove("test_data/output/output_pickled.obj")


@pytest.fixture(scope="module")
def standalone_environment():
    # Start the standalone environment before tests
    try:
        subprocess.run(["bash", "standalone_embed.sh", "start"], check=True)
        yield
    finally:
        # Stop the standalone environment after tests, even if tests fail
        subprocess.run(["bash", "standalone_embed.sh", "stop"], check=True)
        subprocess.run(["bash", "standalone_embed.sh", "delete"], check=True)


def test_create_milvus_vector_db_using_local_files(standalone_environment):
    ctx = click.Context(run)
    try:
        ctx.forward(run, env_file="test_data/.env_local_milvus")
    except SystemExit as e:
        assert e.code == 0


@pytest.fixture(scope="module")
def s3_base():
    # writable local S3 system
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=5555)
    server.start()
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test-secret-key"
    os.environ["AWS_ACCESS_KEY_ID"] = "test-access-key"
    os.environ["TEST_FAKE_S3"] = "true"
    os.environ.pop("AWS_PROFILE", None)

    print("server up")
    yield
    print("moto done")
    server.stop()


def upload_to_s3(s3_client, bucket_name, local_dir, s3_prefix):
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_key = os.path.join(s3_prefix, os.path.relpath(local_file_path, local_dir)).replace("\\", "/")
            with open(local_file_path, "rb") as f:
                s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=f)


MOCK_BUCKET_NAME = "test_bucket"


@pytest.fixture()
def mock_s3_client(s3_base):
    session = Session()
    client = session.create_client("s3", endpoint_url="http://127.0.0.1:5555/")
    client.create_bucket(
        Bucket=MOCK_BUCKET_NAME,
        ACL="public-read",
        CreateBucketConfiguration={
            'LocationConstraint': "us-east-2",  # TODO: make sure this is the same as local default
        },
    )

    S3FileSystem.clear_instance_cache()
    s3 = S3FileSystem(anon=False, client_kwargs={"endpoint_url": "http://127.0.0.1:5555/"})
    s3.invalidate_cache()

    yield client


def test_create_faiss_vector_db_using_s3_files(mock_s3_client):
    upload_to_s3(mock_s3_client, MOCK_BUCKET_NAME, "test_data/input", "s3-input-dir")

    ctx = click.Context(run)
    try:
        ctx.forward(run, env_file="test_data/.env_s3")
    except SystemExit as e:
        assert e.code == 0

    s3 = s3fs.S3FileSystem(anon=False, client_kwargs={"endpoint_url": "http://127.0.0.1:5555/"})
    output_s3_path = "s3://test_bucket/test_data/output_pickled.obj"

    assert s3.exists(output_s3_path)
