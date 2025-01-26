import boto3
import click
import os
import pytest
import s3fs

from moto import mock_aws

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


MOCK_BUCKET_NAME = "test_bucket"


@pytest.fixture
def mock_s3_bucket():
    with mock_aws():
        s3 = boto3.resource("s3")
        s3.create_bucket(
            Bucket=MOCK_BUCKET_NAME,
            CreateBucketConfiguration={
                'LocationConstraint': "us-east-2",  # TODO: make sure this is the same as local default
            },
        )
        yield boto3.client("s3")


def upload_to_s3(mock_s3_client, bucket_name, local_dir, s3_prefix):
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_key = os.path.join(s3_prefix, os.path.relpath(local_file_path, local_dir))
            with open(local_file_path, "rb") as f:
                mock_s3_client.upload_fileobj(f, bucket_name, s3_key)


def test_create_faiss_vector_db_using_s3_files(mock_s3_bucket):

    upload_to_s3(mock_s3_bucket, MOCK_BUCKET_NAME, "test_data/input", "test_data/")

    ctx = click.Context(run)
    try:
        ctx.forward(run, env_file="test_data/.env_s3")
    except SystemExit as e:
        assert e.code == 0

    s3 = s3fs.S3FileSystem()
    output_s3_path = "s3://test_bucket/test_data/output_pickled.obj"

    assert s3.exists(output_s3_path)
