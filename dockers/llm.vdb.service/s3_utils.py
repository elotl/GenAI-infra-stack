import boto3


def save_file_to_s3(file_to_upload, bucket, key):
    """
    Save object to a folder in an S3 bucket.
    """
    s3_client = boto3.client("s3")
    return s3_client.put_object(
        Body=file_to_upload, Bucket=bucket, Key=key
    )
