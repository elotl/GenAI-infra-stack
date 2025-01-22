import os
import pickle
import sys

import boto3
from botocore.exceptions import ClientError

from common import create_vectordb
from config import S3Settings


def list_files_in_s3_folder(bucket_name: str, folder_name: str, s3_client) -> list[str]:
    """
    List all files within a given folder (prefix) in an S3 bucket. Any folders within are ignored.
    Handles pagination to retrieve all files.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        folder_name (str): Folder prefix to list files from
        s3_client: Boto3 S3 client instance
    
    Returns:
        List[str]: List of file keys in the specified folder
    """
    try:
        file_names = []
        paginator = s3_client.get_paginator('list_objects_v2')
        
        # Create a PageIterator from the Paginator
        page_iterator = paginator.paginate(
            Bucket=bucket_name,
            Prefix=folder_name
        )
        
        count = 0
        print(f"Starting to list files in {bucket_name}/{folder_name}")
        
        # Iterate through each page
        for page in page_iterator:
            if "Contents" not in page:
                print(f"No files found in folder {folder_name} in bucket {bucket_name}")
                return []
                
            for content in page["Contents"]:
                # ignoring any folders within the top-level folder
                if not content["Key"].endswith("/"):
                    file_names.append(content["Key"])
                    count += 1
                    if count % 1000 == 0:
                        print(f"Processed {count} files...")
        
        print(f"Total number of files found: {count}")
        return file_names
        
    except ClientError as e:
        print(
            f"Error listing files in the S3 bucket, {bucket_name}, folder, {folder_name}, err: {e}"
        )
        return []

def download_files_from_s3(bucket_name, folder_name, local_dir):
    """
    Download all files from a folder in an S3 bucket to a local directory.
    """

    # Initialize the S3 client
    s3_client = boto3.client("s3")

    # List and download files from the specified S3 folder
    file_names = list_files_in_s3_folder(bucket_name, folder_name, s3_client)
    print(f"Number of files in S3 folder: {len(file_names)}")
    if file_names:
        print(f"Found {len(file_names)} file(s) in the folder '{folder_name}'")
        for file_name in file_names:
            local_file_path = os.path.join(local_dir, os.path.basename(file_name))

            try:
                # download file
                s3_client.download_file(bucket_name, file_name, local_file_path)
                print(
                    f"Downloaded file, {file_name} successfully to directory, {local_dir}"
                )
            except Exception as e:
                print(
                    f"Error while downloading file, {file_name} from S3, {bucket_name}, err: {e}"
                )
    else:
        print(f"No files to download in folder {folder_name} in bucket, {bucket_name}.")
        return 0
    return len(file_names)


if __name__ == "__main__":

    config = S3Settings()

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
    s3_client = boto3.client("s3")
    s3_client.put_object(
        Body=pickle_byte_obj, Bucket=config.s3_bucket_name, Key=config.vectordb_name
    )
    print("Uploaded vectordb to", config.s3_bucket_name, config.vectordb_name)
    sys.exit(0)
