import os
import pickle
import sys

import boto3
from botocore.exceptions import ClientError
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from common import create_vectordb, EMBEDDING_CHUNK_SIZE_DEFAULT, EMBEDDING_CHUNK_OVERLAP_DEFAULT

EMBEDDING_MODEL_NAME_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"

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


def str_to_int(value, name):
    try:
        # Convert the environment variable (or default) to an integer
        int_value = int(value)
    except ValueError:
        print(
            f"Error: Value {name} could not be converted to an integer value, please check."
        )
        sys.exit(1)
    return int_value


if __name__ == "__main__":

    vectordb_input_type = os.environ.get("VECTOR_DB_INPUT_TYPE")
    if vectordb_input_type is None:
        print("Please set environment variable VECTOR_DB_INPUT_TYPE")
        sys.exit(1)
    print("Using Embedding input type: ", vectordb_input_type)

    vectordb_input_arg = os.environ.get("VECTOR_DB_INPUT_ARG")
    if vectordb_input_arg is None:
        print("Please set environment variable VECTOR_DB_INPUT_ARG")
        sys.exit(1)
    print("Using Embedding input arg: ", vectordb_input_arg)

    # This is the bucket that will be used to store both input datasets for
    # RAG as well as the Vector DB created from this dataset
    vectordb_bucket = os.environ.get("VECTOR_DB_S3_BUCKET")
    if vectordb_bucket is None:
        print("Please set environment variable VECTOR_DB_S3_BUCKET")
        sys.exit(1)
    print("Using Vector DB bucket: ", vectordb_bucket)

    # This is the name of the Vector DB file that will be created by this script
    # and will be used by query_rag.py. It has to be unique for each dataset
    # corresponding to a unique VectorDB (or vector store)
    vectordb_file = os.environ.get("VECTOR_DB_S3_FILE")
    if vectordb_file is None:
        print("Please set environment variable VECTOR_DB_S3_FILE")
        sys.exit(1)
    print("Using Vector DB file: ", vectordb_file)

    # This is the chunk size that will be used by the embedding model
    embedding_chunk_size = str_to_int(
        os.environ.get("EMBEDDING_CHUNK_SIZE", EMBEDDING_CHUNK_SIZE_DEFAULT),
        "EMBEDDING_CHUNK_SIZE",
    )
    print("Using Embedding Chunk Size: ", embedding_chunk_size)

 
    embedding_chunk_overlap = str_to_int(
        os.environ.get("EMBEDDING_CHUNK_OVERLAP", EMBEDDING_CHUNK_OVERLAP_DEFAULT), 
        "EMBEDDING_CHUNK_OVERLAP",
    )
    print("Using Embedding Chunk Overlap: ", embedding_chunk_overlap)

    embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", EMBEDDING_MODEL_NAME_DEFAULT)
    print("Using Embedding Model: ", embedding_model_name)

    # Initialize vectorstore and create pickle representation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if vectordb_input_type == "sitemap":
        sitemap_loader = SitemapLoader(
            web_path=vectordb_input_arg, filter_urls=["^((?!.*/v.*).)*$"]
        )
        sitemap_loader.requests_per_second = 1
        docs = sitemap_loader.load()
        print("Count of sitemap docs loaded:", len(docs))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=embedding_chunk_size,
            chunk_overlap=embedding_chunk_overlap,
            length_function=len,
        )
        texts = text_splitter.split_documents(docs)

        # default model name values has been deprecated since 0.2.16, so we choose a specific model
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        vectorstore = FAISS.from_documents(texts, embeddings)

    elif vectordb_input_type == "json-format":
        local_tmp_dir = "/tmp/" + vectordb_file

        # create this temp dir if it does not already exist
        if not os.path.exists(local_tmp_dir):
            os.makedirs(local_tmp_dir)

        # download text docs from S3 bucket + folder (vectordb_s3_input_dir/arg) into this tmp local directory
        num_files = download_files_from_s3(
            vectordb_bucket, vectordb_input_arg, local_tmp_dir
        )
        print(
            f"Number of files downloaded is {num_files}, local tmp dir is {local_tmp_dir}"
        )
        vectorstore = create_vectordb(
            local_tmp_dir,
            embedding_model_name,
            embedding_chunk_size,
            embedding_chunk_overlap,
        )

    else:
        print("Unknown value for VECTOR_DB_INPUT_TYPE:", vectordb_input_type)
        sys.exit(1)

    pickle_byte_obj = pickle.dumps(vectorstore)

    # Persist vectorstore to S3 bucket vectorstores
    s3_client = boto3.client("s3")
    s3_client.put_object(
        Body=pickle_byte_obj, Bucket=vectordb_bucket, Key=vectordb_file
    )
    print("Uploaded vectordb to", vectordb_bucket, vectordb_file)
    sys.exit(0)
