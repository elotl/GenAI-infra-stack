import boto3
from botocore.exceptions import ClientError


def _list_files_in_s3_folder(bucket_name: str, folder_name: str, s3_client) -> list[str]:
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
    s3_client = boto3.client("s3")

    # List and download files from the specified S3 folder
    file_names = _list_files_in_s3_folder(bucket_name, folder_name, s3_client)
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


def save_file_to_s3(file_to_upload, bucket, key):
    """
    Save object to a folder in an S3 bucket.
    """
    s3_client = boto3.client("s3")
    return s3_client.put_object(
        Body=file_to_upload, Bucket=bucket, Key=key
    )