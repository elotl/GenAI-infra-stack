import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def create_s3_folder(bucket_name, s3_folder_name, s3_client):
    """
    Create a folder (prefix) in the given S3 bucket.
    """
    # S3 uses prefixes to simulate folders, the trailing slash indicates a folder
    folder_key = s3_folder_name.rstrip('/') + '/'
    
    try:
        # We create a folder by uploading an empty object
        s3_client.put_object(Bucket=bucket_name, Key=folder_key)
        print(f"Folder '{s3_folder_name}' was created successfully in bucket '{bucket_name}'")
    except ClientError as e:
        print(f"Error creating folder '{s3_folder_name}' in bucket '{bucket_name}', err: {e}")
        return False
    return True

def upload_files_to_s3(bucket_name, s3_folder_name, local_folder, s3_client):
    """
    Uploads all files from the given local folder to the S3 folder.
    """
    for root, _, files in os.walk(local_folder):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            
            # Create the S3 file key by concatenating the folder name with the file name
            s3_file_key = os.path.join(s3_folder_name, file_name).replace("\\", "/")  # Replace backslashes with forward slashes for S3 compatibility
            
            try:
                # Upload the file to the S3 folder
                s3_client.upload_file(local_file_path, bucket_name, s3_file_key)
                print(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_file_key}")
            except FileNotFoundError:
                print(f"File not found: {local_file_path}")
            except NoCredentialsError:
                print("Credentials not available.")
            except ClientError as e:
                print(f"Error uploading file: {e}")

if __name__ == "__main__":
    
    # Get environment variables for the S3 bucket, folder, and local directory where the RAG data is available
    
    # S3 bucket name where both the input data and Vector DB will be stored
    bucket_name = os.getenv('VECTOR_DB_S3_BUCKET')
    
    # Name of the folder to create in S3 where the RAG data will be stored
    s3_folder_name = os.getenv('VECTOR_DB_INPUT_ARG')

    # The local folder containing the RAG dataset's files that is to be uploaded
    local_folder = os.getenv('VECTOR_DB_LOCAL_INPUT_DIR')

    # Initialize the S3 client
    s3_client = boto3.client('s3')

    # Create the folder in S3
    if os.path.isdir(local_folder):
        folder_created = create_s3_folder(bucket_name, s3_folder_name, s3_client)

        if folder_created:
            # Upload files to the created S3 folder
            upload_files_to_s3(bucket_name, s3_folder_name, local_folder, s3_client)
    else:
        print(f"The provided path '{local_folder}' is not a valid directory.")

