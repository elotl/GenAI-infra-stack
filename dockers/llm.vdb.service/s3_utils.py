import json
import s3fs


def load_jsonl_files_from_s3(bucket_name, prefix=""):
    fs = s3fs.S3FileSystem()
    data = []

    # List all files under the given prefix
    files = fs.ls(f"{bucket_name}/{prefix}")

    for file_path in files:
        print("Processing file:", file_path, "...")
        if file_path.endswith('.json'):
            with fs.open(file_path, 'r') as f:
                try:
                    # Try reading as JSONL first
                    for line in f:
                        if line.strip():  # Skip empty lines
                            data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    # If that fails, try reading as regular JSON
                    f.seek(0)  # Go back to start of file
                    data.append(json.load(f))

    return data


def save_file_to_s3(file_to_upload, bucket, key):
    fs = s3fs.S3FileSystem()
    s3_path = f"{bucket}/{key}"

    with fs.open(s3_path, 'wb') as s3_file:
        if isinstance(file_to_upload, str):
            # If it's a file path, read and upload the file content
            with open(file_to_upload, 'rb') as local_file:
                s3_file.write(local_file.read())
        else:
            # If it's in-memory content (bytes or string), upload directly
            s3_file.write(file_to_upload if isinstance(file_to_upload, bytes) else file_to_upload.encode())
