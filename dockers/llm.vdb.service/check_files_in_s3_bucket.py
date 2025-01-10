import boto3
from botocore.exceptions import ClientError
import argparse
from typing import List, Optional
from createvectordb import list_files_in_s3_folder

def main():
    """
    Main function to run the S3 file listing script.
    Accepts command line arguments for bucket name and folder prefix.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='List all files in an S3 bucket folder with pagination')
    parser.add_argument('--bucket', required=True, help='Name of the S3 bucket')
    parser.add_argument('--folder', required=True, help='Folder prefix in the bucket')
    parser.add_argument('--profile', help='AWS profile name (optional)', default=None)
    parser.add_argument('--region', help='AWS region (optional)', default='us-east-1')
    
    args = parser.parse_args()
    
    try:
        # Set up AWS session with optional profile
        if args.profile:
            session = boto3.Session(profile_name=args.profile, region_name=args.region)
        else:
            session = boto3.Session(region_name=args.region)
        
        # Create S3 client
        s3_client = session.client('s3')
        
        # Call the listing function
        files = list_files_in_s3_folder(args.bucket, args.folder, s3_client)
        
        # Print summary and sample results
        if files:
            print(f"Summary of files in {args.bucket}/{args.folder}:")
            print(f"Total files found: {len(files)}")
            print("\nFirst 5 files:")
            for file in files[:5]:
                print(file)
            
            print(f"\nLast 5 files:")
            for file in files[-5:]:
                print(file)
            
            print(f"\nFinal count confirmation: {len(files)} files found")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
