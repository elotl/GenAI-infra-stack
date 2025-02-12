import os

from pydantic_settings import BaseSettings
from pydantic import Field, ValidationError

from typing import Optional


class S3Settings(BaseSettings):
    # This is the directory name on S3 where input files can be found
    s3_dir_name: Optional[str] = Field(
       alias="VECTOR_DB_INPUT_ARG", 
       description="Name of the S3 directory"
    )
    
    # This is the bucket that will be used to store both input datasets for RAG
    # as well as the Vector DB created from this dataset
    s3_bucket_name: Optional[str] = Field(
       alias="VECTOR_DB_S3_BUCKET", 
       description="Name of the S3 bucket"
    )
    
    # This is the name of the Vector DB file that will be created by this script
    vectordb_name: Optional[str] = Field(
        alias="VECTOR_DB_S3_FILE", 
        description="Name of the created Vector DB"
    )

    # AWS ccredentials
    s3_region: Optional[str] = Field(
        None,
        alias="AWS_REGION", 
        description="Region of the S3 bucket"
    )
    s3_access_key: Optional[str] = Field(
        alias="AWS_ACCESS_KEY_ID", 
        description="Access key for S3"
    )
    s3_secret_key: Optional[str] = Field(
        alias="AWS_SECRET_ACCESS_KEY", 
        description="Secret key for S3"
    )

    embedding_chunk_size: int = Field(
        default=1000, 
        description="Chunk size used by the embedding model"
    )
    embedding_chunk_overlap: int = Field(
        default=100, 
        description="Overlap size between chunks"
    )
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the embedding model to use"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


class LocalSettings(BaseSettings):
    local_directory: str = Field(
        description="Directory path for local storage",
    )

    output_filename: str = Field(
        description="Output vectordb filename",
    )

    embedding_chunk_size: int = Field(
        default=1000, 
        description="Chunk size used by the embedding model"
    )
    embedding_chunk_overlap: int = Field(
        default=100, 
        description="Overlap size between chunks"
    )
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the embedding model to use"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


def try_load_settings(env_file):
    if env_file:
        try:
            s3_settings = S3Settings(_env_file=env_file)
            return s3_settings, None
        except ValidationError as e:
            print("ValidationError: ", e)
            try:
                local_settings = LocalSettings(_env_file=env_file)
                return None, local_settings
            except ValidationError as e:
                raise ValueError(f"Missing or invalid configuration: {e}")

    try:
        s3_settings = S3Settings()
        return s3_settings, None
    except ValidationError as e:
        print("ValidationError: ", e)
        try:
            local_settings = LocalSettings()
            return None, local_settings
        except ValidationError as e:
            raise ValueError(f"Missing or invalid configuration: {e}")


class WeaviateSettings(BaseSettings):
    weaviate_uri: Optional[str] = Field(
        ...,
        alias="WEAVIATE_URI",
    )

    weaviate_index_name: Optional[str] = Field(
        ...,
        alias="WEAVIATE_INDEX_NAME",
    )

    class Config:
        env_file = ".env"
        extra = "ignore"

    def is_set(self) -> bool:
        return all([self.weaviate_uri, self.weaviate_index_name])


def try_load_weaviate_settings(env_file):
    if env_file:
        return WeaviateSettings(_env_file=env_file)
    else:
        return WeaviateSettings()
