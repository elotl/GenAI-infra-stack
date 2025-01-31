import os
import pickle

from dataclasses import dataclass

from common import (
    create_milvus_vectordb_from_data,
    create_vectordb_from_data,
    json_to_document,
    load_jsonl_files_from_directory,
)
from config import LocalSettings, S3Settings
from s3_utils import load_jsonl_files_from_s3, save_file_to_s3


@dataclass
class S3VectorDbCreationService:
    config: S3Settings

    def create(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        os.environ["AWS_ACCESS_KEY_ID"] = self.config.s3_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.config.s3_secret_key

        print("Load JSON files")
        data = load_jsonl_files_from_s3(self.config.s3_bucket_name, self.config.s3_dir_name)

        print("Convert to FAISS vectorstore")
        vectorstore = create_vectordb_from_data(
            data,
            self.config.embedding_model_name,
            self.config.embedding_chunk_size,
            self.config.embedding_chunk_overlap,
        )

        pickle_byte_obj = pickle.dumps(vectorstore)

        save_file_to_s3(pickle_byte_obj, self.config.s3_bucket_name, self.config.vectordb_name)
        print("Uploaded vectordb to", self.config.s3_bucket_name, self.config.vectordb_name)


@dataclass
class LocalDirDbCreationService:
    config: LocalSettings

    def create(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        print("Load JSON files")
        data = load_jsonl_files_from_directory(self.config.local_directory)

        print("Convert to FAISS vectorstore")
        vectorstore = create_vectordb_from_data(
            data,
            self.config.embedding_model_name,
            self.config.embedding_chunk_size,
            self.config.embedding_chunk_overlap,
        )

        pickle_byte_obj = pickle.dumps(vectorstore)

        with open(self.config.output_filename, "wb") as file:
            file.write(pickle_byte_obj)
        print(f"Pickle byte object saved to {self.config.output_filename}")


@dataclass
class LocalDirMilvusDbCreationService:
    config: LocalSettings

    def create(self):
        print("Load JSON files")
        data = load_jsonl_files_from_directory(self.config.local_directory)
        docs = []
        for d in data:
            docs.append(json_to_document(d))

        print("Convert to Milvus vectorstore")
        create_milvus_vectordb_from_data(
            docs,
            self.config.embedding_model_name,
            self.config.milvus_uri,
            self.config.milvus_collection_name,
        )

        print(f"Milvus collection saved {self.config.milvus_collection_name}")
