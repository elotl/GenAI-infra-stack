import json
import os
from typing import List

import weaviate

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore


def load_jsonl_files_from_directory(directory):
    data = []
    for filename in os.listdir(directory):
        print("Processing file, ", filename, "..." )
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename)) as f:
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


def get_documents(data):
    texts = [doc["text"] for doc in data]
    metadatas = [doc["metadata"] for doc in data]
    return texts, metadatas


def chunk_documents(data, chunk_size, chunk_overlap):
    """
    Chunks documents while maintaining alignment between text chunks and metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Lists to store chunks and corresponding metadata
    all_chunks = []
    all_metadatas = []

    for doc in data:
        print("Chunking doc with key/ticket ID, ", doc["metadata"].get("ticket") or doc["metadata"].get("key"))
        chunks = text_splitter.split_text(doc["text"])

        doc_metadatas = [doc["metadata"].copy() for _ in chunks]

        # This is just to see if it's used or not
        for i, (chunk, metadata) in enumerate(zip(chunks, doc_metadatas)):
            metadata["chunk_index"] = i
            metadata["chunk_total"] = len(chunks)

        all_chunks.extend(chunks)
        all_metadatas.extend(doc_metadatas)

    print("Number of chunks created: ", len(all_chunks))
    return all_chunks, all_metadatas


def chunk_documents_with_added_metadata(data, chunk_size, chunk_overlap):
    """
    Splits documents into smaller text chunks while preserving alignment with metadata.
    Additionally, each chunk is enriched by embedding its corresponding metadata into the text.

    The method determines the maximum metadata size across all documents and adjusts
    the effective chunk size accordingly to ensure that metadata fits within each chunk.

    Metadata keys with `None` values are excluded from the embedded metadata in the text.
    """
    # TODO: find a better way to find the biggest metadata
    max_size_of_metadata = 0
    for doc in data:
        meta_enhancement = ". ".join([f"{key.title()}: {value}" for key, value in doc["metadata"].items() if value])
        max_size_of_metadata = max(max_size_of_metadata, len(meta_enhancement))

    print(f"Biggest metada has {max_size_of_metadata} characters.")
    effective_chunk_size = chunk_size - max_size_of_metadata
    print(f"Effective chunk size will be {effective_chunk_size}")

    # TODO: handle better
    if effective_chunk_size <= 0:
        raise "Use bigger chunk size"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Lists to store chunks and corresponding metadata
    all_chunks = []
    all_metadatas = []

    for doc in data:
        print("Chunking doc with key/ticket ID, ", doc["metadata"].get("ticket") or doc["metadata"].get("key"))
        chunks = text_splitter.split_text(doc["text"])
        chunks_enriched_with_metadata = []

        doc_metadatas = [doc["metadata"].copy() for _ in chunks]

        meta_enhancement = ". ".join([f"{key.title()}: {value}" for key, value in doc["metadata"].items() if value])

        for i, (chunk, metadata) in enumerate(zip(chunks, doc_metadatas)):
            chunks_enriched_with_metadata.append(chunk + "\n" + meta_enhancement)

            # This is just to see if it's used or not
            metadata["chunk_index"] = i
            metadata["chunk_total"] = len(chunks)

        all_chunks.extend(chunks_enriched_with_metadata)
        all_metadatas.extend(doc_metadatas)

    print("Number of chunks created: ", len(all_chunks))
    return all_chunks, all_metadatas


def create_vectordb_from_data(
    data,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
):
    # no chunking
    # texts, metadatas = get_documents(data)

    # with chunking texts
    # texts, metadatas = chunk_documents(data, chunk_size, chunk_overlap)

    # with adding metadata to text
    print("Start chunking documents")
    texts, metadatas = chunk_documents_with_added_metadata(data, chunk_size, chunk_overlap)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print("Convert to FAISS vectorstore")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore


def create_vectordb_local_weaviate(
    data,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    weaviate_url: str,
    weaviate_index_name: str,
):
    # with adding metadata to text
    print("Start chunking documents")
    texts, metadatas = chunk_documents_with_added_metadata(data, chunk_size, chunk_overlap)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # adapt data
    documents: List[Document] = []
    for txt, met in zip(texts, metadatas):
        document = Document(
            page_content=txt,
            metadata=met
        )
        documents.append(document)

    # TODO: enable connecting to other weaviate than local
    with weaviate.connect_to_custom(
        http_host=weaviate_url,
        http_port=8080,
        http_secure=False,
        grpc_host=weaviate_url,
        grpc_port=50051,
        grpc_secure=False,
    ) as weaviate_client:
        # return WeaviateVectorStore.from_documents(
        #     documents,
        #     embeddings,
        #     client=weaviate_client,
        #     index_name=weaviate_index_name,
        # )
        return WeaviateVectorStore.from_texts(
            texts,
            embeddings,
            client=weaviate_client,
            metadatas=metadatas,
            index_name=weaviate_index_name,
            text_key="text",
        )
