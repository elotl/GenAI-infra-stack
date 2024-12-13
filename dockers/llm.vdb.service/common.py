import json
import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_CHUNK_SIZE_DEFAULT = 1000
EMBEDDING_CHUNK_OVERLAP_DEFAULT = 100

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

def get_documents_with_metadata(data):
    texts = [doc["text"] for doc in data]
    metadatas = [doc["metadata"] for doc in data]
    return texts, metadatas


def chunk_documents_with_metadata(data, chunk_size=1000, chunk_overlap=200):
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
        print("Chunking doc with key, ", doc["metadata"]["key"])
        chunks = text_splitter.split_text(doc["text"])

        doc_metadatas = [doc["metadata"].copy() for _ in chunks]

        # This is just to see if it's used or not
        for i, (chunk, metadata) in enumerate(zip(chunks, doc_metadatas)):
            metadata["chunk_index"] = i
            metadata["chunk_total"] = len(chunks)

        all_chunks.extend(chunks)
        all_metadatas.extend(doc_metadatas)

    return all_chunks, all_metadatas


def create_vectordb(
    local_tmp_dir: str,
    embedding_model_name: str,
    chunk_size: int = EMBEDDING_CHUNK_SIZE_DEFAULT,
    chunk_overlap: int = EMBEDDING_CHUNK_OVERLAP_DEFAULT,
):
    print("Load JSON files")
    data = load_jsonl_files_from_directory(local_tmp_dir)

    # no chunking
    # texts, metadatas = get_documents_with_metadata(data)
    # with chunking texts
    print("Start chunking documents")
    texts, metadatas = chunk_documents_with_metadata(data, chunk_size, chunk_overlap)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print("Convert to FAISS vectorstore")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore


def create_vectordb_pgvector(
    local_tmp_dir: str,
    embedding_model_name: str,
    connection_string: str,
    collection_name: str,
    chunk_size: int = EMBEDDING_CHUNK_SIZE_DEFAULT,
    chunk_overlap: int = EMBEDDING_CHUNK_OVERLAP_DEFAULT,
):
    data = load_jsonl_files_from_directory(local_tmp_dir)

    # no chunking
    # texts, metadatas = get_documents_with_metadata(data)
    # with chunking texts
    texts, metadatas = chunk_documents_with_metadata(data, chunk_size, chunk_overlap)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # TODO: move to imports
    from langchain.vectorstores.pgvector import PGVector
    from langchain_core.documents import Document

    # adapt data
    documents: List[Document] = []
    for txt, met in zip(texts, metadatas):
        document = Document(
            page_content=txt,
            metadata=met
        )
        documents.append(document)

    return PGVector.from_documents(
        embedding=embeddings,
        documents=documents,
        collection_name=collection_name,
        connection_string=connection_string,
    )
