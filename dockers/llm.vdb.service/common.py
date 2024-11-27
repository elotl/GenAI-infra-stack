import json
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_jsonl_files_from_directory(directory):
    data = []
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl") or filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            # Open and read each jsonl file
            with open(file_path, "r") as file:
                for line in file:
                    # Parse each JSON object in the file
                    data.append(json.loads(line.strip()))
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
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Lists to store chunks and corresponding metadata
    all_chunks = []
    all_metadatas = []

    for doc in data:
        chunks = text_splitter.split_text(doc["text"])

        doc_metadatas = [doc["metadata"].copy() for _ in chunks]

        # This is just to see if it's used or not
        for i, (chunk, metadata) in enumerate(zip(chunks, doc_metadatas)):
            metadata["chunk_index"] = i
            metadata["chunk_total"] = len(chunks)

        all_chunks.extend(chunks)
        all_metadatas.extend(doc_metadatas)

    return all_chunks, all_metadatas


def create_vectordb(local_tmp_dir: str, embedding_model_name: str):
    data = load_jsonl_files_from_directory(local_tmp_dir)

    # no chunking
    # texts, metadatas = get_documents_with_metadata(data)
    # with chunking texts
    texts, metadatas = chunk_documents_with_metadata(data)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore
