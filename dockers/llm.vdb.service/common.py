import json
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


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


def create_vectordb_from_data(
    data,
    embedding_model_name: str,
    chunk_size,
    chunk_overlap,
):
    # no chunking
    # texts, metadatas = get_documents_with_metadata(data)

    # with chunking texts
    print("Start chunking documents")
    texts, metadatas = chunk_documents_with_metadata(data, chunk_size, chunk_overlap)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print("Convert to FAISS vectorstore")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore
