import json
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_jsonl_files_from_directory(directory):
    data = []
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory, filename)
            # Open and read each jsonl file
            with open(file_path, "r") as file:
                for line in file:
                    # Parse each JSON object in the file
                    data.append(json.loads(line.strip()))
    return data


def create_vectordb(local_tmp_dir: str, embedding_model_name: str):
    data = load_jsonl_files_from_directory(local_tmp_dir)

    texts = [doc["text"] for doc in data]
    metadatas = [doc["metadata"] for doc in data]

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore
