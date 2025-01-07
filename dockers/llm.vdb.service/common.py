import json
import os
import chardet

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import    CharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader


EMBEDDING_CHUNK_SIZE_DEFAULT = 1000
EMBEDDING_CHUNK_OVERLAP_DEFAULT = 100


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

'''
def load_text_files_from_directory(directory):
    data = []
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        with open(file_path, 'rb') as raw_file:
            raw_data = raw_file.read()
            detected = chardet.detect(raw_data)
            encoding = detected['encoding']
            
        # Open and read each text file
        # with open(file_path, "r", encoding=encoding) as file: 
        #   content = file.read()
        #   data.append({
        #           "metadata": file_path,
        #           "text": content
        #   })
        
        with open(file_path, "r") as file: 
            for line in file:
                # Parse each JSON object in the file
                data.append(json.loads(line.strip()))
                #data.append(line)
    return data
'''

def get_documents_with_metadata(data):
    texts = [doc["text"] for doc in data]
    metadatas = [doc["metadata"] for doc in data]
    return texts, metadatas


def chunk_documents_with_metadata(data, chunk_size=1000, chunk_overlap=100):
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
        chunks = text_splitter.split_text(doc["text"])

        if doc["metadata"] != "":
            if isinstance(doc["metadata"], str):
                doc_metadatas = [doc["metadata"] for _ in chunks]
            else:
                doc_metadatas = [doc["metadata"].copy() for _ in chunks]
            
        # This is just to see if it's used or not
        for i, (chunk, metadata) in enumerate(zip(chunks, doc_metadatas)):
            metadata["chunk_index"] = i
            metadata["chunk_total"] = len(chunks)

        all_chunks.extend(chunks)
        all_metadatas.extend(doc_metadatas)

    return all_chunks, all_metadatas


def create_vectordb(
    is_json: bool,
    local_tmp_dir: str,
    embedding_model_name: str,
    chunk_size: int = EMBEDDING_CHUNK_SIZE_DEFAULT,
    chunk_overlap: int = EMBEDDING_CHUNK_OVERLAP_DEFAULT,
):

    if is_json:
        data = load_jsonl_files_from_directory(local_tmp_dir)

        # no chunking
        # texts, metadatas = get_documents_with_metadata(data)
        # with chunking texts
        texts, metadatas = chunk_documents_with_metadata(data, chunk_size, chunk_overlap)

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        return vectorstore

    else: 

        # read text files from local tmp directory 
        loader = DirectoryLoader(local_tmp_dir, glob="**/*")
        documents = loader.load()
        print(f"Number of documents loaded via DirectoryLoader is {len(documents)}")

        # TODO (improvement for later) Allow users to configure chunk size and overlap values
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs = text_splitter.split_documents(documents)

        # default model name values has been deprecated since 0.2.16, so we choose a specific model
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        vectorstore = FAISS.from_documents(docs, embeddings)

        #local_tmp_dir = "/tmp/" + vectordb_file

        # create this temp dir if it does not already exist
        if not os.path.exists(local_tmp_dir):
            os.makedirs(local_tmp_dir)

        print(
            f"Local tmp dir is {local_tmp_dir}"
        )
        #vectorstore = create_vectordb(
        #    local_tmp_dir,
        #    embedding_model_name,
        #    chunk_size,
        #    chunk_overlap,
        #)

        return vectorstore

