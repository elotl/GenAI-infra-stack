import os
import sys
import boto3
import pickle
import time

from typing import Union
from fastapi import FastAPI
from botocore.exceptions import NoCredentialsError, ClientError

from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

########
# Setup model name and query template parameters
model = "mosaicml/mpt-7b-chat"
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

########
# Fetch RAG context for question, form prompt from context and question, and call model
def get_answer(question: Union[str, None]):

    print("Received question: ", question)
    # retrieve docs relevant to the input question
    docs = retriever.invoke(input=question)
    # default number of docs is 4; make this configurable later
    print ("Number of relevant documents retrieved and that will be used as context for query: ", len(docs))

    # concatenate relevant docs retrieved to be used as context 
    allcontext = ""
    for i in range(len(docs)):
        allcontext += docs[i].page_content
    promptstr = template.format(context=allcontext, question=question)
    
    print("Sending query to the LLM...")
    completions = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": promptstr,
            },
        ],
        max_tokens=64,
        temperature=0.01,
        stream=False,
    )
   
    answer = completions.choices[0].message.content
    print("Received answer: ", answer)
    return answer


########
# Get connection to LLM server
model_llm_server_url = os.environ.get('MODEL_LLM_SERVER_URL')
if model_llm_server_url is None:
    model_llm_server_url = "http://llm-model-serve-serve-svc.default.svc.cluster.local:8000"
    print("Setting environment variable MODEL_LLM_SERVER_URL to default value: ", model_llm_server_url )
llm_server_url = model_llm_server_url + '/v1'

print("Creating an OpenAI client to the hosted model at URL: ", llm_server_url)
try:
    client = OpenAI(base_url=llm_server_url, api_key='na')
except Exception as e:
    print("Error creating client:", e)
    sys.exit(1)

########
# Load vectorstore and get retriever for it

# get env vars needed to access Vector DB
vectordb_bucket = os.environ.get('VECTOR_DB_S3_BUCKET')
print ("Using vector DB s3 bucket: ", vectordb_bucket)
if vectordb_bucket is None:
    print("Please set environment variable VECTOR_DB_S3_BUCKET")
    sys.exit(1)

vectordb_key = os.environ.get('VECTOR_DB_S3_FILE')
print ("Using vector DB s3 file containing vector store: ", vectordb_key)
if vectordb_key is None:
    print("Please set environment variable VECTOR_DB_S3_FILE")
    sys.exit(1)

# Use s3 client to read in vector store
s3_client = boto3.client('s3')
response = None
try:
    response = s3_client.get_object(Bucket=vectordb_bucket, Key=vectordb_key)
except ClientError as e:
    print(f"Error accessing object, {vectordb_key} in bucket, {vectordb_bucket}, err: {e}")
    sys.exit(1)
body = response['Body'].read()

print("Loading Vector DB...\n")
# needs prereq packages: sentence_transformers and faiss-cpu
vectorstore = pickle.loads(body)
retriever = vectorstore.as_retriever()
print("Created Vector DB retriever successfully. \n")

# Uncomment to run a local test
# print("Testing with a sample question:")
# get_answer("who are you?")

########
# Start API service to answer questions
app = FastAPI()
@app.get("/answer/{question}")
def read_item(question: Union[str, None] = None):
    print(f"Received question: {question}")
    answer = get_answer(question)
    return {"question": question, "answer": answer}