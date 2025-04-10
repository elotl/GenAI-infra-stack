
##########################################################
# Env Vars used to run Vector DB creation for RAG on the
# Mini wikipedia dataset
##########################################################

export MODEL_LLM_SERVER_URL=http://llm-model-serve-serve-svc.default.svc.cluster.local:8000

export VECTOR_DB_INPUT_TYPE=text-docs
export VECTOR_DB_INPUT_ARG=mini-rag-wikipedia-input
export VECTOR_DB_S3_BUCKET=my-faiss-vectordbs
export VECTOR_DB_S3_FILE=my-s3-rag-wikipedia

export EMBEDDING_CHUNK_SIZE=1000
export EMBEDDING_CHUNK_OVERLAP=100
export EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

export MODEL_ID="llama2"
export RELEVANT_DOCS=2
export MAX_TOKENS=128             
export MODEL_TEMPERATURE=0.01

export IS_JSON_MODE="False"

export AWS_ACCESS_KEY_ID=change_to_correct_value
export AWS_SECRET_ACCESS_KEY=change_to_correct_value
export ELOTL_QA_IN_A_BOX=/change_to_local_path/GenAI-infra-stack


export VECTOR_STORE_PATH=$ELOTL_QA_IN_A_BOX/examples/rag-mini-vectorstore.txt
