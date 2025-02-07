# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "haystack",
#     "fastembed-haystack",
#     "milvus-haystack",
#     "pymilvus",
#     "sentence-transformers>=3.0.0",
# ]
# ///
import os

from haystack import Document, Pipeline
from haystack.utils import Secret
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder, HuggingFaceAPITextEmbedder, OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder,
)

from milvus_haystack import MilvusDocumentStore, MilvusHybridRetriever

token = "xxx"
os.environ["HUGGINGFACE_HUB_TOKEN"] = token

document_store = MilvusDocumentStore(
    connection_args={"uri": "./milvus.db"},
    drop_old=True,
    sparse_vector_field="sparse_vector",  # Specify a name of the sparse vector field to enable hybrid retrieval.
)

documents = [
    Document(content="My name is Wolfgang and I live in Berlin"),
    Document(content="I saw a black horse running"),
    Document(content="Germany has many big cities"),
    Document(content="fastembed is supported by and maintained by Milvus."),
]

writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)

# dense_embedder = OpenAIDocumentEmbedder()
dense_embedder = HuggingFaceAPIDocumentEmbedder(
    api_type="text_embeddings_inference",
    api_params={
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "url": "http://localhost:11434",
    },
    token=Secret.from_token(token),
)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("sparse_doc_embedder", FastembedSparseDocumentEmbedder())
indexing_pipeline.add_component("dense_doc_embedder", dense_embedder)
indexing_pipeline.add_component("writer", writer)
indexing_pipeline.connect("sparse_doc_embedder", "dense_doc_embedder")
indexing_pipeline.connect("dense_doc_embedder", "writer")

indexing_pipeline.run({"sparse_doc_embedder": {"documents": documents}})

querying_pipeline = Pipeline()
querying_pipeline.add_component("sparse_text_embedder",
                                FastembedSparseTextEmbedder(model="prithvida/Splade_PP_en_v1"))

# dense_text_embedder = OpenAITextEmbedder()
dense_text_embedder = HuggingFaceAPITextEmbedder(
    api_type="text_embeddings_inference",
    api_params={
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "url": "http://localhost:11434",
    },
    token=Secret.from_token(token),
)

querying_pipeline.add_component("dense_text_embedder", dense_text_embedder)
querying_pipeline.add_component(
    "retriever",
    MilvusHybridRetriever(
        document_store=document_store,
        # reranker=WeightedRanker(0.5, 0.5),  # Default is RRFRanker()
    )
)

querying_pipeline.connect("sparse_text_embedder.sparse_embedding", "retriever.query_sparse_embedding")
querying_pipeline.connect("dense_text_embedder.embedding", "retriever.query_embedding")

question = "Who supports fastembed?"

results = querying_pipeline.run(
    {"dense_text_embedder": {"text": question},
     "sparse_text_embedder": {"text": question}}
)

print(results)
print(results["retriever"]["documents"][0])

# Document(id=..., content: 'fastembed is supported by and maintained by Milvus.', embedding: vector of size 1536, sparse_embedding: vector with 48 non-zero elements)
