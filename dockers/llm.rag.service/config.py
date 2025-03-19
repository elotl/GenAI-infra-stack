from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


def validate_float(value):
    if type(value) == float:
        return value
    try:
        return float(value.strip("'").strip('"'))
    except (TypeError, ValueError):
        raise ValueError("Value must be convertible to a float")


def validate_int(value):
    if type(value) == int:
        return value
    try:
        return int(value.strip("'").strip("\""))
    except (TypeError, ValueError):
        raise ValueError("Value must be convertible to an integer")


class WeaviateSettings(BaseSettings):
    weaviate_uri: Optional[str] = Field(
        default="localhost:8080",
        alias="WEAVIATE_URI_WITH_PORT",
    )
    weaviate_grpc_uri: Optional[str] = Field(
        default="localhost:50051",
        alias="WEAVIATE_GRPC_URI_WITH_PORT",
    )
    weaviate_index_name: Optional[str] = Field(
        default="my_custom_index",
        alias="WEAVIATE_INDEX_NAME",
    )
    # TODO: consider moving to hybrid search config
    weaviate_hybrid_search_alpha: float = Field(
        default=0.5,
        alias="WEAVIATE_HYBRID_ALPHA",
    )
    embedding_model_name: Optional[str] = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )
    # TODO: consider moving to hybrid search config
    hybrid_search_relevant_docs: Optional[int] = Field(
        default=2,
        alias="RELEVANT_DOCS",
    )

    @field_validator("weaviate_hybrid_search_alpha", mode="before")
    @classmethod
    def validate_weaviate_hybrid_search_alpha(cls, v):
        return validate_float(v)

    @field_validator("hybrid_search_relevant_docs", mode="before")
    @classmethod
    def validate_hybrid_search_relevant_docs(cls, v):
        return validate_int(v)

    class Config:
        env_file = ".env"
        extra = "ignore"

    def is_set(self) -> bool:
        return all(
            [self.weaviate_uri, self.weaviate_grpc_uri, self.weaviate_index_name]
        )

    def get_weaviate_uri(self):
        return self.weaviate_uri.split(":")[0]

    def get_weaviate_port(self):
        return int(self.weaviate_uri.split(":")[1])

    def get_weaviate_grpc_uri(self):
        return self.weaviate_grpc_uri.split(":")[0]

    def get_weaviate_grpc_port(self):
        return int(self.weaviate_grpc_uri.split(":")[1])


class LlmSettings(BaseSettings):
    llm_server_url: Optional[str] = Field(
        default="http://localhost:9000/v1",
        alias="MODEL_LLM_SERVER_URL",
    )
    model_id: Optional[str] = Field(
        default="rubra-ai/Phi-3-mini-128k-instruct",
        alias="MODEL_ID",
    )
    max_tokens: Optional[int] = Field(
        default=256,
        alias="MAX_TOKENS",
    )
    model_temperature: Optional[float] = Field(
        default=0.01,
        alias="MODEL_TEMPERATURE",
    )

    @field_validator("llm_server_url")
    @classmethod
    def ensure_v1_endpoint(cls, v: str) -> str:
        if not v.endswith("/v1"):
            v = v + "/v1"
        return v

    @field_validator("max_tokens", mode="before")
    @classmethod
    def validate_max_tokens(cls, v):
        return validate_int(v)

    @field_validator("model_temperature", mode="before")
    @classmethod
    def validate_model_temperature(cls, v):
        return validate_float(v)
