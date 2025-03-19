from typing import Optional

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


SYSTEM_PROMPT_DEFAULT = """You are a specialized support ticket assistant. Format your responses following these rules:
                1. Answer the provided question only using the provided context.
                2. Do not add the provided context to the generated answer.
                3. Include relevant technical details when present or provide a summary of the comments in the ticket.
                4. Include the submitter, assignee and collaborator for a ticket when this info is available.
                5. If the question cannot be answered with the given context, please say so and do not attempt to provide an answer.
                6. Do not create new questions related to the given question, instead answer only the provided question.
                7. Provide a clear, direct and factual answer.
                """


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
    model_config = ConfigDict(env_file=".env", extra="ignore")

    weaviate_uri: Optional[str] = Field(
        None,
        alias="WEAVIATE_URI_WITH_PORT",
    )
    weaviate_grpc_uri: Optional[str] = Field(
        None,
        alias="WEAVIATE_GRPC_URI_WITH_PORT",
    )
    weaviate_index_name: Optional[str] = Field(
        None,
        alias="WEAVIATE_INDEX_NAME",
    )
    embedding_model_name: Optional[str] = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )

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
    model_config = ConfigDict(env_file=".env", extra="ignore")

    llm_server_url: Optional[str] = Field(
        None,
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


class HybridSearchSettings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    alpha: Optional[float] = Field(
        default=0.5,
        alias="WEAVIATE_HYBRID_ALPHA",
    )
    relevant_docs: Optional[int] = Field(
        default=2,
        alias="RELEVANT_DOCS",
    )
    system_prompt: Optional[str] = Field(
        default=SYSTEM_PROMPT_DEFAULT,
        alias="SYSTEM_PROMPT",
    )

    @field_validator("alpha", mode="before")
    @classmethod
    def validate_alpha(cls, v):
        return validate_float(v)

    @field_validator("relevant_docs", mode="before")
    @classmethod
    def validate_relevant_docs(cls, v):
        return validate_int(v)


class SqlSearchSettings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    db_and_model_path: Optional[str] = Field(
        default="/app/db/",
        alias="SQL_SEARCH_DB_AND_MODEL_PATH",
    )
    max_context_length: Optional[int] = Field(
        default=8192,
        alias="MODEL_MAX_CONTEXT_LEN",
    )
    # TODO: Read sources from DB instead of using this
    ticket_source: Optional[str] = Field(
        default="https://zendesk.com/api/v2/tickets/",
        alias="SQL_TICKET_SOURCE",
    )

    @field_validator("max_context_length", mode="before")
    @classmethod
    def validate_max_context_length(cls, v):
        return validate_int(v)
