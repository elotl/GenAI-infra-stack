from pydantic_settings import BaseSettings
from pydantic import Field

from typing import Optional


WEAVIATE_HYBRID_ALPHA_DEFAULT = 0.5


def validate_float(value):
    if type(value) == float:
        return value
    try:
        return float(value.strip("'").strip("\""))
    except (TypeError, ValueError):
        raise ValueError("Value must be convertible to a float")


class WeaviateSettings(BaseSettings):
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
    weaviate_hybrid_search_alpha: float = Field(
        default=WEAVIATE_HYBRID_ALPHA_DEFAULT,
        alias="WEAVIATE_HYBRID_ALPHA",
    )

    class Config:
        env_file = ".env"
        extra = "ignore"

    def is_set(self) -> bool:
        return all([self.weaviate_uri, self.weaviate_grpc_uri, self.weaviate_index_name])
