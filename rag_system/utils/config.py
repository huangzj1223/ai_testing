import os
from pydantic import BaseModel, Field

class Configuration(BaseModel):
    milvus_uri: str = Field(default=os.getenv("MILVUS_URI"), description="Milvus URI")
    embedding_model_dim: int = Field(default=512, description="Embedding model dimension")
    llm_api_key: str = Field(default=os.getenv("LLM_API_KEY"), description="LLM API key")
    llm_api_base: str = Field(default=os.getenv("LLM_API_BASE"), description="LLM API base")
    llm_model_name: str = Field(default=os.getenv("LLM_MODEL_NAME"), description="LLM model name")
    local_embedding_model_name: str = Field(default=os.getenv("LOCAL_EMBEDDING_MODEL_NAME"), description="Local embedding model name")
    remote_embedding_model_name: str = Field(default=os.getenv("REMOTE_EMBEDDING_MODEL_NAME"), description="Remote embedding model name")
    remote_embedding_model_url: str = Field(default=os.getenv("REMOTE_EMBEDDING_MODEL_URL"), description="Remote embedding model URL")
    moonshot_api_key: str = Field(default=os.getenv("MOONSHOT_API_KEY"), description="Moonshot API key")
    pg_connection_string: str = Field(default=os.getenv("PG_CONNECTION_STRING"), description="Postgres connection string")
    minio_endpoint: str = Field(default=os.getenv("MINIO_ENDPOINT"), description="Minio endpoint")
    minio_access_key: str = Field(default=os.getenv("MINIO_ACCESS_KEY"), description="Minio access key")
    minio_secret_key: str = Field(default=os.getenv("MINIO_SECRET_KEY"), description="Minio secret key")
    minio_bucket_name: str = Field(default=os.getenv("MINIO_BUCKET_NAME"), description="Minio bucket name")
