from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    QDRANT_URL: str = "http://localhost:6333"
    COLLECTION: str = "repo_chunks"
    EMB_MODEL: str = "BAAI/bge-base-en-v1.5"
    CHUNK_TOKENS: int = 1000
    CHUNK_OVERLAP: int = 200
    REPOS: str = ""
    WEBHOOK_SECRET: str = "replace_with_secret"
    GOOGLE_API_KEY: str = "your_gemini_api_key_here"
    
    # Optimization settings
    BATCH_SIZE: int = 32  # Files processed per batch
    EMBEDDING_BATCH_SIZE: int = 16  # Embeddings generated per batch
    MAX_FILE_SIZE: int = 2_000_000  # Maximum file size in bytes
    MAX_CONCURRENT_INGESTIONS: int = 2  # Maximum parallel ingestions
    
    # GitHub authentication
    GITHUB_TOKEN: str = ""  # Personal access token for private repos

settings = Settings()