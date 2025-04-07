import os
from dataclasses import dataclass
import gel


@dataclass
class Config:
    # Ollama settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemma3:1b")

    # Database settings
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5656")  # Default Gel database port

    # Application settings
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
    VERBOSE: bool = os.getenv("VERBOSE", "True").lower() in ("true", "1", "yes")

    # Gel database client (initialize once)
    client: gel.AsyncIOClient = None

    def __post_init__(self):
        self.client = gel.create_client(
            host=self.DB_HOST,
            port=self.DB_PORT,
        )


# Create a global config instance
config = Config()
