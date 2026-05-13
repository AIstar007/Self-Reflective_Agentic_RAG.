import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Load local .env from project root if present.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
load_dotenv()


@dataclass(frozen=True)
class Settings:
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str
    azure_openai_chat_deployment: str
    azure_openai_embedding_deployment: str
    chroma_persist_dir: str
    chroma_collection_name: str
    retrieval_top_k: int
    retrieval_weak_match_distance: float
    max_reflection_rounds: int

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            azure_openai_api_key=_required_env("AZURE_OPENAI_API_KEY"),
            azure_openai_endpoint=_required_env("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            azure_openai_chat_deployment=_required_env("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            azure_openai_embedding_deployment=os.getenv(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
            ),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
            chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "page_units"),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "3")),
            retrieval_weak_match_distance=float(
                os.getenv("RETRIEVAL_WEAK_MATCH_DISTANCE", os.getenv("RAG_DISTANCE_THRESHOLD", "1.2"))
            ),
            max_reflection_rounds=int(os.getenv("MAX_REFLECTION_ROUNDS", "3")),
        )


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value
