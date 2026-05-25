from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    vector_store: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    policies_dir: str
    chroma_dir: str
    faiss_dir: str


def get_settings() -> Settings:
    vector_store = os.getenv("VECTOR_STORE", "chroma").strip().lower()
    if vector_store not in {"chroma", "faiss"}:
        raise ValueError("VECTOR_STORE must be either 'chroma' or 'faiss'.")

    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
        vector_store=vector_store,
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
        policies_dir=os.getenv("POLICIES_DIR", "data/policies").strip(),
        chroma_dir=os.getenv("CHROMA_DIR", "storage/chroma").strip(),
        faiss_dir=os.getenv("FAISS_DIR", "storage/faiss").strip(),
    )
