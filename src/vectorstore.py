from pathlib import Path
from datetime import datetime, timezone
import shutil

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.config import Settings


def build_embeddings(settings: Settings) -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=settings.embedding_model)


def active_chroma_pointer(settings: Settings) -> Path:
    return Path(settings.chroma_dir).parent / "active_chroma_dir.txt"


def active_chroma_dir(settings: Settings) -> Path:
    pointer = active_chroma_pointer(settings)
    if pointer.exists():
        saved_path = Path(pointer.read_text(encoding="utf-8").strip())
        if saved_path.exists():
            return saved_path
    return Path(settings.chroma_dir)


def build_vector_store(chunks: list[Document], settings: Settings) -> VectorStore:
    embeddings = build_embeddings(settings)

    if settings.vector_store == "faiss":
        shutil.rmtree(settings.faiss_dir, ignore_errors=True)
        vector_store = FAISS.from_documents(chunks, embeddings)
        Path(settings.faiss_dir).mkdir(parents=True, exist_ok=True)
        vector_store.save_local(settings.faiss_dir)
        return vector_store

    index_root = Path(settings.chroma_dir).parent / "chroma_indexes"
    index_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    persist_directory = index_root / f"index_{timestamp}"

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )
    active_chroma_pointer(settings).write_text(
        str(persist_directory), encoding="utf-8"
    )
    return vector_store


def load_vector_store(settings: Settings) -> VectorStore:
    embeddings = build_embeddings(settings)

    if settings.vector_store == "faiss":
        if not Path(settings.faiss_dir).exists():
            raise FileNotFoundError("FAISS index not found. Run: python -m src.ingest")
        return FAISS.load_local(
            settings.faiss_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    persist_directory = active_chroma_dir(settings)
    if not persist_directory.exists():
        raise FileNotFoundError("Chroma index not found. Run: python -m src.ingest")
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )
