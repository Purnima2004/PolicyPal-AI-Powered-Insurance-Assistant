from pathlib import Path
import shutil

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.config import Settings


def build_embeddings(settings: Settings) -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=settings.embedding_model)


def build_vector_store(chunks: list[Document], settings: Settings) -> VectorStore:
    embeddings = build_embeddings(settings)

    if settings.vector_store == "faiss":
        shutil.rmtree(settings.faiss_dir, ignore_errors=True)
        vector_store = FAISS.from_documents(chunks, embeddings)
        Path(settings.faiss_dir).mkdir(parents=True, exist_ok=True)
        vector_store.save_local(settings.faiss_dir)
        return vector_store

    shutil.rmtree(settings.chroma_dir, ignore_errors=True)
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.chroma_dir,
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

    if not Path(settings.chroma_dir).exists():
        raise FileNotFoundError("Chroma index not found. Run: python -m src.ingest")
    return Chroma(
        persist_directory=settings.chroma_dir,
        embedding_function=embeddings,
    )
