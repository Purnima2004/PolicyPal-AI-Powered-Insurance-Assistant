from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_policy_documents(policies_dir: str) -> list[Document]:
    root = Path(policies_dir)
    if not root.exists():
        raise FileNotFoundError(f"Policy directory not found: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")

        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = str(path)
        documents.extend(loaded)

    if not documents:
        raise ValueError(
            f"No policy documents found in {root}. Add PDF, TXT, or MD files first."
        )

    return documents


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)
