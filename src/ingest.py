from src.config import get_settings
from src.documents import load_policy_documents, split_documents
from src.vectorstore import build_vector_store


def ingest_policies() -> dict[str, int | str]:
    settings = get_settings()
    documents = load_policy_documents(settings.policies_dir)
    chunks = split_documents(documents, settings)
    build_vector_store(chunks, settings)

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "vector_store": settings.vector_store,
    }


def main() -> None:
    result = ingest_policies()

    print(f"Loaded documents: {result['documents']}")
    print(f"Indexed chunks: {result['chunks']}")
    print(f"Vector store: {result['vector_store']}")


if __name__ == "__main__":
    main()
