from src.config import get_settings
from src.rag import answer_question


def ask_policy_question(question: str) -> dict[str, str]:
    settings = get_settings()
    response = answer_question(question, settings)
    answer = response["answer"]
    return {
        "answer": answer,
        "content": answer,
    }


def main() -> None:
    print("Insurance Policy QA Bot")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        response = ask_policy_question(question)
        print("\nAnswer:")
        print(response["answer"])
        print()


if __name__ == "__main__":
    main()
