from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import Settings
from src.vectorstore import load_vector_store


SYSTEM_PROMPT = """
You are an insurance policy QA assistant.
Answer only from the provided policy context.
Answer in simple, plain English that an insurance customer can understand.
Keep the tone professional, calm, and helpful.
Do not mention internal retrieval, chunks, embeddings, vector stores, or source documents.
Do not cite sources in the answer.
If the policy context does not contain enough information, say that the available policy information does not provide enough detail to answer confidently.
When relevant, include important limits, exclusions, waiting periods, deductibles, and conditions.

Policy context:
{context}
"""


def build_qa_chain(settings: Settings):
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

    vector_store = load_vector_store(settings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_k},
    )

    llm = ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


def answer_question(question: str, settings: Settings) -> dict:
    chain = build_qa_chain(settings)
    return chain.invoke({"input": question})


def format_sources(response: dict) -> str:
    context = response.get("context", [])
    if not context:
        return "No sources retrieved."

    lines = []
    for index, doc in enumerate(context, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        location = f"{source}, page {page + 1}" if isinstance(page, int) else source
        snippet = " ".join(doc.page_content.split())[:350]
        lines.append(f"{index}. {location}\n   {snippet}")

    return "\n".join(lines)
