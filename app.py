import re
from pathlib import Path

from groq import AuthenticationError
import streamlit as st

from src.chat import ask_policy_question
from src.config import get_settings
from src.documents import SUPPORTED_SUFFIXES
from src.ingest import ingest_policies


st.set_page_config(page_title="Insurance Policy QA Bot", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 7rem;
        padding-bottom: 1.25rem;
    }
    .upload-kicker {
        color: #38bdf8;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 0.85rem;
        text-transform: uppercase;
    }
    .upload-title {
        font-size: 2.65rem;
        font-weight: 700;
        line-height: 1.08;
        margin-bottom: 1rem;
        max-width: 680px;
    }
    .upload-subtitle {
        font-size: 1.05rem;
        color: #9ca3af;
        line-height: 1.65;
        max-width: 680px;
        margin-bottom: 1.4rem;
    }
    .trust-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1rem;
    }
    .trust-pill {
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 999px;
        color: #d1d5db;
        font-size: 0.88rem;
        padding: 0.4rem 0.72rem;
    }
    .upload-panel-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-bottom: 0.35rem;
    }
    .upload-panel-copy {
        color: #9ca3af;
        font-size: 0.92rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    div[data-testid="stFileUploader"] {
        margin-bottom: 0.75rem;
    }
    div[data-testid="stFileUploader"] section {
        border-radius: 8px;
        min-height: 142px;
    }
    @media (max-width: 760px) {
        .block-container {
            padding-top: 2rem;
        }
        .upload-title {
            font-size: 2.1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip()


def save_and_prepare_files(uploaded_files) -> int:
    settings = get_settings()
    policies_dir = Path(settings.policies_dir)
    policies_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing policy files to avoid mixing previous documents
    for file_path in policies_dir.iterdir():
        if file_path.is_file() and file_path.name != ".gitkeep":
            file_path.unlink()

    saved_count = 0
    for uploaded_file in uploaded_files:
        filename = safe_filename(uploaded_file.name)
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue

        destination = policies_dir / filename
        destination.write_bytes(uploaded_file.getbuffer())
        saved_count += 1

    if saved_count:
        ingest_policies()

    return saved_count


if "messages" not in st.session_state:
    st.session_state.messages = []

if "policy_ready" not in st.session_state:
    st.session_state.policy_ready = False


if not st.session_state.policy_ready:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown(
            '<div class="upload-kicker">Policy QA Assistant</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="upload-title">Understand your insurance policy without reading every page.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="upload-subtitle">Upload your policy document and ask natural questions about coverage, claims, exclusions, deductibles, and waiting periods.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="trust-row">
                <span class="trust-pill">PDF, TXT, MD</span>
                <span class="trust-pill">Plain-English answers</span>
                <span class="trust-pill">Policy-focused responses</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="upload-panel-title">Upload your policy</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="upload-panel-copy">Choose one or more files. The assistant will prepare them automatically.</div>',
                unsafe_allow_html=True,
            )
            uploaded_files = st.file_uploader(
                "Policy files",
                type=["pdf", "txt", "md"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            st.caption("Supported files: PDF, TXT, MD")

            if st.button("Upload Files", type="primary", disabled=not uploaded_files, use_container_width=True):
                try:
                    with st.spinner("Preparing your policy assistant. This may take a moment..."):
                        saved_count = save_and_prepare_files(uploaded_files)

                    if saved_count:
                        st.session_state.messages = []
                        st.session_state.policy_ready = True
                        st.rerun()
                    else:
                        st.warning("No supported policy files were uploaded.")
                except Exception as exc:
                    st.error(f"Could not upload files: {exc}")

    st.stop()


st.title("Insurance Policy QA Bot")
st.caption("Ask questions about the policy files you uploaded.")

with st.sidebar:
    st.header("Policy Files")
    if st.button("Upload Different Files", use_container_width=True):
        st.session_state.policy_ready = False
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about coverage, exclusions, claims, waiting periods...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching policy documents..."):
                response = ask_policy_question(question)
            content = response["content"]
        except AuthenticationError:
            content = (
                "Groq rejected the API key. The key is loaded from `.env`, "
                "but Groq says it is invalid. Generate a new Groq API key, "
                "replace `GROQ_API_KEY` in `.env`, then restart Streamlit."
            )
        except Exception as exc:
            content = f"Could not answer yet: {exc}"

        st.markdown(content)
        st.session_state.messages.append({"role": "assistant", "content": content})
