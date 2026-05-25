# Insurance Policy QA Bot

A Retrieval-Augmented Generation (RAG) chatbot for asking questions over insurance policy documents.

Built with:

- LangChain for loading, chunking, retrieval, and QA orchestration
- Chroma by default, with FAISS available as an optional vector store
- FastEmbed for local embeddings
- Groq for fast LLM responses
- Streamlit for a lightweight chat UI

## 1. Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add your `GROQ_API_KEY`.

## 2. Run The Web App

```powershell
streamlit run app.py
```

In the sidebar, upload policy files and click **Embed Uploaded Files**. Supported formats:

- `.pdf`
- `.txt`
- `.md`

After embedding finishes, ask questions in the chat box.

## 3. Optional Terminal Indexing

You can also put files directly in `data/policies` and build the vector index from the terminal:

```powershell
py -m src.ingest
```

By default this creates a Chroma index in `storage/chroma`. To use FAISS, set this in `.env`:

```env
VECTOR_STORE=faiss
```

Then run ingestion again.

If FAISS is not already installed, install the optional dependency first:

```powershell
pip install -r requirements-faiss.txt
```

## 4. Optional Terminal Chat

```powershell
py -m src.chat
```

## Notes

- Answers are constrained to the retrieved policy context.
- Each answer includes source snippets so users can verify where the answer came from.
- If the uploaded policies do not contain enough information, the bot is instructed to say so instead of guessing.
