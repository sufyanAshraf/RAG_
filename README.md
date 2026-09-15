# RAG  

A personal project focused on building a Retrieval-Augmented Generation (RAG) system with FastAPI, vector search, and language models.

This is a learning and portfolio project designed to explore how local business data can be indexed, retrieved, and used to generate grounded answers with an LLM.

## Demo

<video src="https://github.com/sufyanAshraf/ingest_doc/raw/refs/heads/main/demo.mp4" controls width="720"></video>

[Open the demo video](https://github.com/sufyanAshraf/ingest_doc/raw/refs/heads/main/demo.mp4) if the embedded player is not supported.
![Project Demo](https://github.com/sufyanAshraf/ingest_doc/raw/refs/heads/main/demo.gif)

## What this project does

- loads structured local data
- converts records into embeddings
- stores vectors in Pinecone 
- metadata store saperately
- retrieves relevant context with similarity search
- generates a final response using Groq
- includes a evaluation workflow for testing quality
- evaluation include:
- Context Precision
- Context recall
- Faithfulness
- Response Relevancy

## Tech stack

- Python
- FastAPI
- FAISS
- Hugging Face Embeddings
- LangChain
- Groq
- pytest
- Ragas

## Project status

This project is for personal learning, experimentation, and portfolio demonstration. It is not intended as a public product or reusable service for general external use.

## Local setup

```powershell
cd C:\Users\hp\work\ingest_doc
python -m venv RAGENV
.\RAGENV\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run locally

```powershell
cd C:\Users\hp\work\ingest_doc
.\RAGENV\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open:

- http://127.0.0.1:8000/docs

## Conversation context

The API is stateless. Send previous turns in `history` with each follow-up request:

```json
{
	"query": "What about one with a pool?",
	"history": [
		{"role": "user", "content": "Find a hotel in Helsinki"},
		{"role": "assistant", "content": "Here are some options."}
	]
}
```

History is limited to 20 messages, and each message is limited to 2,000 characters. Previous user messages are used for retrieval; both user and assistant messages are included when generating the answer. The response remains `{ "response": "..." }`.

## Notes

- API keys are kept in local config files only.
- This is a personal experimentation project.
- Do not expose this project or its data publicly without review.

## Purpose

The goal is to practice:

- RAG pipeline design
- embedding-based retrieval
- prompt construction
- LLM integration
- evaluation and experimentation

## License

For personal use and portfolio demonstration only.

