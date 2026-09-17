# RAG Local Business Assistant

A Retrieval-Augmented Generation API for discovering hotels, spas, and restaurants in Finland. The service combines structured local data, Pinecone search, LLM-generated metadata filters, and Groq response generation to answer natural-language questions with grounded business information.

Example questions:

- `Find a hotel in Helsinki with a swimming pool`
- `Which spa in Espoo offers sports massage?`
- `Find a restaurant in Vantaa serving biryani and seekh kebab`

## Features

### Retrieval-Augmented Generation

- Loads hotel, massage, and restaurant records from the `data/` directory.
- Converts records into Pinecone-compatible records with searchable text and metadata.
- Creates or connects to a Pinecone index named `rag-hotel`.
- Uses Pinecone hosted `llama-text-embed-v2` embeddings through the integrated model index API.
- Retrieves the five most relevant records from the `services-providers` namespace with hybrid search.
- Builds a grounded answer prompt from retrieved business context.

### BM25 and hybrid search

- Maintains a dense vector index (`rag-hotel`) for semantic similarity search.
- Maintains a separate Pinecone full-text index (`rag-hotel-fts`) for BM25 keyword search over `chunk_text`.
- Upserts every prepared record into both indexes in the `services-providers` namespace.
- Applies the same LLM-generated metadata filter to both dense and BM25 searches.
- Merges and deduplicates candidates by record ID, then reranks the combined set with Pinecone's `bge-reranker-v2-m3` model.
- Returns up to five reranked records to the answer-generation prompt. This combines semantic matches with exact terms such as city names, services, and restaurant dishes.

### Unified query processing

- Uses a single Groq call via `QueryProcessor` to combine safety checking, category detection, and filter extraction.
- Recognizes the categories `hotel`, `spa`, and `restaurant`.
- Extracts category, name, city, region, rating, distance, and requested services in one structured JSON response.
- Maps user terms to known services such as `Swimming Pool`, `Hot Stone Massage`, `Biryani`, and `Chocolate Cake`.
- Applies the generated filters to Pinecone metadata search.
- Reduces the old two-call pattern to one LLM invocation: one call for guardrails + parsing instead of separate guardrail and query-creator calls.

### Query guardrails (legacy / obsolete)

- The previous standalone guardrail module and query builder are now superseded by the unified `QueryProcessor`.
- The deprecated flow is retained only for reference and migration; active request handling now uses the combined parser/guardrail logic.
- Clear violations still result in immediate refusal before retrieval, but the decision is made inside the same model call as the structured filter generation.

### Conversation awareness

- Stores recent query and response pairs in a `HistoryManager`.
- Keeps a sliding window of recent conversation turns.
- Uses recent conversation when building retrieval queries and answer prompts.
- Folds turns that leave the window into an LLM-generated rolling summary.
- Preserves user preferences, locations, requested services, selected places, and unresolved questions in the summary.


### API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/` | Search the local business data and generate an answer. |
| `GET` | `/health` | Return the service health status. |
| `GET` | `/eval` | Evaluation endpoint placeholder. |

## Request and response

Send a natural-language query to `POST /`:

```json
{
	"query": "Find a hotel in Helsinki with a pool"
}
```

Response:

```json
{
	"response": "..."
}
```

Conversation turns are currently maintained by the server-side `HistoryManager`. The client does not send a history field in the current request model.

## Request flow

```mermaid
flowchart TD

    A[User Query] --> B[FastAPI]
    B --> C[HistoryManager]
    B --> QP[QueryProcessor]

    QP -- Blocked --> BR[Refusal Response]
    QP -- Allowed --> D[Structured Pinecone Filter]

    subgraph QUERY["Unified Query Processing"]
        C --> F[History-Aware Retrieval Query]
        D --> E[Hybrid Search Filter]
    end

    subgraph RETRIEVAL["Hybrid Retrieval"]
        E --> G[Dense Vector Search]
        E --> H[BM25 Full-Text Search]
        F --> G
        F --> H
        G --> I[Merge & Deduplicate]
        H --> I
        I --> J[Cross-Encoder Reranking]
        J --> K[Retrieved Business Context]
    end

    K --> L[Grounded Prompt]
    O --> L

    subgraph GENERATION["Answer Generation"]
        L --> M[LLM Answer Model]
        M --> N[Response]
    end

    subgraph CONTEXT["Conversation Context"]
        C --> O[Conversation Context & Rolling Summary]

        N --> P[Add Turn to HistoryManager]

        P --> Q{Window Exceeds 6 Turns?}

        Q -- Yes --> R[LLM Summarization Model]
        R --> S[Update Rolling Summary]
        S --> O
    end

    Q -- No --> O
```

> The old standalone `guardrails.py` and `query_creator.py` flow is obsolete. Active routing now uses a single `QueryProcessor` LLM call that performs both policy enforcement and structured filter extraction.

 
## Technology stack

- Python
- FastAPI
- Pydantic
- Uvicorn
- Pinecone
- Pinecone (dense vector search, BM25 full-text search, and reranking)
- Groq (LLM)
- LangChain Groq integration
- Pytest 

## Project structure

```text
app/
	appInitlize.py       Configuration, Pydantic models, and service initialization
	dataBase.py          Pinecone connection, index setup, upsert, and search
	eval.py              Evaluation module placeholder
	evalData.py          Evaluation query and reference data
	guardrails.py        Legacy guardrail implementation (obsolete)
	history_manager.py   Recent conversation window and rolling summary
	logger.py            Application logging
	main.py              FastAPI application and request orchestration
	models.py            Groq model wrapper
	prepareData.py       Source-record transformation for Pinecone
	prompt.py            Context and answer prompt construction
	query_creator.py     Legacy query parser (obsolete)
	queryProcessor.py    Unified guardrail + query parsing LLM processor
	readData.py          Local data loading

data/
	hotels.txt
	massage.txt
	restaurants.txt

test/
	database_test.py
	liberies_test.py
	main_test.py
	model_test.py
```

## Configuration

Create a local `config.ini` file with the API keys expected by the application:

```ini
[KEYS]
groq_api_key = your-groq-api-key
pinecone_api_key = your-pinecone-api-key
```

Do not commit credentials. Rotate any key that has been exposed in source control or logs.

## Installation

From the repository root:

```powershell
python -m venv RAGENV
.\RAGENV\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the API

```powershell
.\RAGENV\Scripts\python.exe -m uvicorn app.main:app --reload
```

## Run with Docker

Build the image from the repository root:

```powershell
docker build -t rag-pinecone .
```

Run the API with the local configuration mounted read-only:

```powershell
docker run --rm -p 8000:8000 -v "${PWD}\config.ini:/app/config.ini:ro" rag-pinecone
```

Open the interactive API documentation at:

http://127.0.0.1:8000/docs

Example PowerShell request:

```powershell
Invoke-RestMethod `
	-Uri http://127.0.0.1:8000/ `
	-Method Post `
	-ContentType "application/json" `
	-Body '{"query":"Find a hotel in Helsinki with a pool"}'
```

## Tests

Run the test suite with:

```powershell
.\RAGENV\Scripts\python.exe -m pytest
```

The tests cover the database, model wrapper, and API behavior. Conversation-specific tests should be expanded as session isolation and persistence are added.

## Project status

This is a learning and portfolio project focused on RAG pipeline design, vector search, prompt engineering, LLM integration, and evaluation experiments. It is not currently intended as a production multi-user service.

