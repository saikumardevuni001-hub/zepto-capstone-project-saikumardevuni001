# Module 3 — Support Assistant

## Baseline

The graded baseline is completely local and deterministic. Leave `MOCK_LLM` unset or set it to `1`.

### Build the index

```bash
python -m support_assistant.ingest
```

This loads all eight required policy documents, embeds them using `all-MiniLM-L6-v2`, and stores vectors in the ChromaDB collection `zepto_policies`.

### Run API

From repository root:

```bash
uvicorn support_assistant.main:app --host 127.0.0.1 --port 7860
```

Example policy request:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{\"query\":\"How long does delivery take?\"}"
```

Example general request:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the capital of India?\"}"
```

Representative baseline JSON shapes:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials...","sources":["doc_01"],"confidence":1.0}
```

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## Architecture

```text
INGESTION
8 policy .txt files
      |
      v
EMBEDDING
SentenceTransformer(all-MiniLM-L6-v2)
      |
      v
VECTOR STORE
ChromaDB / zepto_policies
      |
      v
QUERY
FastAPI /ask
      |
      v
LANGGRAPH
classify_intent
   |                 |
policy_question   general_question
   |                 |
   v                 v
retrieve_and_answer direct_answer
   |
   v
top-3 Chroma retrieval
   |
   v
GENERATION
MOCK_LLM=1 -> deterministic local template
MOCK_LLM=0 -> optional real LLM + JSON validation/retries
```

Ingestion and embedding are handled by `ingest.py`. Retrieval is handled by `retrieve_chunks()` in `rag.py` and runs in both mock and real modes. `classify_intent`, `retrieve_and_answer`, and `direct_answer` are the three LangGraph nodes. The final structured response is validated by the Pydantic `SupportResponse` model.

The prompt in `prompt.py` follows role-context-task-format-length, includes a negative constraint, and includes a few-shot example. The optional real-LLM path is controlled by `MOCK_LLM=0`; its raw JSON is validated and can retry with a corrective instruction up to two additional times.

## Docker

From repository root:

```bash
docker build -t zepto-support ./support_assistant
docker run --rm -p 7860:7860 zepto-support
```

Then call `POST http://localhost:7860/ask`.

No API key is required for the graded mock mode.
