import os, json
from pathlib import Path
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from .prompt import PROMPT_TEMPLATE
from .ingest import DB, COLLECTION

_model=None
_collection=None

def resources():
    global _model,_collection
    if _model is None:
        _model=SentenceTransformer("all-MiniLM-L6-v2")
    if _collection is None:
        client=chromadb.PersistentClient(path=str(DB))
        _collection=client.get_or_create_collection(COLLECTION)
    return _model,_collection

def retrieve_chunks(query, top_k=3):
    model, collection=resources()
    qvec=model.encode(query,normalize_embeddings=True).tolist()
    result=collection.query(query_embeddings=[qvec],n_results=top_k,
                            include=["documents","metadatas","distances"])
    out=[]
    for i,doc in enumerate(result["documents"][0]):
        meta=result["metadatas"][0][i]
        out.append({"id":meta.get("chunk_id",meta.get("source","unknown")),
                    "document":doc,"distance":result["distances"][0][i]})
    return out

def _mock_policy(chunks):
    top=chunks[0]
    return {"answer":f"Based on the retrieved context: {top['document'][:200]}",
            "sources":[c["id"] for c in chunks],"confidence":1.0}

def _mock_general():
    return {"answer":"I can only answer questions about Zepto policies right now.",
            "sources":[],"confidence":1.0}

def _groq_json(prompt, retries=2):
    key=os.getenv("GROQ_API_KEY")
    if not key:
        return {"answer":"Real LLM mode requires GROQ_API_KEY.","sources":[],"confidence":0.0}
    url="https://api.groq.com/openai/v1/chat/completions"
    body={"model":os.getenv("GROQ_MODEL","llama-3.1-8b-instant"),
          "messages":[{"role":"user","content":prompt}],
          "temperature":0}
    last=None
    for attempt in range(retries+1):
        r=requests.post(url,headers={"Authorization":f"Bearer {key}"},json=body,timeout=30)
        r.raise_for_status()
        raw=r.json()["choices"][0]["message"]["content"]
        try:
            data=json.loads(raw)
            if {"answer","sources","confidence"} <= data.keys():
                data["confidence"]=float(data["confidence"])
                return data
        except Exception as e:
            last=e
        body["messages"].append({"role":"user","content":
            "Corrective instruction: return ONLY valid JSON with answer, sources and confidence. No markdown."})
    return {"answer":f"LLM output validation failed: {last}","sources":[],"confidence":0.0}

def answer_policy(query,chunks):
    if os.getenv("MOCK_LLM","1")!="0":
        return _mock_policy(chunks)
    context="\n\n".join(f"[{c['id']}] {c['document']}" for c in chunks)
    result=_groq_json(PROMPT_TEMPLATE.format(query=query,context=context))
    result["sources"]=result.get("sources") or [c["id"] for c in chunks]
    return result

def answer_general(query):
    if os.getenv("MOCK_LLM","1")!="0":
        return _mock_general()
    return _groq_json(PROMPT_TEMPLATE.format(query=query,context="No retrieval context is needed for this general question."))
