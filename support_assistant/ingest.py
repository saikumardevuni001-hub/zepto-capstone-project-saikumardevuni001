from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE=Path(__file__).resolve().parent
DOCS=BASE/"docs"
DB=BASE/"chroma_db"
COLLECTION="zepto_policies"

def build_index():
    client=chromadb.PersistentClient(path=str(DB))
    collection=client.get_or_create_collection(COLLECTION)
    model=SentenceTransformer("all-MiniLM-L6-v2")
    ids=[]; docs=[]; metas=[]; vectors=[]
    for path in sorted(DOCS.glob("doc_*.txt")):
        text=path.read_text(encoding="utf-8").strip()
        ids.append(path.stem)
        docs.append(text)
        metas.append({"source":path.name,"chunk_id":path.stem})
        vectors.append(model.encode(text,normalize_embeddings=True).tolist())
    if ids:
        collection.upsert(ids=ids,documents=docs,metadatas=metas,embeddings=vectors)
    return len(ids)

if __name__=="__main__":
    print(f"Indexed {build_index()} documents into ChromaDB.")
