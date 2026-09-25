from fastapi import FastAPI
from pydantic import BaseModel
from .graph import build_graph, GraphState
from .ingest import build_index
from .graph import SupportResponse

app=FastAPI(title="Zepto Support Assistant")
graph=build_graph()

class AskRequest(BaseModel):
    query: str

@app.on_event("startup")
def startup():
    build_index()

@app.post("/ask",response_model=SupportResponse)
def ask(req: AskRequest):
    state:GraphState={"query":req.query}
    result=graph.invoke(state)
    return SupportResponse(
        answer=result["answer"],
        sources=result.get("sources",[]),
        confidence=float(result.get("confidence",0))
    )
