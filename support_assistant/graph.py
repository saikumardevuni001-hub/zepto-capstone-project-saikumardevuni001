import os
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from .rag import retrieve_chunks, answer_policy, answer_general

class GraphState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float

class SupportResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

def classify_intent(state: GraphState):
    q=state["query"].lower()
    keywords=["delivery","return","refund","membership","tracking","cancel","gift card","support hours"]
    intent="policy_question" if any(k in q for k in keywords) else "general_question"
    return {"intent": intent}

def retrieve_and_answer(state: GraphState):
    chunks=retrieve_chunks(state["query"], top_k=3)
    result=answer_policy(state["query"], chunks)
    return result

def direct_answer(state: GraphState):
    return answer_general(state["query"])

def route(state: GraphState) -> Literal["retrieve_and_answer","direct_answer"]:
    return "retrieve_and_answer" if state["intent"]=="policy_question" else "direct_answer"

def build_graph():
    graph=StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.add_edge(START,"classify_intent")
    graph.add_conditional_edges("classify_intent",route)
    graph.add_edge("retrieve_and_answer",END)
    graph.add_edge("direct_answer",END)
    return graph.compile()
