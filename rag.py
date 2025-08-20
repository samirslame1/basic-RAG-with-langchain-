# rag.py
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph

template_es = """
Eres un asistente para tareas de preguntas y respuestas.
Usa únicamente los siguientes fragmentos de contexto recuperados para responder a la pregunta.
Si no sabes la respuesta con el contexto dado, responde claramente que no lo sabes.
Mantén la respuesta breve y clara, con un máximo de tres oraciones.

Pregunta: {question}
Contexto:
{context}

Respuesta:
""".strip()

def build_prompt():
    return PromptTemplate.from_template(template_es)

def build_llm(model: str = "gpt-4o-mini"):
    return ChatOpenAI(model=model)  # usa OPENAI_API_KEY del entorno

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def make_graph(vector_store, llm=None, rag_prompt=None):
    llm = llm or build_llm()
    rag_prompt = rag_prompt or build_prompt()

    # === Tus funciones, cerradas sobre vs/llm/prompt ===
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"], k=3)
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = rag_prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Armar el grafo
    gb = StateGraph(State).add_sequence([retrieve, generate])
    gb.add_edge(START, "retrieve")
    return gb.compile()