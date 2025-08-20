from dotenv import load_dotenv
load_dotenv()

from loaders import load_pdf
from preprocess import make_qa_chunks_from_docs
from vectorstore import vector_store, index_documents  # tu instancia global y función
from rag import make_graph, build_llm, build_prompt
import time

PDF_PATH = "data/Preguntas-Redes-I.pdf"

docs = load_pdf(PDF_PATH)
qa_chunks = make_qa_chunks_from_docs(docs)
index_documents(qa_chunks)

llm = build_llm()
rag_prompt = build_prompt()
graph = make_graph(vector_store, llm=llm, rag_prompt=rag_prompt)

for msg, meta in graph.stream(
    {"question": "¿Para qué sirve el probador de cable UTP?"},
    stream_mode="messages",  # tokens/mensajes del nodo generate
):
    # msg es un ChatMessage con .content (texto parcial)
    print(msg.content, end="", flush=True)
    time.sleep(0.09)
print()  # salto de línea al final

