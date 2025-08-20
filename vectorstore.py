# vectorstore.py
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List
from langchain_core.documents import Document

# inicializamos embeddings una sola vez
embeddings = OpenAIEmbeddings()

# y el store
vector_store = InMemoryVectorStore(embeddings)

def index_documents(docs: List[Document]):
    """Agrega documentos al vector store y devuelve los IDs asignados."""
    return vector_store.add_documents(docs)

