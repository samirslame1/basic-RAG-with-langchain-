# loaders.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List

def load_pdf(path: str) -> List[Document]:
    loader = PyPDFLoader(path)
    return loader.load()
