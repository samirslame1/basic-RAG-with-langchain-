# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # usado por langchain_openai
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# Podrías agregar otros: persistencia de Chroma, nombre de colección, etc.
