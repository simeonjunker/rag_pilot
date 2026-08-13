from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.config import CHROMA_DIR


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
)

vector_db = Chroma(
    collection_name="wikipedia",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)