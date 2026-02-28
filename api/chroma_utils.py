from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=160,
    length_function=len
)

embedding_function = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_function
)


def get_retriever_for_user(user_id: str):
    # Filter at query time so each user only searches their own documents
    return vectorstore.as_retriever(
        search_kwargs={"k": 10, "filter": {"user_id": user_id}}
    )


def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".html"):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return text_splitter.split_documents(loader.load())


def index_document_to_chroma(file_path: str, file_id: int, user_id: str = "default") -> bool:
    try:
        splits = load_and_split_document(file_path)
        for split in splits:
            split.metadata["file_id"] = file_id
            split.metadata["user_id"] = user_id
        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


def delete_doc_from_chroma(file_id: int, user_id: str = "default") -> bool:
    try:
        vectorstore._collection.delete(
            where={"$and": [
                {"file_id": {"$eq": file_id}},
                {"user_id": {"$eq": user_id}}
            ]}
        )
        return True
    except Exception as e:
        print(f"Error deleting file {file_id}: {e}")
        return False