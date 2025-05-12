import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


UPLOAD_DIR = "./uploaded_pdfs"
DB_DIR = "./faiss_index"

os.makedirs(UPLOAD_DIR, exist_ok=True)

def clear_faiss_index():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

def save_uploaded_files(uploaded_files):
    paths = []
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        paths.append(file_path)
    return paths

def embed_files_from_paths(file_paths):
    all_docs = []
    for path in file_paths:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = FAISS.from_documents(chunks, embedding)
    vectordb.save_local(DB_DIR)
    return True

def load_faiss_index():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={"device": "cpu"})
    vectordb = FAISS.load_local(DB_DIR, embedding, allow_dangerous_deserialization=True)
    return vectordb.as_retriever(search_kwargs={"k": 3})
