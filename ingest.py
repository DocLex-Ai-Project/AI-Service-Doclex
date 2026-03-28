from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os

DATA_PATH = "data/"

documents = []

# Load files
for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} documents")

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

docs = splitter.split_documents(documents)

print(f"Split into {len(docs)} chunks")

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# FAISS
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

print("✅ FAISS index created successfully")