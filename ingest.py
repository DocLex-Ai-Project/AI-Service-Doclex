from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os
embedding = OllamaEmbeddings(model="nomic-embed-text")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = []

data_folder = "./data"

for file in os.listdir(data_folder):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(data_folder, file))
        documents = loader.load()
        chunks = splitter.split_documents(documents)
        docs.extend(chunks)

vector_db = FAISS.from_documents(docs, embedding)

vector_db.save_local("faiss_index")

print("Legal dataset loaded into FAISS")