from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# initialize embedding
embedding = OllamaEmbeddings(model="nomic-embed-text")

## Sample data ahe natr  indian goverment legal bns laws add karache
texts = [
    "A valid contract under Indian law requires offer, acceptance, and lawful consideration.",
    "An agreement without consideration is void under Indian Contract Act.",
    "Contracts cannot override court orders or legal obligations.",
    "Any clause restricting compliance with court orders is illegal and unenforceable.",
    "Lease agreements must include duration, rent, termination clause, and jurisdiction.",
    "Unreasonably long contract durations may be considered invalid or unfair.",
    "A contract must not contain illegal or impossible conditions.",
    "Absence of termination clause increases legal risk.",
]

# create FAISS DB (in-memory)
vector_db = FAISS.from_texts(texts, embedding)

# retrieval function
def retrieve_context(query: str):
    docs = vector_db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])