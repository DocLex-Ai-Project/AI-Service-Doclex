from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Load embeddings
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Load FAISS index
vector_db = FAISS.load_local(
    "faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

# Load LLM
llm = ChatOllama(model="llama3:8b")


# ✅ ADD THIS FUNCTION (IMPORTANT)
def retrieve_context(query: str):
    docs = vector_db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])


# ✅ EXISTING FUNCTION (KEEP)
def ask_ai(query: str):
    docs = vector_db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an Indian legal AI assistant.

- Understand English, Hindi, Marathi
- Reply in same language

Follow rules strictly:
- Use ONLY provided context
- If missing info → say "insufficient information"
- Highlight risks clearly

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content