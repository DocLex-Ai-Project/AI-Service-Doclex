import re
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama

# ==============================
# LOAD EMBEDDINGS
# ==============================

embedding = OllamaEmbeddings(model="nomic-embed-text")

# ==============================
# LOAD FAISS DB
# ==============================

vector_db = FAISS.load_local(
    "faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

# ==============================
# LOAD LLM
# ==============================

llm = ChatOllama(model="llama3:8b")


# ==============================
# DEBUG FUNCTION
# ==============================

def debug_db():
    print("Total documents:", len(vector_db.docstore._dict))


# ==============================
# SECTION DETECTION
# ==============================

def extract_section(query: str):
    match = re.search(r"section\s*(\d+)", query.lower())
    return match.group(1) if match else None


# ==============================
# LAW DETECTION
# ==============================

def detect_law(query: str):
    q = query.lower()

    if "company" in q or "section" in q:
        return "Companies Act 2013"
    elif "contract" in q or "agreement" in q:
        return "Indian Contract Act 1872"
    elif "fraud" in q or "forgery" in q or "cheating" in q:
        return "BNS Criminal Law"
    else:
        return "General Legal"


# ==============================
# RETRIEVE CONTEXT
# ==============================

def retrieve_context(query: str):
    section = extract_section(query)
    law = detect_law(query)

    if section:
        enhanced_query = f"SECTION {section} {law}"
    else:
        enhanced_query = f"{law} {query}"

    docs = vector_db.similarity_search(enhanced_query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    return context


# ==============================
# ASK AI FUNCTION
# ==============================

def ask_ai(query: str):
    context = retrieve_context(query)

    prompt = f"""
You are an Indian legal AI assistant.

Rules:
- Answer ONLY from CONTEXT
- Do NOT hallucinate
- If section not found → say "Section not available in dataset"
- If insufficient → say "insufficient information"
- Explain in simple legal language

Format:
- Section (if applicable)
- Explanation
- Key Points
- Risk (if any)

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content


# option fucntion for terinmal work only testing and debbugging
# if __name__ == "__main__":
#     debug_db()

#     while True:
#         query = input("\nEnter question: ")

#         if query.lower() in ["exit", "quit"]:
#             break

#         result = ask_ai(query)

#         print("\n==============================")
#         print("LEGAL AI RESPONSE")
#         print("==============================\n")
#         print(result)