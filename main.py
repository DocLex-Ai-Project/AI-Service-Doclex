from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag import retrieve_context
import json
import re

app = FastAPI()

llm = Ollama(model="llama3:8b")

class RequestData(BaseModel):
    text: str


@app.post("/analyze")
def analyze(data: RequestData):

    print("API HIT")

    # 🔹 Step 1: Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(data.text)
    chunks = chunks[:3]  # limit for performance
    combined_text = "\n".join(chunks)

    print("Chunks used:", len(chunks))

    # 🔹 Step 2: RAG context
    context = retrieve_context(combined_text)

    # 🔹 Step 3: STRONG PROMPT (LLM RULES)
    prompt = f"""
You are a strict legal AI system.

Follow these rules strictly:

1. If consideration is missing → contract is VOID → score must be below 40
2. If illegal clause exists → riskLevel = HIGH
3. If major clauses missing → reduce score significantly
4. Ensure score matches reasoning (NO contradictions)

Relevant legal context:
{context}

Document:
{combined_text}

Return ONLY JSON:
{{
  "score": number (0-100),
  "missingFields": [],
  "riskLevel": "LOW | MEDIUM | HIGH",
  "summary": "short explanation (max 40 words)"
}}
"""

    response = llm.invoke(prompt)

    # 🔹 Step 4: Extract JSON
    json_match = re.search(r"\{.*\}", response, re.DOTALL)

    if not json_match:
        return {"error": "Invalid AI response", "raw": response}

    try:
        parsed = json.loads(json_match.group(0))
    except:
        return {"error": "JSON parsing failed", "raw": response}

    # 🔹 Step 5: Normalize values
    score = parsed.get("score", 0)
    if score <= 1:
        score *= 100

    missing_fields = list(set([f.strip().lower() for f in parsed.get("missingFields", [])]))
    risk_level = parsed.get("riskLevel", "LOW")
    short_summary = parsed.get("summary", "")

    # 🔴 Step 6: BACKEND VALIDATION (VERY IMPORTANT)

    # Rule: missing consideration → VOID contract
    if "consideration" in missing_fields:
        score = min(score, 40)
        risk_level = "HIGH"

    # Rule: illegal clause detection
    if "illegal" in short_summary.lower() or "court" in short_summary.lower():
        score = min(score, 30)
        risk_level = "HIGH"

    # Rule: many missing clauses
    if len(missing_fields) >= 4:
        score = min(score, 50)

    # Clean formatting
    missing_fields = [f.title() for f in missing_fields]

    # 🔹 Step 7: Dynamic summary length
    input_length = len(data.text.split())

    if input_length < 200:
        target_words = 80
    elif input_length < 1000:
        target_words = 150
    elif input_length < 3000:
        target_words = 300
    else:
        target_words = 500

    # 🔹 Step 8: FINAL SUMMARY (2nd call)
    final_prompt = f"""
You are a legal AI assistant.

Generate a professional legal summary.

STRICT RULES:
- No headings or bullet points
- Clear paragraph format
- Legal tone
- Focus on risks and missing clauses
- Keep around {target_words} words

Text:
{short_summary}

Return ONLY plain text.
"""

    final_summary = llm.invoke(final_prompt)

    # 🔹 Final response
    return {
        "score": round(score, 2),
        "missingFields": missing_fields,
        "riskLevel": risk_level,
        "summary": final_summary.strip()
    }