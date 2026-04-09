from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag import retrieve_context, ask_ai
from dotenv import load_dotenv
import os
import json
import re


load_dotenv()

app = FastAPI()

API_SECRET = os.getenv("AI_SECRET")


llm = OllamaLLM(model="llama3:8b")


def verify_token(authorization: str = Header(None)):
    if authorization != f"Bearer {API_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized by ai service")


class AnalyzeRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    message: str



@app.post("/analyze")
def analyze(data: AnalyzeRequest, authorization: str = Header(None)):

    verify_token(authorization)

    # Large Text will be spilited for chucking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    print("This is data come form backend node",data)

    chunks = splitter.split_text(data.text)[:3]
    combined_text = "\n".join(chunks)

    #  RAG context
    context = retrieve_context("legal document validation rules " + combined_text)
    print()

    prompt = f"""
You are an expert legal AI system.

Return ONLY valid JSON. No markdown or explanation.

Scoring Guidelines:
- 90–100 → Complete, strong legal document
- 70–89 → Good but missing minor clauses
- 50–69 → Moderate issues, missing important clauses
- 30–49 → High risk, many important clauses missing
- 0–29 → Invalid or legally unusable document

Rules:
- NEVER return 0 unless document is empty or completely invalid
- Missing 1–2 clauses → small reduction
- Missing 3–4 clauses → moderate reduction
- Missing critical clauses (liability, indemnity, dispute resolution) → reduce score but NOT below 40 unless many are missing

Required Clauses:
- Scope of Services
- Payment Terms
- Duration
- Confidentiality
- Termination
- Governing Law
- Liability Clause
- Indemnity Clause
- Dispute Resolution

Output format:
{{
  "score": number,
  "missingFields": ["string"],
  "riskLevel": "LOW" | "MEDIUM" | "HIGH",
  "summary": "short explanation",
  "decision": "APPROVE" | "REJECT",
  "feedback": "short suggestion"
}}

Context:
{context}

Document:
{combined_text}
"""

   # call for ll,
    response = llm.invoke(prompt)
    print("RAW LLM RESPONSE:", response)

 
    try:
        json_match = re.search(r"\{[\s\S]*\}", response)

        if not json_match:
            raise ValueError("No JSON found in response")

        parsed = json.loads(json_match.group(0))

        # 🔥 SAFETY: Prevent unrealistic 0 score
        if parsed.get("score") is not None:
            parsed["score"] = max(parsed["score"], 20)

    except Exception as e:
        print(" This is Error from Ai service Please check json error:", e)

        return {
            "score": None,
            "riskLevel": None,
            "missingFields": [],
            "summary": None,
            "decision": None,
            "feedback": None,
            "raw": response  # Remove natr fakt 
        }

    return parsed


@app.post("/ai/chat")
def chat(req: ChatRequest, authorization: str = Header(None)):

    verify_token(authorization)

    response = ask_ai(req.message)

    return {"response": response}