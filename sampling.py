import os
import json
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from jsonschema import validate, ValidationError
import numpy as np
import glob

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L12-v2")

# Load documents
kb_dir = "./kb_docs"
knowledge_base = []

for md_path in glob.glob(os.path.join(kb_dir, "*.md")):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
        chunks = content.split("\n\n")  # Simple paragraph split
        for chunk in chunks:
            if chunk.strip():
                knowledge_base.append({
                    "filename": os.path.basename(md_path),
                    "text": chunk.strip(),
                    "embedding": embedder.encode(chunk.strip())
                })

# Load schema
with open("./answer_schema.json", "r") as f:
    answer_schema = json.load(f)

# FastAPI app
app = FastAPI()

class AskRequest(BaseModel):
    question: str

def get_top_k_snippets(question: str, k: int = 3):
    q_embed = embedder.encode(question)
    sims = [cosine_similarity([q_embed], [doc["embedding"]])[0][0] for doc in knowledge_base]
    top_indices = np.argsort(sims)[-k:][::-1]
    return [(knowledge_base[i], float(sims[i])) for i in top_indices if sims[i] > 0.4]  # basic threshold

@app.post("/ask")
async def ask(req: AskRequest):
    snippets = get_top_k_snippets(req.question)

    answer = "Based on the documentation, here's what we found: " + " ".join([s[0]['text'] for s in snippets])[:300]
    category = infer_category(snippets)
    confidence = round(min(1.0, sum(s[1] for s in snippets) / len(snippets)) if snippets else 0.3, 2)

    sources = [{"doc": s[0]["filename"], "snippet": s[0]["text"][:100]} for s in snippets]

    response = {
        "answer": answer,
        "category": category,
        "confidence": confidence,
        "sources": sources
    }

    try:
        validate(instance=response, schema=answer_schema)
    except ValidationError as e:
        return {"error": "Schema validation failed", "details": str(e)}

    return response

def infer_category(snippets):
    if not snippets:
        return "other"
    docname = snippets[0][0]["filename"].lower()
    if "api" in docname:
        return "api"
    elif "security" in docname:
        return "security"
    elif "pricing" in docname:
        return "pricing"
    elif "faq" in docname or "troubleshooting" in docname:
        return "support"
    else:
        return "other"

if __name__ == "__main__":
    uvicorn.run("sampling:app", host="0.0.0.0", port=8000, reload=True)
