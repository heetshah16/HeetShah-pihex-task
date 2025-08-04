import requests
import json

API_URL = "http://localhost:8000/ask"
EVAL_FILE = "eval_questions.jsonl"

def ask_question(question):
    try:
        response = requests.post(API_URL, json={"question": question})
        return response.json() if response.status_code == 200 else {"error": response.text}
    except Exception as e:
        return {"error": str(e)}

def run_eval():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            question = entry.get("question")
            print(f"\nQuestion: {question}")
            result = ask_question(question)
            if "error" in result:
                print("❌ Error:", result["error"])
            else:
                print("✅ Answer:", result.get("answer"))
                print("   Category:", result.get("category"))
                print("   Confidence:", result.get("confidence"))
                print("   Sources:", [s['doc'] for s in result.get("sources", [])])

if __name__ == "__main__":
    run_eval()
