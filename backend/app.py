import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Set GOOGLE_API_KEY in backend/.env")

genai.configure(api_key=api_key)

# You can swap to "gemini-1.5-pro" if you have access; flash is cheaper/faster.
MODEL_NAME = "gemini-1.5-flash"

app = Flask(__name__)
CORS(app)

SYSTEM_INSTRUCTIONS = """You are Prompt Companion: a concise, helpful assistant.
You can act in modes: summarization, code_help, grammar_check, general.
Always keep outputs short and structured. When giving code, prefer minimal, runnable snippets.
"""

def build_prompt(category: str, user_text: str):
    cat = (category or "auto").lower()
    if cat == "summarization":
        return f"""Task: Summarize the user's text clearly in 3–5 bullet points.
Text:
{user_text}
Output: concise bullets, no opinion."""
    elif cat == "code_help":
        return f"""Task: Provide practical coding help. If code is asked, give a minimal working example.
User message:
{user_text}
Output: short explanation + code block if relevant."""
    elif cat == "grammar_check":
        return f"""Task: Correct grammar and clarity while preserving meaning.
Text:
{user_text}
Output: 1) Corrected version 2) Brief note of key changes."""
    elif cat == "general":
        return f"""Task: Answer helpfully and concisely.
Message:
{user_text}"""
    else:
        # auto-detect: ask Gemini to pick the best mode first, then answer
        return f"""You can respond as summarization, code_help, grammar_check, or general.
First, infer the best mode for the message, then produce the response for that mode.
Message:
{user_text}
In the first line, include a tag like [mode:summarization] (no extra text before it)."""

@app.post("/api/respond")
def respond():
    data = request.get_json(force=True)
    user_text = (data.get("prompt") or "").strip()
    category = (data.get("category") or "auto").strip().lower()
    if not user_text:
        return jsonify({"error": "Empty prompt"}), 400

    generation_config = {
        "temperature": 0.4,
        "top_p": 0.9,
        "top_k": 32,
        "max_output_tokens": 1200,
        "response_mime_type": "text/plain"
    }

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTIONS,
        generation_config=generation_config
    )

    prompt = build_prompt(category, user_text)
    try:
        resp = model.generate_content(prompt)
        text = resp.text or "(no response)"
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Extract detected mode if auto
    detected = None
    if category == "auto" and text.startswith("[mode:"):
        # e.g., [mode:summarization] ...
        try:
            first_line, rest = text.split("\n", 1)
            detected = first_line.strip("[]").split(":", 1)[1]
            text = rest.strip()
        except Exception:
            detected = "general"

    return jsonify({
        "category": category,
        "detected": detected,
        "response": text
    })
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
