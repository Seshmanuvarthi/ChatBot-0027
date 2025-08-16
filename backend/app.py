
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import google.generativeai as genai
import os
import json
from google.api_core.exceptions import ResourceExhausted


load_dotenv()
app = Flask(__name__,
            template_folder='../frontend',
            static_folder='../frontend')


api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
  
    raise ValueError("GOOGLE_API_KEY not found in environment variables. "
                     "Please create a .env file with your key.")
genai.configure(api_key=api_key)

@app.route("/")
def home():
    """Renders the main HTML page."""
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """
    Handles the user's request, sends it to the Gemini API with a strict JSON
    output configuration, and returns the response.
    """
    try:
        data = request.get_json()
        user_input = data.get("message", "")

        if not user_input.strip():
            return jsonify({"error": "Message cannot be empty"}), 400

        prompt = f"""
        Classify the following user input into one of these categories:
        - Grammar Correction
        - Code Help
        - Summarization
        - General Question

        Then, give the improved/solved/final response accordingly. If the response
        is code, wrap it in a markdown code block (e.g., ```python\n...code...\n```).

        User Input: {user_input}
        """

        generation_config = {
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "OBJECT",
                "properties": {
                    "category": {"type": "STRING"},
                    "answer": {"type": "STRING"}
                }
            }
        }

  
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=generation_config
        )
        response = model.generate_content(prompt)

        result = json.loads(response.text.strip())
        return jsonify(result)

    except ResourceExhausted as e:
 
        return jsonify({
            "error": "API rate limit exceeded. Please wait a moment and try again, "
                     "or check your API key's quota limits."
        }), 429
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Failed to parse model response: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
