from flask import Flask, request, jsonify
from langchain_mistralai.chat_models import ChatMistralAI
import os

app = Flask(__name__)

def get_chat_model():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in the environment")
    # Initialize the ChatMistralAI with desired parameters
    model = ChatMistralAI(
        api_key=api_key,
        model="mistral-large-latest",
        temperature=0,
        max_retries=2
    )
    return model

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    messages = data.get("messages")  # Expecting a list of (role, message) tuples
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    model = get_chat_model()
    response = model.invoke(messages)
    return jsonify({"response": response.content})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
