from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load model dan tokenizer
model_name = "Ynanaaaaaa/W.E.D.I"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})

@app.route("/")
def home():
    return """
    <h2>W.E.D.I Chatbot API</h2>
    <p>Kirim POST ke <code>/chat</code> dengan JSON <code>{"message": "halo"}</code></p>
    """

if __name__ == "__main__":
    app.run(debug=True)