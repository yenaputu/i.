from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load model
model_name = "microsoft/DialoGPT-small"  # Ganti dengan model kamu jika berhasil
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Pastikan pad token ada
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    # Encode input
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=200,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

@app.route("/")
def home():
    return "<h2>W.E.D.I Chatbot API</h2><p>Kirim POST ke /chat dengan JSON {'message': 'halo'}</p>"

if __name__ == "__main__":
    app.run(debug=True)