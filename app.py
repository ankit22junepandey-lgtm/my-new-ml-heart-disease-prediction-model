# app.py
from flask import Flask, request, jsonify
import joblib
import traceback

app = Flask(__name__)

# Load model on startup
MODEL_PATH = "model.joblib"
model = joblib.load(MODEL_PATH)

@app.route("/")
def index():
    return {"status": "ok", "msg": "Model server running"}

@app.route("/predict/", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        # Example expects: {"title": "...", "text": "..."}
        text = (data.get("title","") + " " + data.get("text","")).strip()
        # Preprocess the text exactly like during training:
        # e.g., vectorize = joblib.load("vectorizer.joblib")
        # X = vectorize.transform([text])
        # pred = model.predict(X)[0]
        # For demonstration, assume model accepts raw text:
        pred = model.predict([text])[0]
        # If you have probabilities:
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([text]).max()
        return jsonify({"prediction": str(pred), "confidence": float(prob) if prob is not None else None})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
