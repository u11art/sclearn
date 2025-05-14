from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Загрузка модели и векторизатора
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/api/check", methods=["POST"])
def check_news():
    try:
        data = request.get_json()
        text = data.get("news_text")
        
        if not text:
            return jsonify({"error": "No news text provided"}), 400
        
        text_tfidf = vectorizer.transform([text])
        probabilities = model.predict_proba(text_tfidf)[0]
        prediction = probabilities[1] >= 0.5  # True если вероятность фейка >= 50%
        
        return jsonify({
            "is_fake": bool(prediction),
            "fake_probability": round(float(probabilities[1]) * 100, 2),  # вероятность в процентах
            "true_probability": round(float(probabilities[0]) * 100, 2)   # вероятность в процентах
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)