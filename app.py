from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Загрузка модели и векторизатора
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        text = request.form["news_text"]
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]
        result = "Фейк!" if prediction == 1 else "Настоящая новость."
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)