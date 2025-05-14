import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pickle

# Загрузка данных
true_news = pd.read_csv("dataset/True.csv")
fake_news = pd.read_csv("dataset/Fake.csv")

# Разметка данных: 0 = настоящая новость, 1 = фейк
true_news["label"] = 0
fake_news["label"] = 1

# Объединение и перемешивание
data = pd.concat([true_news, fake_news]).sample(frac=1, random_state=42)
X = data["text"] 
y = data["label"]

# Векторизация текста
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_tfidf = vectorizer.fit_transform(X)

# Обучение модели
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=100,
    random_state=42
)
model.fit(X_tfidf, y)

# Сохранение модели и векторизатора
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Модель и векторизатор сохранены в model.pkl и vectorizer.pkl")