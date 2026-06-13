# ==========================================
# TRAIN ML MODEL USING VADER LABELS
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

print("Loading labeled dataset...")

df = pd.read_csv(
    "data/reviews_with_sentiment.csv"
)

df = df.dropna()

X = df["reviews.text"]
y = df["sentiment"]

print("Vectorizing text...")

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,2),
    stop_words="english"
)

X_vector = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vector,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Logistic Regression...")

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(X_test)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        pred
    )
)

joblib.dump(
    model,
    "models/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel Saved Successfully")