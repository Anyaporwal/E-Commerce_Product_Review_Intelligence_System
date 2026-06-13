import joblib

model = joblib.load(
    "models/sentiment_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

while True:

    review = input(
        "\nEnter review: "
    )

    if review.lower() == "exit":
        break

    review_vector = (
        vectorizer.transform(
            [review]
        )
    )

    prediction = (
        model.predict(
            review_vector
        )
    )[0]

    print(
        "Prediction:",
        prediction
    )