from analytics import (
    load_data,
    get_sentiment
)

print(
    "Loading reviews..."
)

df = load_data()

print(
    "Generating sentiment labels..."
)

df["sentiment"] = (
    df["reviews.text"]
    .apply(get_sentiment)
)

print()

print(
    df["sentiment"]
    .value_counts()
)

print()

print(
    "Dataset ready."
)

df.to_csv(
    "data/reviews_with_sentiment.csv",
    index=False
)

print(
    "Saved successfully."
)