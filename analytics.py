# ==========================================
# ANALYTICS MODULE
# E-Commerce Review Intelligence System
# ==========================================

import pandas as pd
import re
import nltk

from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# ==========================================
# DOWNLOAD NLTK DATA
# ==========================================

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("vader_lexicon")

# ==========================================
# GLOBAL OBJECTS
# ==========================================

lemmatizer = WordNetLemmatizer()

stop_words = set(
    stopwords.words("english")
)

custom_stopwords = {
    "amazon",
    "product",
    "item",
    "purchase",
    "buy",
    "bought",
    "one",
    "would",
    "get",
    "got",
    "use",
    "using",
    "used",
    "like",
    "also",
    "really",
    "much",
    "even",
    "still",
    "could",
    "well",
    "good",
    "great",
    "nice",
    "best"
}

stop_words.update(custom_stopwords)

sia = SentimentIntensityAnalyzer()

# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    df = pd.read_csv(
        "data/amazon_reviews.csv",
        low_memory=False
    )

    columns = [
        "name",
        "categories",
        "reviews.rating",
        "reviews.text",
        "reviews.title",
        "reviews.date"
    ]

    df = df[columns]

    df = df.dropna()

    return df


# ==========================================
# CLEAN TEXT
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z]",
        " ",
        text
    )

    words = text.split()

    cleaned = []

    for word in words:

        if (
            word not in stop_words
            and len(word) > 2
        ):

            word = (
                lemmatizer
                .lemmatize(word)
            )

            cleaned.append(word)

    return cleaned


# ==========================================
# SENTIMENT ANALYSIS
# ==========================================

def get_sentiment(text):

    score = sia.polarity_scores(
        str(text)
    )["compound"]

    if score >= 0.05:
        return "Positive"

    elif score <= -0.05:
        return "Negative"

    else:
        return "Neutral"


# ==========================================
# TOP NEGATIVE WORDS
# ==========================================

def top_negative_words(
    df,
    top_n=15
):

    negative_reviews = df[
        df["sentiment"] == "Negative"
    ]

    all_words = []

    for review in negative_reviews[
        "reviews.text"
    ]:

        all_words.extend(
            clean_text(review)
        )

    return Counter(
        all_words
    ).most_common(top_n)


# ==========================================
# TOP POSITIVE WORDS
# ==========================================

def top_positive_words(
    df,
    top_n=15
):

    positive_reviews = df[
        df["sentiment"] == "Positive"
    ]

    all_words = []

    for review in positive_reviews[
        "reviews.text"
    ]:

        all_words.extend(
            clean_text(review)
        )

    return Counter(
        all_words
    ).most_common(top_n)


# ==========================================
# REVIEW LENGTH
# ==========================================

def add_review_length(df):

    df["review_length"] = (

        df["reviews.text"]

        .astype(str)

        .apply(len)
    )

    return df


# ==========================================
# TOP PRODUCTS
# ==========================================

def top_products(df):

    products = (
        df["name"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    products.columns = [
        "Product",
        "Reviews"
    ]

    return products


# ==========================================
# TOP CATEGORIES
# ==========================================

def top_categories(df):

    categories = (
        df["categories"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    categories.columns = [
        "Category",
        "Reviews"
    ]

    return categories


# ==========================================
# RECOMMENDATION SCORE
# ==========================================

def recommendation_score(df):

    total = len(df)

    positive = len(
        df[
            df["sentiment"]
            == "Positive"
        ]
    )

    return round(
        (positive / total) * 100,
        2
    )


# ==========================================
# BUSINESS INSIGHTS
# ==========================================

def generate_insights(df):

    total = len(df)

    positive = len(
        df[
            df["sentiment"]
            == "Positive"
        ]
    )

    negative = len(
        df[
            df["sentiment"]
            == "Negative"
        ]
    )

    neutral = len(
        df[
            df["sentiment"]
            == "Neutral"
        ]
    )

    avg_rating = round(
        df["reviews.rating"].mean(),
        2
    )

    recommendation = recommendation_score(df)

    complaint = top_negative_words(
        df,
        1
    )

    praise = top_positive_words(
        df,
        1
    )

    complaint_word = (
        complaint[0][0]
        if complaint
        else "N/A"
    )

    praise_word = (
        praise[0][0]
        if praise
        else "N/A"
    )

    insights = [

        f"📌 {recommendation}% of reviews are positive.",

        f"📌 Average rating is {avg_rating}.",

        f"📌 Negative reviews count: {negative}.",

        f"📌 Neutral reviews count: {neutral}.",

        f"📌 Total reviews analyzed: {total}."
    ]

    return insights