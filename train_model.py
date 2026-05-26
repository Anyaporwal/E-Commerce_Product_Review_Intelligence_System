# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Download stopwords only once
nltk.download('stopwords')


# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

# Read CSV file
df = pd.read_csv(
    "data/amazon_reviews.csv"
)

print("Dataset loaded successfully")

# Keep only required columns
df = df[
    ['reviews.text',
     'reviews.rating']
]

# Remove missing values
df = df.dropna()

print(
    "Total rows:",
    len(df)
)


# ==========================================
# CREATE SENTIMENT LABELS
# ==========================================

'''
Convert ratings into labels

4–5 => Positive
3   => Neutral
1–2 => Negative
'''

def get_sentiment(rating):

    if rating >= 4:
        return "Positive"

    elif rating == 3:
        return "Neutral"

    else:
        return "Negative"


df["sentiment"] = (
    df["reviews.rating"]
    .apply(get_sentiment)
)

print("Sentiment labels created")


# ==========================================
# CHECK CLASS DISTRIBUTION
# ==========================================

print("\nSentiment Distribution:")

print(
    df["sentiment"]
    .value_counts()
)

'''
This helps detect imbalance.

Often datasets contain:

Positive 25000+
Negative 3000
Neutral 1000

This causes models
to overpredict Positive
'''


# ==========================================
# TEXT PREPROCESSING
# ==========================================

stop_words = set(
    stopwords.words(
        'english'
    )
)

def clean_text(text):

    # convert text to lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(
        r"http\S+",
        "",
        text
    )

    # remove symbols/numbers
    text = re.sub(
        r"[^a-zA-Z]",
        " ",
        text
    )

    words = text.split()

    # remove stopwords
    words = [

        word

        for word in words

        if word not in stop_words
    ]

    return " ".join(
        words
    )


print(
    "Cleaning reviews..."
)

df["clean_review"] = (
    df["reviews.text"]
    .apply(clean_text)
)

print(
    "Cleaning completed"
)


# ==========================================
# CONVERT TEXT INTO NUMBERS
# ==========================================

'''
TF-IDF converts text
into numerical vectors

ngram_range=(1,2)

Includes:

single words:
battery

word pairs:
battery drains

This improves sentiment
understanding
'''

vectorizer = TfidfVectorizer(

    max_features=10000,

    ngram_range=(1,2)
)


X = vectorizer.fit_transform(
    df["clean_review"]
)

y = df["sentiment"]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

'''
stratify=y keeps
same class proportions
in train and test data
'''

X_train, X_test, y_train, y_test = (

    train_test_split(

        X,
        y,

        test_size=0.2,

        stratify=y,

        random_state=42

    )

)


# ==========================================
# MODEL TRAINING
# ==========================================

print(
    "Training model..."
)

'''
class_weight='balanced'

Gives higher importance
to minority classes
'''

model = LogisticRegression(

    max_iter=1000,

    class_weight='balanced'
)

model.fit(

    X_train,

    y_train
)

print(
    "Training complete"
)


# ==========================================
# MODEL EVALUATION
# ==========================================

pred = model.predict(
    X_test
)

print("\nClassification Report:\n")

print(

    classification_report(

        y_test,

        pred
    )

)


# ==========================================
# SAVE FILES
# ==========================================

joblib.dump(

    model,

    "models/sentiment_model.pkl"
)

joblib.dump(

    vectorizer,

    "models/tfidf_vectorizer.pkl"
)

print(
    "\nModel saved successfully"
)