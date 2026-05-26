# =====================================
# IMPORT LIBRARIES
# =====================================

import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# download stopwords once
nltk.download('stopwords')

# =====================================
# LOAD DATA
# =====================================

print("Loading dataset...")

df = pd.read_csv("data/amazon_reviews.csv")

print("Dataset loaded")

# Keep useful columns only
df = df[['reviews.text','reviews.rating']]

# Remove null values
df = df.dropna()

print("Rows:",len(df))

# =====================================
# CREATE SENTIMENT LABELS
# =====================================

def get_sentiment(rating):

    if rating >=4:
        return "Positive"

    elif rating==3:
        return "Neutral"

    else:
        return "Negative"


df["sentiment"]=(
df["reviews.rating"]
.apply(get_sentiment)
)

print("Sentiments created")

# =====================================
# TEXT PREPROCESSING
# =====================================

stop_words=set(
stopwords.words('english')
)

def clean_text(text):

    # convert lowercase
    text=text.lower()

    # remove URLs
    text=re.sub(
    r"http\S+",
    "",
    text
    )

    # keep letters only
    text=re.sub(
    "[^a-zA-Z]",
    " ",
    text
    )

    words=text.split()

    # remove stopwords
    words=[
    word for word in words
    if word not in stop_words
    ]

    return " ".join(words)


print("Cleaning reviews...")

df["clean_review"]=(
df["reviews.text"]
.apply(clean_text)
)

print("Done cleaning")

# =====================================
# TFIDF CONVERSION
# =====================================

vectorizer=TfidfVectorizer(
max_features=5000
)

X=vectorizer.fit_transform(
df["clean_review"]
)

y=df["sentiment"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train,X_test,y_train,y_test=(
train_test_split(
X,
y,
test_size=.2,
random_state=42
)
)

# =====================================
# TRAIN MODEL
# =====================================

print("Training model...")

model=LogisticRegression(
max_iter=1000
)

model.fit(
X_train,
y_train
)

print("Training complete")

# =====================================
# TEST MODEL
# =====================================

pred=model.predict(
X_test
)

print(
classification_report(
y_test,
pred
)
)

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(
model,
"models/sentiment_model.pkl"
)

joblib.dump(
vectorizer,
"models/tfidf_vectorizer.pkl"
)

print("Model saved")