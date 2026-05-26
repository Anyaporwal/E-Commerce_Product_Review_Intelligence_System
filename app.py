import streamlit as st
import joblib

# ===================
# LOAD MODEL
# ===================

model=joblib.load(
"models/sentiment_model.pkl"
)

vectorizer=joblib.load(
"models/tfidf_vectorizer.pkl"
)

# ===================
# PAGE
# ===================

st.title(
"Amazon Review Intelligence System"
)

st.write(
"Analyze customer review sentiment"
)

review=st.text_area(
"Enter review:"
)

if st.button(
"Analyze"
):

    if review!="":

        vector=(
        vectorizer.transform(
        [review]
        )
        )

        prediction=(
        model.predict(
        vector
        )
        )

        probability=(
        model.predict_proba(
        vector
        )
        )

        confidence=(
        probability.max()*100
        )

        st.success(
        f"Sentiment: {prediction[0]}"
        )

        st.write(
        f"Confidence: {confidence:.2f}%"
        )

    else:

        st.warning(
        "Please enter text"
        )