# ==========================================
# E-COMMERCE REVIEW INTELLIGENCE SYSTEM
# ==========================================

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from analytics import (
    load_data,
    get_sentiment,
    top_negative_words,
    top_positive_words,
    add_review_length,
    top_products,
    top_categories,
    recommendation_score,
    generate_insights
)

# ==========================================
# LOAD ML MODEL
# ==========================================

ml_model = joblib.load(
    "models/sentiment_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="E-Commerce Review Intelligence",
    page_icon="🛒",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def get_data():

    df = load_data()

    df["sentiment"] = (
        df["reviews.text"]
        .apply(get_sentiment)
    )

    df = add_review_length(df)

    return df


df = get_data()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("☰ Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Review Prediction",
        "About"
    ]
)

# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title(
        "🛒 E-Commerce Product Review Intelligence System"
    )

    st.markdown(
        "Analyze customer reviews using NLP, Sentiment Analysis, and Business Intelligence."
    )

    # ======================================
    # KPI CARDS
    # ======================================

    total_reviews = len(df)

    positive = len(
        df[
            df["sentiment"] == "Positive"
        ]
    )

    negative = len(
        df[
            df["sentiment"] == "Negative"
        ]
    )

    neutral = len(
        df[
            df["sentiment"] == "Neutral"
        ]
    )

    avg_rating = round(
        df["reviews.rating"].mean(),
        2
    )

    recommendation = (
        recommendation_score(df)
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "📦 Reviews",
        f"{total_reviews:,}"
    )

    c2.metric(
        "😊 Positive",
        positive
    )

    c3.metric(
        "😡 Negative",
        negative
    )

    c4.metric(
        "⭐ Avg Rating",
        avg_rating
    )

    c5.metric(
        "👍 Recommendation",
        f"{recommendation}%"
    )

    st.divider()

    # ======================================
    # SENTIMENT + RATING CHARTS
    # ======================================

    col1, col2 = st.columns(2)

    with col1:

        sentiment_chart = px.pie(
            df,
            names="sentiment",
            title="Sentiment Distribution"
        )

        st.plotly_chart(
            sentiment_chart,
            use_container_width=True
        )

    with col2:

        rating_chart = px.histogram(
            df,
            x="reviews.rating",
            nbins=5,
            title="Rating Distribution"
        )

        st.plotly_chart(
            rating_chart,
            use_container_width=True
        )

    st.divider()

    # ======================================
    # TOP KEYWORDS
    # ======================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "😡 Top Complaint Keywords"
        )

        negative_df = pd.DataFrame(

            top_negative_words(df),

            columns=[
                "Keyword",
                "Count"
            ]
        )

        fig = px.bar(
            negative_df,
            x="Keyword",
            y="Count"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "😊 Top Praised Keywords"
        )

        positive_df = pd.DataFrame(

            top_positive_words(df),

            columns=[
                "Keyword",
                "Count"
            ]
        )

        fig2 = px.bar(
            positive_df,
            x="Keyword",
            y="Count"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.divider()

    # ======================================
    # REVIEW LENGTH ANALYSIS
    # ======================================

    st.subheader(
        "📝 Review Length Analysis"
    )

    length_chart = px.histogram(
        df,
        x="review_length",
        nbins=30,
        title="Review Length Distribution"
    )

    st.plotly_chart(
        length_chart,
        use_container_width=True
    )

    st.divider()

    # ======================================
    # TOP PRODUCTS
    # ======================================

    st.subheader(
        "🔥 Top Reviewed Products"
    )

    products_df = top_products(df)

    product_chart = px.bar(
        products_df,
        x="Product",
        y="Reviews",
        title="Top Products by Review Count"
    )

    st.plotly_chart(
        product_chart,
        use_container_width=True
    )

    st.divider()

    # ======================================
    # TOP CATEGORIES
    # ======================================

    st.subheader(
        "📂 Top Categories"
    )

    category_df = top_categories(df)

    category_chart = px.bar(
        category_df,
        x="Category",
        y="Reviews",
        title="Top Categories"
    )

    st.plotly_chart(
        category_chart,
        use_container_width=True
    )

    st.divider()

    # ======================================
    # SEARCH REVIEWS
    # ======================================

    st.subheader(
        "🔍 Search Reviews"
    )

    search = st.text_input(
        "Search keyword"
    )

    if search:

        filtered = df[
            df["reviews.text"]
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

        st.write(
            f"Found {len(filtered)} reviews"
        )

        st.dataframe(

            filtered[
                [
                    "name",
                    "reviews.rating",
                    "sentiment",
                    "reviews.text"
                ]
            ].head(50),

            use_container_width=True
        )

    st.divider()

    # ======================================
    # BUSINESS INSIGHTS
    # ======================================

    st.subheader(
        "💡 Business Insights"
    )

    insights = generate_insights(df)

    for insight in insights:

        st.info(
            insight
        )

# ==========================================
# REVIEW PREDICTION
# ==========================================

elif page == "Review Prediction":

    st.title(
        "🔍 Real-Time Sentiment Prediction"
    )

    st.markdown(
        """
Analyze customer reviews using:

 1️⃣ VADER Sentiment Analysis (Rule-Based)

 2️⃣ Machine Learning Model (Logistic Regression)

Compare both predictions side-by-side.
        """
    )

    review = st.text_area(
        "Enter Customer Review",
        height=180,
        placeholder="Example: Battery drains quickly and device overheats."
    )

    if st.button(
        "Analyze Review"
    ):

        if review.strip():

            # ==================================
            # VADER PREDICTION
            # ==================================

            vader_prediction = (
                get_sentiment(review)
            )

            # ==================================
            # ML PREDICTION
            # ==================================

            review_vector = (
                vectorizer.transform(
                    [review]
                )
            )

            ml_prediction = (
                ml_model.predict(
                    review_vector
                )[0]
            )

            st.divider()

            col1, col2 = st.columns(2)

            # ==================================
            # VADER RESULT
            # ==================================

            with col1:

                st.subheader(
                    " VADER Prediction"
                )

                if vader_prediction == "Positive":

                    st.success(
                        f" {vader_prediction}"
                    )

                elif vader_prediction == "Negative":

                    st.error(
                        f"😡 {vader_prediction}"
                    )

                else:

                    st.warning(
                        f"😐 {vader_prediction}"
                    )

            # ==================================
            # ML RESULT
            # ==================================

            with col2:

                st.subheader(
                    "ML Prediction"
                )

                if ml_prediction == "Positive":

                    st.success(
                        f"😊 {ml_prediction}"
                    )

                elif ml_prediction == "Negative":

                    st.error(
                        f"😡 {ml_prediction}"
                    )

                else:

                    st.warning(
                        f"😐 {ml_prediction}"
                    )

            st.divider()

            if vader_prediction == ml_prediction:

                st.success(
                    f"✅ Both models agree: {vader_prediction}"
                )

            else:

                st.info(
                    f"⚠️ Models disagree. VADER = {vader_prediction}, ML = {ml_prediction}"
                )

            st.subheader(
                "📝 Review Text"
            )

            st.write(review)
# ==========================================
# ABOUT PAGE
# ==========================================

else:

    st.title("ℹ️ About Project")

    st.markdown(
        """
# 🛒 E-Commerce Product Review Intelligence System

An AI-powered analytics solution that processes **Amazon product reviews** to extract
actionable business insights using **Natural Language Processing (NLP), Sentiment Analysis, and Data Visualization**.

---

## 📊 Dataset Overview

This project is built on a real-world **Amazon Product Reviews dataset** containing:

- Product names and categories  
- Customer ratings and review text  
- Review timestamps and metadata  
- Recommendation and user feedback signals  

It simulates large-scale e-commerce customer behavior and feedback patterns.

---

## 🎯 Objective

- Automatically analyze customer sentiment at scale  
- Identify product strengths and recurring pain points  
- Extract meaningful insights from unstructured review text  
- Support data-driven product and business decisions  

---

## 🧠 Tech Stack

- Python  
- Pandas  
- NLTK (VADER Sentiment Analysis)  
- Scikit-learn (TF-IDF + Logistic Regression)  
- Streamlit (Interactive Dashboard)  
- Plotly (Data Visualization)  

---

## 🚀 Key Features

- Real-time sentiment analysis (VADER + ML model comparison)  
- Interactive business intelligence dashboard  
- Top complaint and positive keyword extraction  
- Product and category performance analysis  
- Review search and filtering system  
- Recommendation score and engagement metrics  

---

## 📌 Business Impact

This system enables e-commerce businesses to transform raw customer feedback into
**data-driven business intelligence**.

It helps in:

- Improving product quality based on customer pain points  
- Identifying features that drive positive customer satisfaction  
- Monitoring large-scale review trends in real time  
- Reducing manual effort in analyzing thousands of reviews  
- Supporting faster and smarter product decisions  

The solution is directly applicable to **e-commerce analytics, customer experience teams, and product management dashboards**, making it suitable for real-world enterprise-level use cases.

---
        """
    )