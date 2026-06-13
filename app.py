# ==========================================
# E-COMMERCE REVIEW INTELLIGENCE SYSTEM
# ==========================================

import streamlit as st
import pandas as pd
import plotly.express as px

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

st.sidebar.title("🧭 Navigation")

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

    st.write(
        "Enter a customer review below."
    )

    review = st.text_area(
        "Review",
        height=150
    )

    if st.button(
        "Analyze Review"
    ):

        if review.strip():

            prediction = (
                get_sentiment(review)
            )

            if prediction == "Positive":

                st.success(
                    f"😊 Sentiment: {prediction}"
                )

            elif prediction == "Negative":

                st.error(
                    f"😡 Sentiment: {prediction}"
                )

            else:

                st.warning(
                    f"😐 Sentiment: {prediction}"
                )

# ==========================================
# ABOUT PAGE
# ==========================================

else:

    st.title(
        "ℹ️ About Project"
    )

    st.markdown(
        """
## E-Commerce Product Review Intelligence System

### Technologies Used

- Python
- Pandas
- NLP
- NLTK
- VADER Sentiment Analysis
- Plotly
- Streamlit

### Features

- Real-Time Sentiment Prediction
- Customer Complaint Analysis
- Top Praised Features
- Product Analytics
- Category Analytics
- Review Search
- Recommendation Score
- Business Insights Dashboard

### Business Value

Helps organizations analyze customer feedback,
identify recurring issues,
and improve products using data-driven insights.
        """
    )