import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit_shadcn_ui as ui

st.set_page_config(layout="wide")

plt.style.use("dark_background")
sns.set_style("darkgrid", {"axes.facecolor": "#181818", "grid.color": "#2a2a2a"})
plt.rcParams.update({
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "grid.color": "#2a2a2a",
    "figure.facecolor": "#181818"
})

# Load dataset
@st.cache_data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    # Drop unnecessary columns
    columns_to_drop = ["img_link", "product_link", "user_id", "user_name", "review_id", "review_title", "review_content"]
    df.drop(columns=columns_to_drop, errors="ignore", inplace=True)

    # Extract main category
    df["category_top"] = df["category"].astype(str).apply(lambda x: x.split('|')[0])

    # Clean and convert numeric columns
    for col in ["discounted_price", "actual_price", "rating_count"]:
        df[col] = df[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "discount_percentage" in df.columns:
        df["discount_percentage"] = df["discount_percentage"].astype(str).str.replace("%", "").astype(float)

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Handle missing values
    df["rating"].fillna(df["rating"].mean(), inplace=True)
    df["rating_count"].fillna(df["rating_count"].median(), inplace=True)
    df.dropna(inplace=True)

    # Profit margin column
    df["profit_margin"] = df["actual_price"] - df["discounted_price"]

    return df

# Load data
data = load_and_clean_data("./amazon.csv")

st.title("Amazon Product Data Visualization")

# Metrics Summary
col1, col2, col3, col4 = st.columns(4)

with col1:
    ui.metric_card("Total Products", data.shape[0], "Total number of products in the dataset")

with col2:
    ui.metric_card("Average Rating", round(data["rating"].mean(), 2), "Average rating of all products")

with col3:
    ui.metric_card("Top Category", data["category_top"].mode()[0], "Most common product category")

with col4:
    ui.metric_card("Average Rating Count", int(data["rating_count"].mean()), "Average rating count")

# Load trained model for feature importance
@st.cache_resource
def load_model():
    return joblib.load("final_demand_prediction_xgb.pkl")

model = load_model()

# Get feature importance
def plot_feature_importance():
    feature_names = ["Discount %", "Rating", "Rating Count", "Category", "Price Difference"]
    feature_importance = model.feature_importances_

    fig = px.bar(x=feature_names, y=feature_importance, labels={'x': "Feature", 'y': "Importance"},
                 title="Feature Importance in Demand Prediction", template="plotly_dark")
    st.plotly_chart(fig)

# Visualization Functions
def plot_rating_distribution(data):
    fig = px.histogram(data, x="rating", nbins=10, marginal="box", title="Distribution of Product Ratings", template="plotly_dark")
    st.plotly_chart(fig)

def plot_correlation_heatmap(data):
    corr_matrix = data[["discounted_price", "actual_price", "rating", "rating_count"]].corr()
    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap", template="plotly_dark")
    st.plotly_chart(fig)

def plot_category_distribution(data):
    fig = px.bar(data["category_top"].value_counts().head(10), x=data["category_top"].value_counts().head(10).index, 
                 y=data["category_top"].value_counts().head(10).values, title="Product Distribution by Top-Level Category", template="plotly_dark")
    st.plotly_chart(fig)

def plot_price_vs_discount(data):
    fig = px.scatter(data, x="actual_price", y="discount_percentage", color="category_top", size="rating_count", 
                     title="Price vs Discount Percentage", template="plotly_dark")
    fig.update_layout(margin=dict(l=100, r=100, b=100, t=100), height=485)
    st.plotly_chart(fig)

def plot_avg_profit_margin(data):
    avg_profit_margin = data.groupby("category_top")["profit_margin"].mean().reset_index()
    fig = px.bar(avg_profit_margin, x="category_top", y="profit_margin", color="profit_margin",
                 title="Average Profit Margin by Category",
                 labels={"category_top": "Category", "profit_margin": "Profit Margin (₹)"},
                 text="profit_margin",
                 template="plotly_dark")
    st.plotly_chart(fig)

# Layout for Visualizations
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    with st.container(border=True):
        st.subheader("Distribution of Product Ratings")
        plot_rating_distribution(data)

with col2:
    with st.container(border=True):
        st.subheader("Price vs. Discount Percentage")
        plot_price_vs_discount(data)

with col3:
    with st.container(border=True):
        st.subheader("Feature Importance in Demand Prediction")
        plot_feature_importance()

col1, col2 = st.columns([1, 1])

with col1:
    with st.container(border=True):
        st.subheader("Correlation Heatmap")
        plot_correlation_heatmap(data)

with col2:
    with st.container(border=True):
        st.subheader("Average Profit Margin by Category")
        plot_avg_profit_margin(data)

with st.container(border=True):
    st.subheader("Product Distribution by Top-Level Category")
    plot_category_distribution(data)
