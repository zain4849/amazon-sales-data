import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# from wordcloud import WordCloud
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

# Loading and clean the dataset
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
    
    # Handling missing values
    df["rating"].fillna(df["rating"].mean(), inplace=True)
    df["rating_count"].fillna(df["rating_count"].median(), inplace=True)
    df.dropna(inplace=True)
    
    # Create profit margin column
    df["profit_margin"] = df["actual_price"] - df["discounted_price"]
    
    return df

# Load data
data = load_and_clean_data("./amazon.csv")

# Streamlit Title
st.title("📊 Amazon Product Data Visualization")

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

# Visualizations
def plot_rating_distribution(data):
    fig = px.histogram(data, x="rating", nbins=10, marginal="box", title="Distribution of Product Ratings", template="plotly_dark")
    st.plotly_chart(fig)

def plot_correlation_heatmap(data):
    corr_matrix = data[["discounted_price", "actual_price", "rating", "rating_count"]].corr().round(1)
    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap", template="plotly_dark")
    st.plotly_chart(fig)

def plot_price_vs_discount(data):
    fig = px.scatter(data, x="actual_price", y="discount_percentage", color="category_top", size="rating_count", 
                     title="Price vs Discount Percentage", template="plotly_dark")
    fig.update_layout(margin=dict(l=100, r=100, b=100, t=100),height=485)
    st.plotly_chart(fig)

def plot_avg_profit_margin(data):
    avg_profit_margin = data.groupby("category_top")["profit_margin"].mean().round(0).astype(int).reset_index()
    fig = px.bar(
        avg_profit_margin, 
        x="category_top", 
        y="profit_margin", 
        color="profit_margin",
        title="Average Profit Margin by Category",
        labels={"category_top": "Category", "profit_margin": "Profit Margin (₹)"},
        text="profit_margin"
    )
    st.plotly_chart(fig)

def plot_category_distribution_donut(data):
    # Get top 10 product categories
    top_categories = data["category_top"].value_counts().head(10)

    # Create a Plotly Donut Chart
    fig = px.pie(
        names=top_categories.index,
        values=top_categories.values,
        title="Product Distribution by Category (Donut Chart)",
        labels={"values": "Number of Products", "names": "Category"},
        hole=0.4,  # Creates donut effect
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig)

# Layout for Visualizations
col1, col2 = st.columns([1,2])

with col1:
    with st.container(border=True):
        st.subheader("Distribution of Product Ratings")
        plot_rating_distribution(data)

with col2:
    with st.container(border=True):
        st.subheader("Price vs. Discount Percentage")
        plot_price_vs_discount(data)

col1, col2 = st.columns([1,1])

with col1:
    with st.container(border=True):
        st.subheader("Correlation Heatmap")
        plot_correlation_heatmap(data)

with col2:
    with st.container(border=True):
        st.subheader("Average Profit Margin by Category")
        plot_avg_profit_margin(data)

# **Replaced Bar Chart with Donut Chart**
with st.container(border=True):
    st.subheader("Product Distribution by Top-Level Category (Donut Chart)")
    plot_category_distribution_donut(data)
