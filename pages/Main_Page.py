import streamlit as st

# Title and Heading
st.title('📊 Amazon Sales Data Visualization & Analysis')
st.markdown("""
Welcome to the **Amazon Product Sales Data Analysis** app! 
Here, we dive deep into the Amazon products dataset, offering insights on ratings, reviews, pricing, and much more.

This app provides an interactive and visually appealing way to explore the data and understand trends like never before.
""")

# Add a brief description
st.subheader('📝 About the Dataset')
st.markdown("""
This dataset contains the details of **1000+ Amazon products**, including their reviews, ratings, prices, and more. Let's explore the data and uncover some interesting insights!
""")

# Features overview with emojis and better formatting
st.subheader('🔍 Key Features of the Dataset')
st.markdown("""
- **Product ID**: Unique identifier for each product
- **Product Name**: Name of the product
- **Category**: Product's category (e.g., Electronics, Clothing)
- **Discounted Price**: Price after discount
- **Actual Price**: Original price
- **Discount Percentage**: Discount offered on the product
- **Rating**: Average rating from customers
- **Rating Count**: Number of people who have rated the product
- **About Product**: Product description
- **User ID**: ID of the user who wrote the review
- **User Name**: Name of the reviewer
- **Review Title**: Short review summary
- **Review Content**: Detailed review
- **Image Link**: URL of the product image
- **Product Link**: Official product page link
""")

# Add a little teaser for visualizations
st.subheader('📈 What Can You Explore?')
st.markdown("""
- **Price Distribution**: How are product prices spread across categories?
- **Rating Insights**: Average ratings for products and their distribution
- **Discount Insights**: Which categories offer the best discounts?
- **Top Products**: Top-rated products and most reviewed ones
- **Sentiment Analysis**: Analyzing reviews for product sentiment

Feel free to explore the app's various features and discover key trends in Amazon's product data.
""")

# Call to action for the user to interact with the app
st.subheader('🚀 Get Started!')
st.markdown("""
Navigate through the sidebar to choose what you'd like to analyze. Whether it's **price vs. rating** or **discount analysis**, we’ve got insights for you!
""")

# Add some fun/engaging content
st.subheader('💡 Fun Fact:')
st.markdown("""
Did you know? The highest-rated product in this dataset has a perfect rating of **5 stars**! ⭐
""")
