import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langdetect import detect
from googletrans import Translator
import re
from datetime import datetime
import logging
import streamlit as st
import hashlib
import base64
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Product Review Scraper", page_icon=":sparkles:")

# Adding a brief description
st.sidebar.title("Product Review Scraper")
st.sidebar.markdown("This app allows you to scrape and analyze Amazon product reviews.")

# Adding sidebar navigation
nav_options = ["Scrape Reviews", "Data Display", "Word Cloud", "Rating Distribution", "Sentiment Distribution", "Download"]
selected_nav = st.sidebar.radio("Navigation", nav_options)

# Adding tooltips
st.sidebar.info("Enter the Amazon product URL and click 'Scrape Reviews' to fetch the reviews.")
st.sidebar.info("Navigate through the different sections to explore the analysis results.")

# Adding custom CSS styles
st.markdown(
    """
    <style>
    .highlight {
        color: #ff4b4b;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the BERT tokenizer and model at the global level
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model = model.to('cpu')  # Use CPU for inference


def clean_text(text):
    try:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        return text.strip()
    except Exception as e:
        logging.error("Error cleaning text: %s", str(e))
        return None


def scrape_reviews(url, max_reviews=1000):
    reviews_data = []
    page = 1
    try:
        while len(reviews_data) < max_reviews:
            response = requests.get(url + f"&pageNumber={page}")
            response.raise_for_status()  # Raises a HTTPError for bad responses
            soup = BeautifulSoup(response.content, "html.parser")
            reviews = soup.find_all("div", {"data-hook": "review"})
            if not reviews:
                break

            for review in reviews:
                if len(reviews_data) >= max_reviews:
                    break

                text_element = review.find("span", {"data-hook": "review-body"})
                text = ' '.join(text_element.stripped_strings) if text_element else None

                rating_element = review.find("i", {"data-hook": "review-star-rating"})
                rating = float(rating_element.text.split()[0]) if rating_element else None

                date_element = review.find("span", {"data-hook": "review-date"})
                review_date = date_element.text.strip() if date_element else None

                reviewer_name_element = review.find("span", {"class": "a-profile-name"})
                reviewer_name = reviewer_name_element.text.strip() if reviewer_name_element else None

                verified_purchase = bool(review.find("span", {"data-hook": "avp-badge"}))

                if None not in [text, rating, review_date, reviewer_name]:
                    reviews_data.append({
                        "text": text,
                        "rating": rating,
                        "date": review_date,
                        "reviewer_name": reviewer_name,
                        "verified_purchase": verified_purchase
                    })

            page += 1
    except requests.RequestException as e:
        logging.error("Failed to retrieve or parse page: %s", str(e))

    return reviews_data

def process_reviews(reviews_df):
    processed_reviews = []
    try:
        for _, row in reviews_df.iterrows():
            place, date, year = None, None, None
            if 'Reviewed in' in row['date']:
                parts = row['date'].replace('Reviewed in ', '').split(' on ')
                if len(parts) == 2:
                    place = parts[0].capitalize()
                    date = datetime.strptime(parts[1], '%B %d, %Y')
                    year = date.year

            processed_reviews.append({
                "text": row['text'],
                "rating": row['rating'],
                "place": place,
                "date": date,
                "year": year,
                "reviewer_name": row['reviewer_name'],
                "verified_purchase": row['verified_purchase'],
                "language": detect(row['text']) if row['text'] else 'en',
                "translated_text": row['text']
            })
    except Exception as e:
        logging.error("Error processing reviews: %s", str(e))
    return pd.DataFrame(processed_reviews)


@st.cache_data
def collect_reviews(url, max_reviews=1000):
    try:
        reviews_data = scrape_reviews(url, max_reviews)
        print(f"Number of reviews scraped: {len(reviews_data)}")
        reviews_df = pd.DataFrame(reviews_data)
        processed_reviews = process_reviews(reviews_df)
        print(f"Number of processed reviews: {len(processed_reviews)}")
        return processed_reviews
    except Exception as e:
        logging.error("Failed in collecting or processing reviews: %s", str(e))
        return pd.DataFrame()



def preprocess_text(text):
    """
    Preprocesses the given text by removing URLs and replacing contractions.
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Replace contractions
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        # ... (contractions dictionary)
    }

    for contraction, replacement in contractions.items():
        text = text.replace(contraction, replacement)

    return text


def get_sentiment_bert(row, model, tokenizer):
    text = row['translated_text']

    # Ensure consistent tokenization length
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits

    # Apply softmax to get probabilities for each class
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()

    return probabilities


def main():
    # Add Amazon logo
    amazon_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Amazon_logo.svg/1200px-Amazon_logo.svg.png"
    st.image(amazon_logo, width=200)

    st.title("Product Review Scraper")

    # Input form for Amazon product URL
    url = st.text_input("Enter the Amazon product URL:")

    if st.button("Scrape Reviews"):
        if url:
            try:
                # Display progress bar while scraping
                progress_bar = st.progress(0)
                progress_text = st.empty()

                # Scrape and process reviews
                processed_df = collect_reviews(url)

                if not processed_df.empty:
                    # Update progress bar to 100% and clear progress text
                    progress_bar.progress(100)
                    progress_text.empty()

                    # Show success message
                    st.success(f"Successfully scraped {len(processed_df)} reviews!")

                    # Redirect to Data Display page
                    st.session_state.processed_df = processed_df
                    st.session_state.current_page = "data_display"
                    st.experimental_rerun()
                else:
                    st.warning("No reviews found for the given product.")
            except requests.RequestException:
                st.error("Invalid URL or unable to reach the website.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a valid Amazon product URL.")

    if "current_page" in st.session_state:
        if st.session_state.current_page == "data_display":
            show_data_display(st.session_state.processed_df)
        elif st.session_state.current_page == "word_cloud":
            show_word_cloud(st.session_state.processed_df)
        elif st.session_state.current_page == "rating_distribution":
            show_rating_distribution(st.session_state.processed_df)
        elif st.session_state.current_page == "sentiment_distribution":
            show_sentiment_distribution(st.session_state.processed_df)
        elif st.session_state.current_page == "download":
            show_download_options(st.session_state.processed_df)


def show_data_display(df):
    st.subheader("Scraped Review Data")
    st.dataframe(df)

    # Button to move to word cloud page
    if st.button("Next"):
        st.session_state.current_page = "word_cloud"
        st.experimental_rerun()


def show_word_cloud(df):
    st.subheader("Word Cloud")
    text = " ".join(df["text"])
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.imshow(wordcloud, interpolation="bilinear")
    ax1.axis("off")
    st.pyplot(fig1)

    plt.savefig('/Users/snehaagrawal/Documents/SEM 2/Web Mining/word_cloud.png')

    # Button to move to rating distribution page
    if st.button("Next"):
        st.session_state.current_page = "rating_distribution"
        st.experimental_rerun()


def show_rating_distribution(df):
    st.subheader("Distribution of Review Ratings")
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    rating_counts = df['rating'].value_counts().sort_index()
    total_reviews = len(df)
    labels = [f"{rating} ({count} / {count/total_reviews:.2%})" for rating, count in rating_counts.items()]
    ax2.pie(rating_counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.set_aspect('equal')
    ax2.set_title('Rating Distribution')
    st.pyplot(fig2)

    plt.savefig('/Users/snehaagrawal/Documents/SEM 2/Web Mining/rating_distribution.png')

    # Button to move to sentiment distribution page
    if st.button("Next"):
        st.session_state.current_page = "sentiment_distribution"
        st.experimental_rerun()


def show_sentiment_distribution(df):
    # Preprocess the data
    df = df.copy()  # A copy to avoid modifying the original DataFrame
    df.columns = df.columns.str.lower()
    df = df.drop(columns=['verified_purchase'])
    df['date_reviewed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.drop(columns=['date'])

    df['translated_text'] = df['text'].apply(preprocess_text)
    df['sentiment_probs'] = df.apply(lambda row: get_sentiment_bert(row, model, tokenizer), axis=1)

    # Extracting individual sentiment probabilities
    df['negative_prob'] = df['sentiment_probs'].apply(lambda x: x[0])
    df['neutral_prob'] = df['sentiment_probs'].apply(lambda x: x[1])
    df['positive_prob'] = df['sentiment_probs'].apply(lambda x: x[2])

    print("Rating value counts:")
    print(df['rating'].value_counts())

    # BERT probability value counts
    print("BERT probability value counts:")
    print(df['negative_prob'].value_counts())
    print(df['neutral_prob'].value_counts())
    print(df['positive_prob'].value_counts())

    # Assign sentiment based on both BERT probabilities and rating
    def assign_sentiment(row):
        rating = row['rating']
        positive_prob = row['positive_prob']
        negative_prob = row['negative_prob']

        # Assign sentiment based on rating
        if rating >= 4.5:
            return 'Positive'
        elif rating <= 3:
            return 'Negative'
        else:  # this will cover the case of rating == 3.0 too
            if positive_prob > 0.5:
                return 'Positive'
            elif negative_prob > 0.5:
                return 'Negative'
            else:
                return 'Neutral'

    df['sentiment'] = df.apply(assign_sentiment, axis=1)

    print("Sentiment value counts:")
    print(df['sentiment'].value_counts())

    # Class distribution
    class_counts = df['sentiment'].value_counts()

    # If any class has zero instances, assign a minimum number of instances to that class
    min_instances = 10  # Minimum number of instances for each class
    for cls in ['Negative', 'Neutral', 'Positive']:
        if cls not in class_counts.index:
            df.loc[:min_instances, 'sentiment'] = cls

    # Under-sampling and Over-sampling using SMOTE
    over = SMOTE(sampling_strategy='auto', random_state=42)
    under = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    X = df[['negative_prob', 'neutral_prob', 'positive_prob']]
    y = df['sentiment']

    # Check if resampling is needed
    unique_classes = y.nunique()
    if unique_classes >= 2:
        X_resampled, y_resampled = pipeline.fit_resample(X, y)

        # Creating resampled DataFrame with 'translated_text' and 'sentiment' columns
        df_resampled = pd.DataFrame({
            'translated_text': df['translated_text'],
            'sentiment': y_resampled
        })
    else:
        st.warning("Not enough classes for resampling. Need at least 2 distinct classes.")
        df_resampled = df[['translated_text', 'sentiment']]

    # Store the resampled DataFrame in the session state
    st.session_state.df_resampled = df_resampled

    print("Unique sentiment values:", df_resampled['sentiment'].unique())

    # Sentiment Distribution Before and After Oversampling
    st.subheader("Sentiment Distribution Before Oversampling")
    sentiment_counts_original = df['sentiment'].value_counts().reindex(['Negative', 'Neutral', 'Positive'],
                                                                       fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bars = sentiment_counts_original.plot(kind='bar', color=['red', 'yellow', 'green'], ax=ax3)
    ax3.set_xlabel('Sentiment')
    ax3.set_ylabel('Count')
    ax3.set_xticklabels(['Negative', 'Neutral', 'Positive'], rotation=0)
    for i, bar in enumerate(bars.patches):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height()}",
                 ha='center', va='bottom')

    st.pyplot(fig3)

    plt.savefig('/Users/snehaagrawal/Documents/SEM 2/Web Mining/sentiment_distribution_before_oversampling.png')

    st.subheader("Sentiment Distribution After Oversampling")
    sentiment_counts_resampled = df_resampled['sentiment'].value_counts().reindex(['Negative', 'Neutral', 'Positive'],
                                                                                  fill_value=0)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    bars = sentiment_counts_resampled.plot(kind='bar', color=['red', 'yellow', 'green'], ax=ax4)
    ax4.set_xlabel('Sentiment')
    ax4.set_ylabel('Count')
    ax4.set_xticklabels(['Negative', 'Neutral', 'Positive'], rotation=0)
    for i, bar in enumerate(bars.patches):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height()}",
                 ha='center', va='bottom')

    st.pyplot(fig4)

    plt.savefig('/Users/snehaagrawal/Documents/SEM 2/Web Mining/sentiment_distribution_after_oversampling.png')

    # Button to move to download page
    if st.button("Next"):
        st.session_state.current_page = "download"
        st.experimental_rerun()
        st.session_state.df_resampled = df_resampled


def show_download_options(df):
    st.subheader("Download Tables")

    # Download Original Table
    if st.button("Download Original Table"):
        csv_data = df.to_csv(index=False)
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="original_table.csv">Download Original Table</a>'
        st.markdown(href, unsafe_allow_html=True)

    # Download Resampled Table
    if 'df_resampled' in st.session_state:
        if st.button("Download Resampled Table"):
            df_resampled = st.session_state.df_resampled

            # Map sentiment labels to numeric values
            sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
            df_resampled['sentiment'] = df_resampled['sentiment'].map(sentiment_mapping)

            # Select only the 'translated_text' and 'sentiment' columns
            df_resampled = df_resampled[['translated_text', 'sentiment']]

            csv_data = df_resampled.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="resampled_table.csv">Download Resampled Table</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.error("Resampled data is not available for download.")


if __name__ == "__main__":
    main()

