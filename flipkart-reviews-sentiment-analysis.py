import urllib.request
from bs4 import BeautifulSoup
import time
import random
from pymongo import MongoClient
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
from spacy.lang.en.stop_words import STOP_WORDS

# MongoDB connection
def connect_to_mongodb():
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["flipkart_reviews"]
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        exit()

# Scrape reviews from Flipkart
def scrape_reviews(base_url, product_name, pages=10):
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (X11; Linux x86_64)",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
    ]

    def soup(url):
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        req = urllib.request.Request(url, headers=headers)
        thepage = urllib.request.urlopen(req)
        soupdata = BeautifulSoup(thepage, "html.parser")
        thepage.close()
        return soupdata

    reviews = []
    hyperlinks = []
    for page_num in range(1, pages + 1):
        print(f"Scraping page {page_num} for product: {product_name}...")
        full_url = base_url.format(page_num)
        soup1 = soup(full_url)

        # Extract hyperlinks
        for link in soup1.find_all('a', href=True):
            hyperlinks.append(link['href'])

        # Extract reviews
        for record in soup1.find_all("div", {"class": "DOjaWF gdgoEp col-9-12"}):
            for review_block in record.find_all("div", {"class": "cPHDOP col-12-12"}):
                review = {"product_name": product_name}
                rating_tag = review_block.find("div", {"class": "XQDdHH Ga3i8K"})
                review['rating'] = rating_tag.text if rating_tag else None
                title_tag = review_block.find("p", {"class": "z9E0IG"})
                review['title'] = title_tag.text if title_tag else None
                text_tag = review_block.find("div", {"class": "ZmyHeo"})
                review['text'] = text_tag.div.text.replace('READ MORE', '') if text_tag and text_tag.div else None
                name_tag = review_block.find("p", {"class": "_2NsDsF AwS1CA"})
                review['reviewer'] = name_tag.text if name_tag else None

                if any(value is not None for value in review.values()):
                    reviews.append(review)

        time.sleep(random.uniform(5, 10))

    return reviews, hyperlinks

# Store reviews in MongoDB
def store_reviews_in_mongodb(reviews, collection_name):
    db = connect_to_mongodb()
    collection = db[collection_name]
    collection.insert_many(reviews)
    print(f"Inserted {len(reviews)} reviews into MongoDB.")

# Preprocess reviews using spaCy
def preprocess_reviews():
    db = connect_to_mongodb()
    collection = db["raw_reviews"]
    processed_collection = db["preprocessed_reviews"]

    nlp = spacy.load("en_core_web_sm")
    reviews = collection.find()

    for review in reviews:
        text = review.get('text', '')
        if text:
            doc = nlp(text)
            review["sentences"] = [sent.text for sent in doc.sents]
            review["tokenized_words"] = [token.text for token in doc]
            processed_collection.insert_one(review)

    print("Preprocessed data has been stored in the 'preprocessed_reviews' collection.")

# Sentiment Analysis Function
def analyze_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return {"polarity": polarity, "sentiment": sentiment}

# Perform sentiment analysis and store results
def classify_sentiment():
    db = connect_to_mongodb()
    collection = db["preprocessed_reviews"]
    sentiment_collection = db["sentiment_analyzed_reviews"]

    reviews = collection.find()
    for review in reviews:
        text = review.get('text', '')
        if text:
            sentiment_result = analyze_sentiment(text)
            review["sentiment_polarity"] = sentiment_result["polarity"]
            review["sentiment_category"] = sentiment_result["sentiment"]
            sentiment_collection.insert_one(review)

    print("Sentiment analysis results have been stored in the 'sentiment_analyzed_reviews' collection.")

# Generate visualizations for sentiment distribution
def generate_summary_report():
    db = connect_to_mongodb()
    collection = db["sentiment_analyzed_reviews"]
    reviews = list(collection.find())
    df = pd.DataFrame(reviews)

    # Group by product and calculate sentiment statistics
    summary = df.groupby("product_name").agg(
        total=pd.NamedAgg(column="sentiment_category", aggfunc="count"),
        positive=pd.NamedAgg(column="sentiment_category", aggfunc=lambda x: (x == "Positive").sum()),
        negative=pd.NamedAgg(column="sentiment_category", aggfunc=lambda x: (x == "Negative").sum()),
        neutral=pd.NamedAgg(column="sentiment_category", aggfunc=lambda x: (x == "Neutral").sum()),
        polarity=pd.NamedAgg(column="sentiment_polarity", aggfunc="mean")
    ).reset_index()

    print("\n--- Product Sentiment Summary ---")
    print(summary)

    # Visualize sentiment distribution for each product
    for product in summary["product_name"]:
        product_data = df[df["product_name"] == product]
        sentiment_counts = product_data["sentiment_category"].value_counts()

        # Pie Chart
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=["#66b3ff", "#ff9999", "#99ff99"]
        )
        plt.title(f"Sentiment Distribution for {product}")

        # Bar Plot
        plt.subplot(1, 2, 2)
        sns.countplot(
            x="sentiment_category",
            data=product_data,
            order=["Positive", "Neutral", "Negative"],
            palette="viridis"
        )
        plt.title(f"Sentiment Count for {product}")
        plt.xlabel("Sentiment Category")
        plt.ylabel("Count")
        plt.tight_layout()

        # Save the plot for each product
        plt.savefig(f"{product}_sentiment_distribution.png")
        plt.show()

    print("Visualizations saved for each product's sentiment distribution.")

# Visualize most frequent words in positive and negative reviews
def visualize_frequent_words():
    db = connect_to_mongodb()
    collection = db["sentiment_analyzed_reviews"]
    reviews = list(collection.find())
    df = pd.DataFrame(reviews)

    # Separate positive and negative reviews
    positive_reviews = df[df["sentiment_category"] == "Positive"]["text"].dropna()
    negative_reviews = df[df["sentiment_category"] == "Negative"]["text"].dropna()

    # Function to extract most common words
    def get_most_common_words(texts, num_words=20):
        all_words = " ".join(texts).lower().split()
        filtered_words = [word for word in all_words if word.isalpha() and word not in STOP_WORDS]
        word_counts = Counter(filtered_words)
        return word_counts.most_common(num_words)

    # Get most common words for positive and negative reviews
    positive_words = get_most_common_words(positive_reviews)
    negative_words = get_most_common_words(negative_reviews)

    # Create word clouds
    def create_wordcloud(word_counts, title):
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(word_counts))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title, fontsize=16)
        plt.show()

    # Create bar plots
    def create_barplot(word_counts, title):
        words, counts = zip(*word_counts)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(counts), y=list(words), palette="viridis")
        plt.title(title, fontsize=16)
        plt.xlabel("Frequency")
        plt.ylabel("Words")
        plt.tight_layout()
        plt.show()

    # Visualize positive words
    print("\nMost Frequent Words in Positive Reviews:")
    create_wordcloud(positive_words, "Word Cloud for Positive Reviews")
    create_barplot(positive_words, "Most Frequent Words in Positive Reviews")

    # Visualize negative words
    print("\nMost Frequent Words in Negative Reviews:")
    create_wordcloud(negative_words, "Word Cloud for Negative Reviews")
    create_barplot(negative_words, "Most Frequent Words in Negative Reviews")

# Main script
if __name__ == "__main__":
    # List of product review page URLs and product names
    products = [
        {"name": "iPhone 16", "url": "https://www.flipkart.com/apple-iphone-16-white-128-gb/product-reviews/itm7c0281cd247be?pid=MOBH4DQF849HCG6G&page={}"},
        {"name": "Samsung Galaxy S24", "url": "https://www.flipkart.com/samsung-galaxy-s24-5g-onyx-black-256-gb/product-reviews/itm325da4a26d7bb?pid=MOBGX2F3HVJYNHUV&lid=LSTMOBGX2F3HVJYNHUVPMFBMF&page={}"},
        {"name": "Google Pixel 8", "url": "https://www.flipkart.com/google-pixel-8-rose-128-gb/product-reviews/itm67e2a2531aaac?pid=MOBGT5F22JFCABET&lid=LSTMOBGT5F22JFCABETVKHMHM&page={}"},
        {"name": "Motorola Edge 30 Pro", "url": "https://www.flipkart.com/motorola-edge-30-pro-stardust-white-128-gb/product-reviews/itmc60ad815a2cbd?pid=MOBG9CKYEEUCNHUM&lid=LSTMOBG9CKYEEUCNHUMJDG8TW&page={}"},
        {"name": "Vivo V50 5G", "url": "https://www.flipkart.com/vivo-v50-5g-starry-night-256-gb/product-reviews/itm12bbdca230795?pid=MOBH8Z32NHHVHUZZ&lid=LSTMOBH8Z32NHHVHUZZ6RA0CH&page={}"},
        {"name": "OPPO Reno13 5G", "url": "https://www.flipkart.com/oppo-reno13-5g-luminous-blue-256-gb/product-reviews/itmc0ea16ae55e86?pid=MOBH7JWEXKPXQDCA&lid=LSTMOBH7JWEXKPXQDCAMNRGHC&page={}"},
        {"name": "LG G8X", "url": "https://www.flipkart.com/lg-g8x-black-128-gb/product-reviews/itme8a4f5f473aa4?pid=MOBFZKQWFRFMHKQK&lid=LSTMOBFZKQWFRFMHKQKVOP4L4&page={}"},
        {"name": "REDMI Note 13 Pro+ 5G", "url": "https://www.flipkart.com/redmi-note-13-pro-5g-fusion-white-256-gb/product-reviews/itma571bf09ef6e4?pid=MOBGZF9PRDCEYTFM&lid=LSTMOBGZF9PRDCEYTFMYEFSMB&page={}"}
    ]

    # Phase 1: Scrape and store reviews for all products
    all_reviews = []
    for product in products:
        reviews, hyperlinks = scrape_reviews(product["url"], product["name"])
        all_reviews.extend(reviews)

    store_reviews_in_mongodb(all_reviews, "raw_reviews")

    '''
    # Display hyperlinks
    print("Hyperlinks retrieved during scraping:")
    for link in hyperlinks:
        print(link)
    '''

    # Phase 2: Preprocess reviews
    preprocess_reviews()

    # Phase 3: Perform sentiment analysis
    classify_sentiment()

    # Generate summary report
    generate_summary_report()

    # Visualize most frequent words
    visualize_frequent_words()

