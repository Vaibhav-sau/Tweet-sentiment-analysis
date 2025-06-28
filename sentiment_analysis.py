import tweepy
import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Clean Tweet Function
def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+|\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    return " ".join(tokens)

# Training the model
def train_model():
    df = pd.read_csv("https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv")
    df = df[['label', 'tweet']]
    df['clean_tweet'] = df['tweet'].apply(clean_text)
    df['label'] = df['label'].map({0: 'Negative', 1: 'Positive'})

    X = df['clean_tweet']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Classification Report:\n", classification_report(y_test, model.predict(X_test_vec)))

    return model, vectorizer

# Fetch real-time tweets using Tweepy v2
def fetch_tweets(keyword, bearer_token, max_results=20):
    client = tweepy.Client(bearer_token=bearer_token)
    response = client.search_recent_tweets(query=keyword + " -is:retweet lang:en", max_results=max_results)
    tweets = [tweet.text for tweet in response.data] if response.data else []
    return tweets

# Analysis
def analyze_tweets(tweets, model, vectorizer):
    cleaned = [clean_text(tweet) for tweet in tweets]
    vectors = vectorizer.transform(cleaned)
    preds = model.predict(vectors)

    df_result = pd.DataFrame({
        "Tweet": tweets,
        "Sentiment": preds
    })

    print(df_result)
    sns.countplot(x='Sentiment', data=df_result)
    plt.title("Sentiment Analysis of Live Tweets")
    plt.show()

# Running the pipeline
if __name__ == "__main__":
    BEARER_TOKEN = "Your Bearer Token"  # Replace this with your actual bearer token
    keyword = "AI"  # Change this to any keyword you want to fetch the tweets on

    model, vectorizer = train_model()
    tweets = fetch_tweets(keyword, BEARER_TOKEN, max_results=30)
    analyze_tweets(tweets, model, vectorizer)
