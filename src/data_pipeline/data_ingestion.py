import pandas as pd
import requests
from bs4 import BeautifulSoup
import tweepy
import praw
from textblob import TextBlob
from google.cloud import storage


def download_mlb_data(bucket_name, local_path):
    """
    Downloads MLb dataset from google cloud storage

    Args:
        bucket_name: Pointer to google cloud storage bucket
        local_path: Pointer to storage of downloaded files
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs()
    for blob in blobs:
        file_path = os.path.join(local_path, blob.name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob.download_to_filename(file_path)


def download_news_articles(url, local_path):
    """
    Downloads news function thats from given URL

    Args:
        url: URL of the news article
        local_path: Local path to store downloaded articles
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parse")
        articles = soup.find_all("article") # Adjust based on website's structure

        os.makedirs(local_path, exist_ok=True)
        for i, article in enumarate(articles):
            title = article.find("h2").text.strip() if article.find("h2") else f"article_{i+1}"
            content = article.text.strip()
            with open("{local_path}/{title.replace(' ', '_')}.txt", "w", encoding="utf-8") as f:
                f.write(content)
        else:
            print(f"Failed to frtch news articles.HTTP Status: {response.status_code}")


def analyze_sentiment(text):
    """
    Fucntion to analyze sentiments in a given text

    Args:
        text: Text to analyze
    Returns:
        A dictionary with polarity and subjective scores
    """
    analysis = TextBlob(text)
    return {"polarity": analysis.sentiment.polarity, "subjectivity": analysis.sentiment.subjectivity}


def download_social_media_data(api_credentials, local_path):
    """
    Function to download social media data ie Twitter and Reddit using API credentials

    Args:
        api_credentials: Dictionary containing API credentials for social media platforms.
        local_path: Local path to store downloaded social media data.
    """
    # Twitter
    twitter_auth = tweepy.OAuth1UserHandler(
        api_credentials["twitter_api_key"],
        api_credentials["twitter_api_secret"],
        api_credentials["twitter_access_token"],
        api_credentials["twitter_access_token_secret"]
    )
    twitter_api = tweepy.API(twitter_auth)
    tweets = twitter_Api.search_tweets(q="#baseball", count=50)

    os.makedirs(local_path, exist_ok=True)
    with open(f"{local_path}/twitter_data.txt", "w", encoding="utf-8") as f:
        for tweet in tweets:
            sentiment = analyze_sentiment(tweet.text)
            f.write(f"Tweet: {tweet.text}\nSentiment: {sentiment}\n\n")

    # Reddit
    reddit = praw.Reddit(
        client_id=api_credentials["reddit_client_id"],
        client_secret=api_credentials["reddit_client_secret"],
        user_agent=api_credentials["reddit_user_agent"]
    )
    subreddit = reddit.subreddit("baseball")
    reddit_posts = subreddit.hot(limit=50)

    with open(f"{local_path}/reddit_data.txt", "w", encoding="utf-8") as f:
        for posts in reddit_posts:
            content = f"title: {post.title}\nContent: {post.selftext}"
            sentiment = analyze_sentiment(content)
            f.write(f"{content}\nSentiment: {sentiment}\n\n")


if __name__ == "__main__":
    bucket_name = "your-gcs-bucket-name"  # Replace with your actual bucket name
    local_path = "data/raw"

    news_url = "https://example.com/baseball-news"  # Replace with actual news source URL
    api_credentials = {
        "twitter_api_key": "your-twitter-api-key",
        "twitter_api_secret": "your-twitter-api-secret",
        "twitter_access_token": "your-twitter-access-token",
        "twitter_access_token_secret": "your-twitter-access-token-secret",
        "reddit_client_id": "your-reddit-client-id",
        "reddit_client_secret": "your-reddit-client-secret",
        "reddit_user_agent": "your-reddit-user-agent"
    }

    download_mlb_data(bucket_name, local_path)
    download_news_articles(news_url, f"{local_path}/news_articles")
    download_social_media_data(api_credentials, f"{local_path}/social_medial")
