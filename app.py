import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/reviews.csv")

# Function to classify sentiment
def get_sentiment(review):

    polarity = TextBlob(review).sentiment.polarity

    if polarity > 0:
        return "Positive"

    elif polarity < 0:
        return "Negative"

    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Review"].apply(get_sentiment)

# Display results
print("\nCustomer Feedback Analysis")
print(df)

# Count sentiments
sentiment_counts = df["Sentiment"].value_counts()

print("\nSentiment Summary")
print(sentiment_counts)

# Graph
plt.figure(figsize=(6,4))

plt.bar(
    sentiment_counts.index,
    sentiment_counts.values
)

plt.title("Customer Feedback Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.show()