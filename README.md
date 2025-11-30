# Week-2
🌟 Customer Experience Analytics for Fintech Apps
A Comparative Study of User Reviews from CBE, Bank of Abyssinia, and Dashen Bank
✨ Overview

This project explores the digital heartbeat of three major Ethiopian fintech apps — Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank.
Every review left by a customer carries an emotion, a frustration, a moment of joy, or a quiet request. By studying these voices, this project aims to understand how users feel when they interact with these apps and what their experiences reveal about the quality of service, the design, and the reliability of digital banking in Ethiopia.

The analysis uses natural language processing techniques to uncover sentiment, extract hidden themes, and compare how customers perceive each bank’s mobile app. Through sentiment scoring and topic modeling, the project paints a clear picture of user experience across the three platforms.

🎯 Project Purpose

At its core, this project seeks to transform unstructured user reviews into meaningful insights.
It captures the emotional tone of each comment using VADER sentiment analysis, then dives deeper with LDA topic modeling to reveal the major themes that define customer experience — from login issues to transfer delays, from UI design to customer service responsiveness.

By studying and comparing these patterns, the project provides a foundation for improving fintech applications in ways that feel intuitive, reliable, and human-centered.

🛠️ Data Processing & Preparation

The raw reviews are cleaned through tokenization, stopword removal, lowercasing, and lemmatization to create a polished, standardized text structure. These processed tokens serve as the foundation for both sentiment analysis and topic modeling.

Below is a simplified example of the preprocessing pipeline:

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

df["tokens"] = df["review"].apply(preprocess)


This prepares the text for deeper analysis.

💬 Sentiment Analysis (VADER)

To understand the emotions hidden within reviews, VADER is used.
It returns four values — positive, negative, neutral, and compound — which reflect how users emotionally respond to the app experience.

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

df["sentiment"] = df["review"].apply(lambda x: sia.polarity_scores(x))
df["compound"] = df["sentiment"].apply(lambda x: x["compound"])


A compound score closer to 1.0 means highly positive feedback, while scores near -1.0 reflect strong dissatisfaction.

🔍 Topic Modeling with LDA (scikit-learn)

Since Gensim was unavailable, the project uses scikit-learn’s LDA model.
Reviews are first converted into a document–term matrix, allowing the algorithm to extract dominant themes.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
dtm = vectorizer.fit_transform(df["tokens"])

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(dtm)

words = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx+1}:")
    print([words[i] for i in topic.argsort()[-10:]])


This reveals the top words that define each hidden topic, such as app crashes, update issues, smooth transactions, or login failures.

📊 Insights & Interpretation

With sentiment scores and extracted topics, the project compares the emotional and thematic patterns across the three banks.
For instance, one app may receive more negative comments about failed transactions, while another might be praised for ease of use but criticized for slow customer support.
This comparison highlights strengths to be preserved and weaknesses that need immediate attention.

The findings become a guide — lighting the way toward better user experience design, stronger app stability, and a more trustworthy digital banking ecosystem.

🌱 Why This Work Matters

Fintech apps are no longer optional; they are essential companions in everyday financial life. Each review represents someone trying to send money home, pay a bill, or access their savings.
By listening carefully to these voices, this project contributes to building digital services that feel smoother, kinder, and more reliable for millions of users.

✨ Closing Note

This project combines sentiment, storytelling, and statistics to reveal what customers truly feel about Ethiopia’s leading fintech apps. Through data, it offers clarity; through analysis, direction; and through insight, a path toward better digital banking experiences.