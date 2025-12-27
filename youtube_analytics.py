"""
Advanced YouTube Analytics (CSV-based)
Author: Balusupati Dakshina Murthy
"""

# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from gensim import corpora, models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

nltk.download('vader_lexicon')
nltk.download('stopwords')


# =========================
# 2. LOAD DATA
# =========================
DATA_PATH = "data/youtube_data.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully")
print(df.head())


# =========================
# 3. DATA CLEANING
# =========================
df['published_date'] = pd.to_datetime(df['published_date'])
df['view_count'] = pd.to_numeric(df['view_count'])
df['like_count'] = pd.to_numeric(df['like_count'])
df['comment_count'] = pd.to_numeric(df['comment_count'])

print("\nData Types After Cleaning:")
print(df.dtypes)


# =========================
# 4. FEATURE ENGINEERING
# =========================
df['engagement'] = df['like_count'] + df['comment_count']
df['like_ratio'] = df['like_count'] / df['view_count']
df['upload_hour'] = df['published_date'].dt.hour
df['upload_day'] = df['published_date'].dt.day_name()

print("\nFeature Engineering Completed")


# =========================
# 5. TIME SERIES ANALYSIS
# =========================
df_sorted = df.sort_values('published_date')

plt.figure()
plt.plot(df_sorted['published_date'], df_sorted['view_count'])
plt.title("Views Over Time")
plt.xlabel("Published Date")
plt.ylabel("Views")
plt.tight_layout()
plt.savefig("visuals/views_trend.png")
plt.close()


# =========================
# 6. UPLOAD TIME VS VIEWS
# =========================
plt.figure()
sns.boxplot(x='upload_hour', y='view_count', data=df)
plt.title("Upload Hour vs Views")
plt.tight_layout()
plt.savefig("visuals/upload_time_vs_views.png")
plt.close()


# =========================
# 7. SENTIMENT ANALYSIS (TITLES)
# =========================
sia = SentimentIntensityAnalyzer()

df['title_sentiment'] = df['title'].apply(
    lambda x: sia.polarity_scores(str(x))['compound']
)

df['sentiment_label'] = df['title_sentiment'].apply(
    lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral'
)

plt.figure()
df['sentiment_label'].value_counts().plot(kind='bar')
plt.title("Title Sentiment Distribution")
plt.tight_layout()
plt.savefig("visuals/sentiment_distribution.png")
plt.close()


# =========================
# 8. TOPIC MODELING (LDA)
# =========================
stop_words = set(stopwords.words('english'))

texts = [
    [word.lower() for word in title.split() if word.lower() not in stop_words]
    for title in df['title']
]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,
    random_state=42
)

print("\nLDA Topics:")
for topic in lda_model.print_topics():
    print(topic)


# =========================
# 9. CORRELATION ANALYSIS
# =========================
plt.figure()
sns.heatmap(
    df[['view_count', 'like_count', 'comment_count', 'engagement']].corr(),
    annot=True
)
plt.title("Engagement Correlation Heatmap")
plt.tight_layout()
plt.savefig("visuals/correlation_heatmap.png")
plt.close()


# =========================
# 10. ENGAGEMENT PREDICTION
# =========================
X = df[['view_count', 'upload_hour']]
y = df['engagement']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"\nEngagement Prediction R² Score: {score:.2f}")


# =========================
# 11. FINAL MESSAGE
# =========================
print("\nYouTube Analytics Pipeline Completed Successfully")
print("Visuals saved in /visuals folder")
