# SOCIAL MEDIA SENTIMENT ANALYSIS
# Machine Learning

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# 1. LOAD DATASET

df = pd.read_csv( r"C:\Users\dilip\OneDrive\文件\ML PROJECT DATASETS\Social Media Sentiment Analysis Dataset\social_media_sentimate_data.csv",
    encoding="latin1"
)

print("Dataset loaded successfully!")
print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())


# 2. CHECK COLUMNS

print("\nColumn Names:")
print(df.columns.tolist())


# 3. CHECK MISSING VALUES

print("\nMissing Values:")
print(df[["clean_tweet", "sentiment"]].isnull().sum())


# 4. REMOVE MISSING VALUES

df = df.dropna(subset=["clean_tweet", "sentiment"])


# 5. INPUT AND OUTPUT

X = df["clean_tweet"]
y = df["sentiment"]


# 6. DISPLAY SENTIMENT COUNTS

print("\nSentiment Counts:")
print(y.value_counts())


# 7. SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# 8. CONVERT TEXT INTO NUMERICAL VALUES

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# 9. CREATE MODEL

model = LogisticRegression(
    max_iter=1000
)


# 10. TRAIN MODEL

model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")


# 11. PREDICTION

y_pred = model.predict(X_test_tfidf)


# 12. ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(accuracy * 100, "%")


# 13. CLASSIFICATION REPORT

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 14. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.title("Social Media Sentiment - Confusion Matrix")

plt.show()


# 15. SENTIMENT DISTRIBUTION GRAPH

df["sentiment"].value_counts().plot(
    kind="bar"
)

plt.title("Sentiment Distribution")

plt.xlabel("Sentiment")

plt.ylabel("Number of Tweets")

plt.xticks(rotation=0)

plt.show()


# 16. TEST NEW SOCIAL MEDIA COMMENTS

new_comments = [
    "I really love this amazing technology",
    "This is the worst product ever",
    "This product is okay"
]


# Convert new comments into TF-IDF

new_comments_tfidf = vectorizer.transform(new_comments)


# Predict sentiment

predictions = model.predict(new_comments_tfidf)


# Display predictions

print("\nNew Tweet Sentiment Predictions:")

for comment, prediction in zip(new_comments, predictions):

    print("\nTweet:", comment)

    print("Predicted Sentiment:", prediction)