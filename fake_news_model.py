# Fake News Detection Model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake.head(5000), true.head(5000)])

# Shuffle dataset
data = data.sample(frac=1).reset_index(drop=True)

# Combine title and text
data["content"] = data["title"] + " " + data["text"]

X = data["content"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict
predictions = model.predict(X_test_vec)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Test custom input
while True:
    news = input("\nEnter news text (or type quit): ")

    if news.lower() == "quit":
        break

    news_vec = vectorizer.transform([news])
    result = model.predict(news_vec)

    if result[0] == 1:
        print("Prediction: REAL NEWS")
    else:
        print("Prediction: FAKE NEWS")