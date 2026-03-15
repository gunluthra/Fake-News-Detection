import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data["content"] = data["title"] + " " + data["text"]

X = data["content"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Streamlit UI
st.title("Fake News Detection AI")

user_input = st.text_area("Enter news text:")

if st.button("Check News"):
    vec = vectorizer.transform([user_input])
    prediction = model.predict(vec)

    if prediction[0] == 1:
        st.success("This looks like REAL news")
    else:
        st.error("This looks like FAKE news")