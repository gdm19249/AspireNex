import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from local spam.csv file
try:
    data = pd.read_csv("spam.csv", encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv("spam.csv", encoding='ISO-8859-1')

# Assuming the columns are 'v1' for label and 'v2' for message
data = data.rename(columns={'v1': 'label', 'v2': 'message'})

# Encode the labels (ham = 0, spam = 1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Naive Bayes Classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_predictions = nb_model.predict(X_test_tfidf)

# Logistic Regression Classifier
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
lr_predictions = lr_model.predict(X_test_tfidf)

# Support Vector Machine Classifier
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
svm_predictions = svm_model.predict(X_test_tfidf)

# we are using Streamlit python library for UI and can take user input
st.title('SMS Spam Detection')

# Model Evaluation
st.header("Model Evaluation")
st.subheader("Naive Bayes")
st.text(f"Accuracy: {accuracy_score(y_test, nb_predictions)}")
st.text(classification_report(y_test, nb_predictions))

st.subheader("Logistic Regression")
st.text(f"Accuracy: {accuracy_score(y_test, lr_predictions)}")
st.text(classification_report(y_test, lr_predictions))

st.subheader("Support Vector Machine")
st.text(f"Accuracy: {accuracy_score(y_test, svm_predictions)}")
st.text(classification_report(y_test, svm_predictions))

# Function to predict new messages
def predict_message(model, message):
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)
    return "spam" if prediction[0] == 1 else "ham"

# Here we can take user input
st.header("Classify New Message")
message = st.text_area("Enter a message to classify")

model_choice = st.selectbox("Choose a model", ("Naive Bayes", "Logistic Regression", "SVM"))

if st.button("Classify"):
    if model_choice == "Naive Bayes":
        result = predict_message(nb_model, message)
    elif model_choice == "Logistic Regression":
        result = predict_message(lr_model, message)
    elif model_choice == "SVM":
        result = predict_message(svm_model, message)
    st.write(f"The message is classified as: {result}")

