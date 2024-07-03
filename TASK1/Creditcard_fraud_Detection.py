import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Here we load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    return data

data = load_data()

# we are using Streamlit python library for UI and can take user input
st.title("Credit Card Fraud Detection")

# Display the data frame
st.write("Data Frame:")
st.write(data.head())

# We check for missing values
st.write("Checking for missing values...")
st.write(data.isnull().sum())

# now split the data into features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train the models
@st.cache_resource
def train_models(X_train, y_train):
    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    return lr_model, dt_model, rf_model

lr_model, dt_model, rf_model = train_models(X_train, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

st.write("### Model Performance")

# Evaluate Logistic Regression
lr_accuracy, lr_report = evaluate_model(lr_model, X_test, y_test)
st.write("**Logistic Regression**")
st.write(f"Accuracy: {lr_accuracy}")
st.write("Classification Report:")
st.text(lr_report)

# Evaluate Decision Tree
dt_accuracy, dt_report = evaluate_model(dt_model, X_test, y_test)
st.write("**Decision Tree**")
st.write(f"Accuracy: {dt_accuracy}")
st.write("Classification Report:")
st.text(dt_report)

# Evaluate Random Forest
rf_accuracy, rf_report = evaluate_model(rf_model, X_test, y_test)
st.write("**Random Forest**")
st.write(f"Accuracy: {rf_accuracy}")
st.write("Classification Report:")
st.text(rf_report)

# User input for prediction 
st.write("### Predict a New Transaction")
input_data = []
for column in X.columns:
    value = st.number_input(f"{column}", format="%f")
    input_data.append(value)

# Convert input data to DataFrame
input_data_df = pd.DataFrame([input_data], columns=X.columns)

# Scale the input data
input_data_scaled = scaler.transform(input_data_df)

# Prediction button
model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
if st.button("Predict"):
    if model_choice == "Logistic Regression":
        prediction = lr_model.predict(input_data_scaled)
    elif model_choice == "Decision Tree":
        prediction = dt_model.predict(input_data_scaled)
    elif model_choice == "Random Forest":
        prediction = rf_model.predict(input_data_scaled)

    result = "Fraudulent" if prediction[0] == 1 else "Legitimate"
    st.write(f"The transaction is: {result}")
