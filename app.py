import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

# Create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Dynamically create input fields for each feature
input_data = {}
for feature in X.columns:
    input_data[feature] = st.text_input(feature)

# Create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    try:
        # Ensure all inputs are provided and are numeric
        features = np.array([float(input_data[feature]) for feature in X.columns]).reshape(1, -1)
        # Make prediction
        prediction = model.predict(features)
        # Display result
        if prediction[0] == 0:
            st.write("Legitimate transaction")
        else:
            st.write("Fraudulent transaction")
    except ValueError:
        st.write("Please enter valid numbers for all features.")
    except Exception as e:
        st.write(f"An error occurred: {e}")

# Display model accuracy
st.write(f"Training Accuracy: {train_acc * 100:.2f}%")
st.write(f"Testing Accuracy: {test_acc * 100:.2f}%")
