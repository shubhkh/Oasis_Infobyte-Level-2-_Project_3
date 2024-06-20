import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv("creditcard.csv")

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Sample legitimate transactions to balance the dataset
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Prepare features and target
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=41)

# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate model
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Web app
st.title("Credit Card Fraud Detection Model")
input_df = st.text_input("Enter all required feature values separated by commas")

submit = st.button("Submit")

if submit:
    input_df_splited = input_df.split(',')
    try:
        features = np.asarray(input_df_splited, dtype=float)  # Changed from np.float to float
        prediction = model.predict(features.reshape(1, -1))

        if prediction[0] == 0:
            st.write("Legal Transaction")
        else:
            st.write("Fraud Transaction")
    except ValueError:
        st.write("Please enter valid feature values")
        
