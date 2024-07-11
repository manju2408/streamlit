import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Load the pre-trained model
model = pickle.load(open(r"C:\Users\Xyz\nb.pkl", 'rb'))

# Load the CountVectorizer used for training
with open(r"C:\Users\Xyz\bow.pkl", 'rb') as f:
    bow = pickle.load(f)

st.title("Email Spam/Ham Classifier")

# Input email text
Email = st.text_input("Paste the email here:")
# bow = CountVectorizer(stop_words='english')
# Check if the email input is not empty
if Email:

    data = bow.transform([Email]).toarray()

    spam_ham = model.predict(data)[0]


    if st.button('Submit'):
        st.write("The email is:",spam_ham)