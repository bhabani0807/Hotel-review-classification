import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import re
import string

# Define the text cleaning function
def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9]+", " ", text)
    text = re.sub('[''""…]', '', text)
    return text

# Load the best classifier model
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Load the sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

st.title('Hotel Review Sentiment Analysis')

# Get user input
user_input = st.text_area("Enter a hotel review:")

if st.button('Predict Sentiment'):
    # Clean and embed the input
    cleaned_input = clean_text(user_input)
    embedded_input = sentence_model.encode([cleaned_input])
    
    # Make prediction
    prediction = best_model.predict(embedded_input)
    
    st.write(f"The sentiment of this review is: {prediction[0]}")