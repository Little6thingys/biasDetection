import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle
import train  # Assuming train.py exists and contains the necessary functions/classes

label_vectorizer = train.load_vectorizer("label")
sentiment_vectorizer = train.load_vectorizer("sentiment")
toxic_vectorizer = train.load_vectorizer("toxic")


def mainFunc(text):

    label_classifier = train.load_classifier("label")
    toxic_classifier = train.load_classifier("toxic")
    sentiment_classifier = train.load_classifier("sentiment")


    
    # Predict for the given text
    X_test_phrase_vectorized1 = label_vectorizer.transform([text])
    prediction1 = label_classifier.predict(X_test_phrase_vectorized1)
    X_test_phrase_vectorized2 = toxic_vectorizer.transform([text])
    prediction2 = toxic_classifier.predict(X_test_phrase_vectorized2)
    X_test_phrase_vectorized3 = sentiment_vectorizer.transform([text])
    prediction3 = sentiment_classifier.predict(X_test_phrase_vectorized3)
    confidence1 = np.max(label_classifier.predict_proba(X_test_phrase_vectorized1)) * 100
    confidence2 = np.max(toxic_classifier.predict_proba(X_test_phrase_vectorized2)) * 100
    confidence3 = np.max(sentiment_classifier.predict_proba(X_test_phrase_vectorized3)) * 100
    # Return the predicted label and confidence percentage
    return prediction2, confidence2, prediction3, confidence3, prediction1, confidence1


