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

vectorizer = CountVectorizer()


def life_vectorizer(out):
    # Download NLTK resources
    #nltk.download('punkt')
    #nltk.download('stopwords')

    print("Reading csv...")
    # Load the dataset
    df = pd.read_csv("news.csv")
    df.sample(frac=1).reset_index(drop=True)
    df = df[0:100000]
    #
    print("Done reading.")

    # Rename columns
    df = df.rename(columns={"text": "text", out: out})
    df.dropna(inplace=True)

    # Convert 'text' column to string type
    print("Converting text to string...")
    df['text'] = df['text'].astype(str)
    # Define custom stopwords as a list
    custom_stopwords = [
        'when', 'both', 'themselves', 'mightn', 'once', 'or', 'but', 'itself', 'above', 'so', 'hadn', 
        'and', 'an', 'during', 'had', 'at', "that'll", "wouldn't", 'which', 'o', 'those', 'against', 
        'down', 'she', 'aren', 'we', 'own', 'was', 'any', 'him', 'same', "isn't", "shan't", 'ourselves', 
        'about', 'their', 'such', "wasn't", "shouldn't", 'below', 'in', 'wasn', "it's", 'as', 't', 
        "haven't", 'what', 'hasn', 'off', 'why', 'how', 'after', 'to', "hasn't", 'am', 'too', 'theirs', 
        "mustn't", 'into', 'a', 'its', "doesn't", 'than', "needn't", "mightn't", 'i', 'now', "couldn't", 
        'our', 'through', "she's", 'who', 'won', "you'll", 'her', 'with', 'there', 're', 'himself', 'up', 
        'nor', 'until', 'over', 'should', 'most', 'having', 'more', 'll', 'yourself', 'his', 'ain', 'my', 
        'been', 'these', "should've", 'd', 'whom', 'ma', 'because', 'from', "weren't", 'further', 'your', 
        'myself', 'be', 'don', 'herself', 'they', 'if', 'where', 's', 'only', 'not', 'doesn', 'yours', 
        'few', 'some', "didn't", 'can', 'isn', 'you', 'are', 'were', 'being', 'mustn', 'couldn', 'before', 
        "you've", 'by', 'does', 'do', 'have', 'needn', 'for', 'very', 'while', 'did', 'this', 'will', 'no', 
        'all', 'that', "you're", 'is', 'ours', "aren't", "hadn't", 'wouldn', 'out', 'the', 'yours', 'didn', 
        "don't", 'of', 'y', 'again', 'it', 'here', 'haven', 'between', 'shouldn', 'them', 'under', 'me', 
        "you'd", 've', 'each', 'hers', 'm', 'has', 'doing', 'on', 'other', 'just', 'weren', 'he', "won't", 
        'then', 'shan'
    ]

    # Define stemmer
    stemmer = PorterStemmer()

    # Preprocessing function
    def preprocess_text(text):
        tokens = word_tokenize(text)

        tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in custom_stopwords]
        return ' '.join(tokens)


    # Apply preprocessing to the text column
    df['text'] = df['text'].apply(preprocess_text)
    # Split data into features and labels
    X = df['text']
    y = df[out]

    # Vectorize the text data
    X_vectorized = vectorizer.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)   
    #print(X_train.length) 

    return vectorizer

def life(out):
    # Download NLTK resources
    #nltk.download('punkt')
    #nltk.download('stopwords')

    print("Reading csv...")
    # Load the dataset
    df = pd.read_csv("news.csv")
    df.sample(frac=1).reset_index(drop=True)
    df = df[0:100000]
    #
    print("Done reading.")

    # Rename columns
    df = df.rename(columns={"text": "text", out: out})
    df.dropna(inplace=True)

    # Convert 'text' column to string type
    print("Converting text to string...")
    df['text'] = df['text'].astype(str)
    # Define custom stopwords as a list
    custom_stopwords = [
        'when', 'both', 'themselves', 'mightn', 'once', 'or', 'but', 'itself', 'above', 'so', 'hadn', 
        'and', 'an', 'during', 'had', 'at', "that'll", "wouldn't", 'which', 'o', 'those', 'against', 
        'down', 'she', 'aren', 'we', 'own', 'was', 'any', 'him', 'same', "isn't", "shan't", 'ourselves', 
        'about', 'their', 'such', "wasn't", "shouldn't", 'below', 'in', 'wasn', "it's", 'as', 't', 
        "haven't", 'what', 'hasn', 'off', 'why', 'how', 'after', 'to', "hasn't", 'am', 'too', 'theirs', 
        "mustn't", 'into', 'a', 'its', "doesn't", 'than', "needn't", "mightn't", 'i', 'now', "couldn't", 
        'our', 'through', "she's", 'who', 'won', "you'll", 'her', 'with', 'there', 're', 'himself', 'up', 
        'nor', 'until', 'over', 'should', 'most', 'having', 'more', 'll', 'yourself', 'his', 'ain', 'my', 
        'been', 'these', "should've", 'd', 'whom', 'ma', 'because', 'from', "weren't", 'further', 'your', 
        'myself', 'be', 'don', 'herself', 'they', 'if', 'where', 's', 'only', 'not', 'doesn', 'yours', 
        'few', 'some', "didn't", 'can', 'isn', 'you', 'are', 'were', 'being', 'mustn', 'couldn', 'before', 
        "you've", 'by', 'does', 'do', 'have', 'needn', 'for', 'very', 'while', 'did', 'this', 'will', 'no', 
        'all', 'that', "you're", 'is', 'ours', "aren't", "hadn't", 'wouldn', 'out', 'the', 'yours', 'didn', 
        "don't", 'of', 'y', 'again', 'it', 'here', 'haven', 'between', 'shouldn', 'them', 'under', 'me', 
        "you'd", 've', 'each', 'hers', 'm', 'has', 'doing', 'on', 'other', 'just', 'weren', 'he', "won't", 
        'then', 'shan'
    ]

    # Define stemmer
    stemmer = PorterStemmer()

    # Preprocessing function
    def preprocess_text(text):
        tokens = word_tokenize(text)

        tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in custom_stopwords]
        return ' '.join(tokens)


    # Apply preprocessing to the text column
    df['text'] = df['text'].apply(preprocess_text)
    # Split data into features and labels
    X = df['text']
    y = df[out]

    # Vectorize the text data
    X_vectorized = vectorizer.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)   
    #print(X_train.length)  


    print("Initializing classifier...")
    # Train a Multinomial Naive Bayes classifier
    classifier = MultinomialNB()
    print("Fitting...")
    classifier.fit(X_train, y_train)
    print("Done with " + out + ". ")

    return classifier
    
def train_classifiers():
    out = ["label", "sentiment", "toxic"]
    for o in out:
        print("Starting training classifier for " + o + ". ")
        classifier = life(o)
        with open(o + ".pickle","wb") as f:
            pickle.dump(classifier, f)

def load_classifier(out):
    with open(out + ".pickle","rb") as f:
        return pickle.load(f)


def load_vectorizer(out):
    with open(out + "vectorizer.pickle","rb") as f:
        return pickle.load(f)

if __name__ == "__main__": 
    train_classifiers()