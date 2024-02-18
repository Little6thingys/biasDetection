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

# Download NLTK resources
#nltk.download('punkt')
#nltk.download('stopwords')
vectorizer = CountVectorizer()

# Load the dataset
df = pd.read_csv("news.csv")

# Rename columns
df = df.rename(columns={"text": "text", "label": "label"})
df.dropna(inplace=True)

# Convert 'text' column to string type
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
y = df['label']

# Vectorize the text data
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)     

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
#accuracy = accuracy_score(y_test, classifier.predict(X_test))
#print(accuracy)
def mainFunc(text):

    
    # Predict for the given text
    X_test_phrase_vectorized = vectorizer.transform([text])
    prediction = classifier.predict(X_test_phrase_vectorized)
    confidence = np.max(classifier.predict_proba(X_test_phrase_vectorized)) * 100
    
    # Return the predicted label and confidence percentage
    return prediction[0], confidence
# Test the function with a sample text
#result, confidence, unique = mainFunc("News plays a pivotal role in our society, serving as a vital source of information that keeps people informed about current events and developments both locally and globally. It serves as a platform for disseminating factual information, providing insights into various issues, and fostering public discourse. However, the landscape of news media is diverse, encompassing a wide range of sources with varying perspectives and agendas. While some news outlets strive to uphold principles of objectivity and impartiality, others may be influenced by commercial interests, political affiliations, or ideological biases. As consumers of news, it is crucial to critically evaluate the credibility and reliability of sources, seek out diverse viewpoints, and remain vigilant against misinformation and sensationalism. Ultimately, a well-informed populace is essential for a functioning democracy, and the responsible consumption of news is integral to fostering informed citizenship and promoting societal progress.")
#print("Predicted label:", result)
#print("Confidence percentage:", confidence)
#print(unique)
