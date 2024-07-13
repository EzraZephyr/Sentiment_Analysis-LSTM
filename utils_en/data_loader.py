import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    # Clean out HTML tags

    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Get rid of all non-letter elements

    text = text.lower()
    # Convert everything to lowercase

    words = word_tokenize(text)
    # Tokenize the text

    lemmatizer = WordNetLemmatizer()
    # Initialize the lemmatizer

    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    # Lemmatize all words and filter out stopwords, keep the rest
    # Stopwords are words that don't carry much meaning and appear frequently, like 'the', 'is', 'a'

    text = ' '.join(words)
    # Rejoin the processed words

    return text

def dataloader():
    filename = './data/IMDB Dataset.csv'
    data = pd.read_csv(filename)
    # Use pandas to load the data into a DataFrame for easier handling later

    x = data['review']
    y = data['sentiment']
    # Extract reviews and sentiment labels

    for i in range(len(x)):
        x[i] = clean_text(x[i])

        if i % 1000 == 0:
            print(i)
        # To easily monitor the data processing progress

    with open('preprocessed_data/x.pkl', 'wb') as f:
        pickle.dump(x, f)
    with open('preprocessed_data/y.pkl', 'wb') as f:
        pickle.dump(y, f)
    # Serialize the processed data into binary files for later use
