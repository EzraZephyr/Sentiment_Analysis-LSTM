import pickle
from nltk.tokenize import word_tokenize
from tensorflow import keras
from sklearn.model_selection import train_test_split

def vocab_builder():
    with open('./preprocessed_data/x.pkl', 'rb') as f:
        x = pickle.load(f)
    with open('./preprocessed_data/y.pkl', 'rb') as f:
        y = pickle.load(f)

    index_to_word, all_words = ['PAD',], []
    # Initialize the vocabulary and word list. Since we need to pad sentences with less than 100 words,
    # the first entry '0' in the vocabulary will be for padding 'PAD'

    i = 0
    for line in x:
        i += 1
        if i % 1000 == 0:
            print(i)
        # Check how far we have traversed

        words = word_tokenize(line)
        # Tokenize each line of text

        all_words.append(words)
        # Add tokenized text to the word list

        for word in words:
            if word not in index_to_word:
                index_to_word.append(word)
                # Build a unique vocabulary to map indices to words

    word_to_index = {word: idx for idx, word in enumerate(index_to_word)}
    # Build a reverse dictionary to map words to indices

    word_count = len(index_to_word)

    corpus_idx = []
    # Used to store the index data of each sentence after conversion

    for line in all_words:
        temp = []
        for word in line:
            temp.append(word_to_index[word])
            # This is the complete indexed data of a piece of text, i.e., a review

        temp = keras.preprocessing.sequence.pad_sequences([temp], maxlen=100, padding='post')
        # maxlen=100 trims reviews longer than 100 words to 100
        # padding='post' pads reviews shorter than 100 words with '0' at the end

        corpus_idx.append(temp[0])
        # Since pad_sequences returns a 2D array, we need to use temp[0] to extract the content
        # Otherwise, corpus_idx would become a 3D array

    label_to_index = {"positive": 1, "negative": 0}
    y_converted = [label_to_index[label] for label in y]
    # Build a dictionary to convert target values to 1 or 0

    X_train, X_test, y_train, y_test = train_test_split(corpus_idx, y_converted, test_size=0.2)
    # Split the data into training and testing sets in an 8:2 ratio

    with open('./preprocessed_data/index_to_word.pkl', 'wb') as f:
        pickle.dump(index_to_word, f)
    with open('./preprocessed_data/word_to_index.pkl', 'wb') as f:
        pickle.dump(word_to_index, f)
    with open('./preprocessed_data/word_count.pkl', 'wb') as f:
        pickle.dump(word_count, f)
    with open('./preprocessed_data/corpus_idx.pkl', 'wb') as f:
        pickle.dump(corpus_idx, f)
    with open('./preprocessed_data/y_converted.pkl', 'wb') as f:
        pickle.dump(y_converted, f)
    with open('./preprocessed_data/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('./preprocessed_data/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('./preprocessed_data/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('./preprocessed_data/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    # Save the data
